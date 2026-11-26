"""Web: serves the dream page, runs each turn (speech -> director -> scene + voice),
and watches the heartbeat of every other service."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect, WebSocketState

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("web")
logging.getLogger("httpx").setLevel(logging.WARNING)


class _QuietHealth(logging.Filter):
    """Heartbeats arrive every few seconds; keep them out of the access log."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "/healthz" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(_QuietHealth())

DIRECTOR_URL = os.environ.get("DIRECTOR_URL", "http://director:9010").rstrip("/")
VOICE_URL = os.environ.get("VOICE_URL", "http://voice:9020").rstrip("/")
IMAGERY_URL = os.environ.get("IMAGERY_URL", "").rstrip("/")
# The quiet minute: a few seconds after the dreamer has spoken, the storyteller's director
# decides what happens next by itself; the page asks again when it has acted that out.
DIRECT_AFTER_S = float(os.environ.get("DIRECT_AFTER_S", "3"))
DIRECT_EVERY_S = float(os.environ.get("DIRECT_EVERY_S", "40"))     # at most this often
DIRECT_IDLE_MAX_S = float(os.environ.get("DIRECT_IDLE_MAX_S", "600"))  # then the dream rests
HEARTBEAT_S = float(os.environ.get("HEARTBEAT_S", "3"))
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
STARTED = time.time()

class NotReady(Exception):
    """A service is still waking up (downloading or loading its model)."""


# A dream's id as the storyteller accepts it; a page that sends anything else gets a new one.
SESSION_ID = re.compile(r"[A-Za-z0-9_-]{6,64}")
GENDER = {"male": 1, "female": 2}
MAX_CHARS = 1500          # characters of the player's words that one turn takes


_SENTENCE_END = re.compile(r"[.!?…]+[\"'”»)]*\s+")


HEBREW_LETTER = re.compile(r"[\u0590-\u05ff]")
LATIN_LETTER = re.compile(r"[A-Za-z]")


def written_in(text: str, default: str) -> str:
    """The language a text is written in, judged by its letters; the voice must read
    Hebrew words with the Hebrew voice even when the storyteller slipped languages."""
    he, en = len(HEBREW_LETTER.findall(text or "")), len(LATIN_LETTER.findall(text or ""))
    if he == en == 0:
        return default
    return "he" if he >= en else "en"


def split_first_sentence(text: str, min_first: int = 25, min_rest: int = 15) -> list[str]:
    """The reply in one or two parts: the first sentence that is long enough to stand on its
    own ("Snow..." alone is too short), then everything after it."""
    for m in _SENTENCE_END.finditer(text):
        head, tail = text[:m.end()].strip(), text[m.end():].strip()
        if len(head) >= min_first:
            return [head, tail] if len(tail) >= min_rest else [text]
    return [text]


_CLAUSE_END = re.compile(r"[,;:]\s+|\s[-–—]\s+")


def split_for_speech(text: str, long_first: int = 45, min_clause: int = 12,
                     min_rest: int = 15) -> list[str]:
    """Clips to synthesize one after another. A long first sentence is cut at its first
    comma as well, so the dreamer starts talking sooner; the rest is synthesized while
    the first clip plays."""
    parts = split_first_sentence(text)
    head = parts[0]
    if len(head) > long_first:
        for m in _CLAUSE_END.finditer(head):
            a, b = head[:m.end()].strip(), head[m.end():].strip()
            if len(a) >= min_clause:
                if len(b) >= min_rest:
                    return [a, b] + parts[1:]
                break
    return parts


http = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=3.0))


# --------------------------------------------------------------------------- media cache
class MediaCache:
    """Short-lived audio clips and images, served to the page by id."""

    def __init__(self, size: int = 96):
        self.items: OrderedDict[str, tuple[bytes, str]] = OrderedDict()
        self.size = size

    def put(self, data: bytes, mime: str, ext: str) -> str:
        key = f"{uuid.uuid4().hex}.{ext}"
        self.items[key] = (data, mime)
        while len(self.items) > self.size:
            self.items.popitem(last=False)
        return f"/media/{key}"

    def get(self, key: str):
        return self.items.get(key)


media = MediaCache()
pictures: dict[tuple[str, str], str] = {}

# Things that already look right as procedural particles (groves, swarms, skies).
SPRITE_SKIP = {"fireflies", "tree", "flower", "mushroom", "rock", "cloud", "star", "rainbow", "water", "fire"}

# The place as the painter is asked for it, when the storyteller gave no picture of its own.
PLACE_WORDS = {
    "void": "", "forest": "a deep old forest", "meadow": "a wide meadow full of flowers",
    "desert": "rolling desert dunes", "ocean": "the open ocean, waves to the horizon",
    "mountains": "high mountains and valleys", "cave": "a vast glowing cave", "city": "a quiet city",
    "space": "outer space, nebulae and stars", "moon": "the grey surface of the moon, black starry sky",
    "clouds": "above a sea of clouds", "snow": "a snowy land", "underwater": "deep under the sea, rays of light",
}
TIME_WORDS = {"night": "at night", "dawn": "at dawn", "day": "in daylight", "dusk": "at dusk"}
WEATHER_WORDS = {"mist": "misty", "rain": "rain falling", "snow": "snow falling", "storm": "a storm"}


def place_prompt(setting: dict, backdrop: str = "") -> str:
    """What the painter is asked to paint around the dreamer ("" for the black void)."""
    if backdrop.strip():
        return backdrop.strip()
    words = PLACE_WORDS.get(setting.get("kind") or "void", setting.get("kind") or "")
    if not words:
        return ""
    more = [TIME_WORDS.get(setting.get("time") or ""), WEATHER_WORDS.get(setting.get("weather") or "")]
    return ", ".join([words] + [m for m in more if m])


def describe(label: str | None, kind: str, color: str | None) -> str:
    text = (label or kind).strip()
    if color and color.lower() not in text.lower() and not color.startswith("#"):
        text = f"{color} {text}"
    return text


# --------------------------------------------------------------------------- heartbeats
class Monitor:
    def __init__(self):
        self.targets = {"director": DIRECTOR_URL, "voice": VOICE_URL}
        if IMAGERY_URL:
            self.targets["imagery"] = IMAGERY_URL
        self.status: dict[str, dict] = {"web": {"status": "ok", "detail": "serving the dream"}}
        for name in self.targets:
            self.status[name] = {"status": "down", "detail": "no heartbeat yet"}
        if not IMAGERY_URL:
            self.status["imagery"] = {"status": "off", "detail": "image generation not configured"}

    def ok(self, name: str) -> bool:
        return self.status.get(name, {}).get("status") in ("ok", "degraded")

    def ready(self, part: str) -> bool:
        """director: the storyteller can answer; stt / tts: hearing / speaking works."""
        if part == "director":
            return self.status.get("director", {}).get("status") == "ok"
        voice = self.status.get("voice", {})
        return (voice.get(part) or {}).get("state") == "ready"

    async def _probe(self, name: str, url: str) -> dict:
        t0 = time.perf_counter()
        try:
            r = await http.get(f"{url}/healthz", timeout=2.5)
            r.raise_for_status()
            body = r.json()
            body["latency_ms"] = round(1000 * (time.perf_counter() - t0))
            body["last_seen"] = time.time()
            return body
        except Exception as e:  # noqa: BLE001
            prev = self.status.get(name, {})
            return {"status": "down", "detail": f"no heartbeat ({type(e).__name__})",
                    "last_seen": prev.get("last_seen")}

    async def run(self):
        while True:
            names = list(self.targets)
            results = await asyncio.gather(*(self._probe(n, self.targets[n]) for n in names))
            changed = False
            for name, res in zip(names, results):
                old = self.status.get(name, {})
                if (old.get("status"), old.get("detail"), _sub(old), old.get("progress")) != \
                        (res.get("status"), res.get("detail"), _sub(res), res.get("progress")):
                    changed = True
                self.status[name] = res
            self.status["web"] = {"status": "ok", "detail": f"{len(clients)} dreamer(s) connected",
                                  "uptime_s": round(time.time() - STARTED)}
            if changed:
                await broadcast({"type": "status", "services": self.snapshot()})
            for c in list(clients):
                c.kick()
            await asyncio.sleep(HEARTBEAT_S)

    def snapshot(self) -> dict:
        return {k: {kk: v.get(kk) for kk in ("status", "detail", "progress", "stt", "tts", "model",
                                             "tokens_per_s", "last_latency_s", "workers")
                    if v.get(kk) is not None}
                for k, v in self.status.items()}


def _sub(d: dict):
    return json.dumps([d.get("stt"), d.get("tts"), d.get("workers")], sort_keys=True, default=str)


monitor = Monitor()
clients: set["Dreamer"] = set()


async def broadcast(msg: dict):
    for c in list(clients):
        await c.send(msg)


async def tell_writing():
    """Every open page draws fewer frames while the storyteller writes anyone's answer: the
    model and the pages share one graphics card."""
    await broadcast({"type": "writing", "on": any(c.busy for c in clients)})


# --------------------------------------------------------------------------- one connected player
class Dreamer:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.session = uuid.uuid4().hex     # until the page names its own dream
        self.lang = "he"
        self.voice = "neutral"
        self.listener = ""                  # the player's gender as the dreamer speaks to them
        self.last_reply = ""
        self.busy = False
        self.audio_meta: dict | None = None
        self.self_key = ""
        self.place_key = ""
        self.pending: dict | None = None    # the player's last words, waiting for a service to wake up
        self.next_up = False                # ...or for the dreamer to finish its answer
        self.clip_log: list[tuple] = []     # (seconds since speak() began, synthesis seconds)
        self.directing: asyncio.Task | None = None
        self.directed_at = 0.0
        self.last_turn_at = time.time()

    async def send(self, msg: dict):
        if self.ws.client_state != WebSocketState.CONNECTED:
            return
        try:
            await self.ws.send_json(msg)
        except Exception:  # noqa: BLE001 - the page went away mid-send
            pass

    async def stage(self, name: str):
        await self.send({"type": "stage", "stage": name})

    # ----------------------------------------------------------------- session
    async def hello(self, data: dict):
        sid = data.get("session")
        # The page keeps the id the "session" answer carries, so a bad one is replaced once.
        self.session = sid if isinstance(sid, str) and SESSION_ID.fullmatch(sid) else uuid.uuid4().hex
        self.lang = data.get("lang") if data.get("lang") in ("he", "en") else "he"
        await self._open("/session")

    async def reset(self, data: dict):
        if data.get("lang") in ("he", "en"):
            self.lang = data["lang"]
        await self._open("/reset")

    async def _open(self, path: str):
        try:
            r = await http.post(f"{DIRECTOR_URL}{path}",
                                json={"session_id": self.session, "lang": self.lang}, timeout=10)
            r.raise_for_status()
            d = r.json()
        except Exception as e:  # noqa: BLE001
            log.warning("director unavailable for %s: %s", path, e)
            await self.send({"type": "session", "session": self.session, "resumed": False,
                             "offline": True, "scene": None, "lang": self.lang})
            return
        self.voice = d.get("voice", "neutral")
        self.listener = d.get("listener", "")
        self.lang = d.get("lang", self.lang)
        self.last_reply = d.get("reply", "")
        await self.send({"type": "session", "session": self.session, "resumed": d.get("resumed"),
                         "scene": d.get("scene"), "lang": self.lang, "reply": self.last_reply,
                         "facts": d.get("facts", [])})
        if d.get("scene"):
            self.request_pictures(d["scene"], [], everything=True)
            if d.get("resumed"):
                self.direct_soon(6)          # a dream picked up again comes alive by itself
        if not d.get("resumed") and self.last_reply:
            await self.speak(self.last_reply, self.lang)

    # ----------------------------------------------------------------- the turn pipeline
    async def wait_for(self, part: str, turn: dict):
        """Keep the player's words and tell the page what we are waiting for. The heartbeat
        loop runs the turn as soon as the service is ready (see kick)."""
        self.pending, self.next_up = turn, False
        await self.send({"type": "waiting", "for": part, "services": monitor.snapshot()})

    def kick(self):
        p = self.pending
        if not p or self.busy:
            return
        if monitor.ready("stt" if p.get("audio") is not None else "director"):
            self.pending = None
            tries = p.pop("tries", 0) + 1
            if tries > 20:                  # the service says ready but keeps refusing
                asyncio.create_task(self.send({"type": "notice", "key": "stumbled",
                                               "detail": "a service kept refusing the turn"}))
                return
            asyncio.create_task(self.turn(**p, tries=tries))

    async def turn(self, text: str | None = None, audio: bytes | None = None, mime: str = "",
                   lang_hint: str | None = None, tries: int = 0):
        if self.busy:
            # The dreamer is still answering: these words come next (typed words add up).
            p = self.pending
            if audio is None and p and p.get("audio") is None and p.get("text"):
                text = f"{p['text']} {text or ''}"
            if text:
                text = text.strip()[:MAX_CHARS]
            self.pending = {"text": text, "audio": audio, "mime": mime, "lang_hint": lang_hint,
                            "tries": tries}
            self.next_up = True
            await self.send({"type": "notice", "key": "busy"})
            return
        if audio is not None and not monitor.ready("stt"):
            return await self.wait_for("stt", {"audio": audio, "mime": mime, "lang_hint": lang_hint,
                                               "tries": tries})
        self.busy = True
        self.stop_directing()               # the player's words come first
        await tell_writing()
        self.last_turn_at = time.time()
        t0 = time.perf_counter()
        timings = {}
        self.clip_log = []
        try:
            if audio is not None:
                await self.stage("hearing")
                try:
                    text = await self.transcribe(audio, mime, lang_hint)
                except NotReady:
                    return await self.wait_for("stt", {"audio": audio, "mime": mime, "lang_hint": lang_hint,
                                                       "tries": tries})
                timings["stt_s"] = round(time.perf_counter() - t0, 2)
                if text is None:
                    await self.send({"type": "notice", "key": "stt_down"})
                    return
                if not text:
                    await self.send({"type": "notice", "key": "not_heard"})
                    return
            text = (text or "").strip()[:MAX_CHARS]
            if not text:
                return
            await self.send({"type": "transcript", "text": text})
            if not monitor.ready("director"):
                return await self.wait_for("director", {"text": text, "lang_hint": lang_hint, "tries": tries})
            await self.stage("dreaming")
            t1 = time.perf_counter()
            early, first, final, speaking, spoke_at = None, None, None, None, 0.0
            try:
                async for ev in self.director_events(text, lang_hint):
                    kind = ev.get("type")
                    if kind == "heard":
                        # Words the dream left out fade from the player's line as well.
                        await self.send({"type": "transcript", "text": ev.get("text", "")})
                    elif kind in ("first", "reply") and early is None:
                        # The dreamer's words while the storyteller is still writing: its first
                        # sentence as soon as it is complete, then the rest.
                        self.lang = ev.get("lang", self.lang)
                        self.voice = ev.get("voice") or self.voice
                        self.listener = ev.get("listener", self.listener)
                        words = ev["text"] if kind == "first" else ev["reply"]
                        if kind == "first" and first is not None:
                            continue
                        self.last_reply = words
                        await self.send({"type": "reply", "text": words, "lang": self.lang})
                        if first is None:
                            timings["first_words_s"] = round(time.perf_counter() - t1, 2)
                            spoke_at = time.perf_counter()
                            speaking = asyncio.create_task(self.speak(words, self.lang))
                        elif words.startswith(first):
                            rest = words[len(first):].strip()
                            if rest:
                                speaking = asyncio.create_task(
                                    self._then_speak(speaking, rest, self.lang))
                        else:
                            log.warning("the reply does not start with its spoken first sentence")
                        if kind == "first":
                            first = words
                        else:
                            early = ev
                    elif kind == "final":
                        final = ev
            except NotReady:
                if first is None:
                    return await self.wait_for("director", {"text": text, "lang_hint": lang_hint,
                                                            "tries": tries})
                final = {"fallback": True, "error": "the storyteller went away mid-answer"}
            if early is None and first is not None:
                early = {"reply": first}
            timings["director_s"] = round(time.perf_counter() - t1, 2)
            d = final or {"fallback": True, "error": "the storyteller sent no answer"}
            if d.get("fallback") and early is None:
                # The storyteller failed twice; the story is unchanged, so the player can say it again.
                await self.send({"type": "notice", "key": "stumbled", "detail": d.get("error", "")})
                return
            if not d.get("fallback"):
                lang = d.get("lang", self.lang)
                self.lang, self.voice = lang, d.get("voice", self.voice)
                self.listener = d.get("listener", self.listener)
                if d.get("scene"):
                    await self.send({"type": "scene", "scene": d["scene"], "added": d.get("added", []),
                                     "changed": d.get("changed", []), "removed": d.get("removed", []),
                                     "effect": d.get("effect", "none"), "ending": d.get("ending", False)})
                full = d.get("reply", "")
                if early is None:
                    self.last_reply = full
                    await self.send({"type": "reply", "text": full, "lang": lang})
                    spoke_at = time.perf_counter()
                    speaking = asyncio.create_task(self.speak(full, lang))
                elif full != early["reply"] and full.startswith(early["reply"]):
                    # The grand finale adds a closing line after the words already spoken.
                    self.last_reply = full
                    await self.send({"type": "reply", "text": full, "lang": lang})
                    speaking = asyncio.create_task(
                        self._then_speak(speaking, full[len(early["reply"]):].strip(), lang))
                if d.get("scene"):
                    self.request_pictures(d["scene"], d.get("added", []), d.get("backdrop", ""))
            first_sound = await speaking if speaking else None
            if first_sound is not None:
                timings["tts_s"] = round(first_sound, 2)
                timings["to_first_sound_s"] = round(spoke_at - t0 + first_sound, 2)
            timings["total_s"] = round(time.perf_counter() - t0, 2)
            timings["clips"] = self.clip_log[-8:]
            await self.send({"type": "timings", **timings})
            if not d.get("fallback") and not d.get("ending"):
                self.direct_soon(DIRECT_AFTER_S)
        finally:
            self.busy = False
            await tell_writing()
            await self.stage("idle")
            if self.next_up:                # words that arrived meanwhile go next, at once
                self.next_up = False
                self.kick()

    # ----------------------------------------------------------------- the quiet minute
    def stop_directing(self):
        if self.directing and not self.directing.done():
            self.directing.cancel()
        self.directing = None

    def direct_soon(self, delay: float):
        self.stop_directing()
        self.directing = asyncio.create_task(self._direct(delay))

    def more(self):
        """The page has acted out what it was given and the player is still quiet."""
        idle = time.time() - self.last_turn_at
        if (self.busy or self.pending or time.time() - self.directed_at < DIRECT_EVERY_S
                or idle > DIRECT_IDLE_MAX_S or (self.directing and not self.directing.done())):
            return
        self.direct_soon(0)

    async def _direct(self, delay: float):
        await asyncio.sleep(delay)
        if self.busy or not monitor.ready("director"):
            return
        self.directed_at = time.time()
        try:
            r = await http.post(f"{DIRECTOR_URL}/direct", timeout=60,
                                json={"session_id": self.session, "pictures": monitor.ok("imagery")})
            if r.status_code != 200:
                return                      # interrupted by the player, or nothing to do
            d = r.json()
        except Exception as e:  # noqa: BLE001 - a quiet minute can always be skipped
            log.info("director's minute skipped: %s", e)
            return
        if d.get("next") or d.get("quest") or d.get("fulfilled"):
            await self.send({"type": "beats", "next": d.get("next", []), "quest": d.get("shown") or "",
                             "progress": d.get("progress", 0), "steps": d.get("steps", 6),
                             "fulfilled": bool(d.get("fulfilled"))})

    async def transcribe(self, audio: bytes, mime: str, lang_hint: str | None) -> str | None:
        """The words heard, "" for silence, None when hearing failed."""
        files = {"audio": ("speech.webm", audio, mime or "application/octet-stream")}
        form = {"lang": lang_hint or "auto", "prompt": self.last_reply[:200]}
        try:
            r = await http.post(f"{VOICE_URL}/stt", files=files, data=form, timeout=60)
        except Exception as e:  # noqa: BLE001
            log.warning("speech-to-text unreachable: %s", e)
            return None
        if r.status_code == 503:
            raise NotReady()
        if r.status_code == 422:            # a recording with no decodable sound
            log.info("speech-to-text could not decode the recording: %s", r.text[:200])
            return ""
        if r.status_code != 200:
            log.warning("speech-to-text failed: %s %s", r.status_code, r.text[:300])
            return None
        return (r.json().get("text") or "").strip()

    async def _then_speak(self, before, text: str, lang: str):
        first = await before if before else None
        await self.speak(text, lang)
        return first

    async def director_events(self, text: str, lang_hint: str | None):
        """The storyteller's answer as it streams: {"type": "reply"} early, then "final".
        Raises NotReady when the model is still loading (nothing has been recorded then)."""
        body = {"session_id": self.session, "text": text, "lang": lang_hint,
                "pictures": monitor.ok("imagery")}
        said = False
        try:
            async with http.stream("POST", f"{DIRECTOR_URL}/turn/stream", json=body, timeout=200) as r:
                if r.status_code == 503:
                    raise NotReady()
                if r.status_code != 200:
                    log.warning("director failed: %s %s", r.status_code, (await r.aread())[:300])
                    yield {"type": "final", "fallback": True, "error": f"storyteller error {r.status_code}"}
                    return
                async for line in r.aiter_lines():
                    if line.strip():
                        ev = json.loads(line)
                        said = said or ev.get("type") != "heard"    # the dreamer has begun
                        yield ev
        except NotReady:
            raise
        except httpx.TransportError as e:
            if not said:
                # The storyteller went away before a word (a restart, say). It records a turn
                # only once it is complete, so the player's words wait for it to come back.
                log.warning("director went away mid-turn (%s); waiting for it", type(e).__name__)
                monitor.status["director"] = {**monitor.status.get("director", {}),
                                              "status": "down", "detail": "restarting"}
                raise NotReady() from e
            log.warning("director stream broke: %s: %s", type(e).__name__, e)
            yield {"type": "final", "fallback": True,
                   "error": f"storyteller unreachable ({type(e).__name__})"}
        except Exception as e:  # noqa: BLE001
            log.warning("director stream broke: %s: %s", type(e).__name__, e)
            yield {"type": "final", "fallback": True,
                   "error": f"storyteller unreachable ({type(e).__name__})"}

    async def speak(self, text: str, lang: str) -> float | None:
        """Voice the reply. The first sentence is sent on its own so the dreamer starts
        talking while the rest is still being synthesized. Returns seconds to first sound."""
        if not text:
            return None
        lang = written_in(text, lang)
        if not monitor.ready("tts"):
            await self.send({"type": "speak", "text": text, "lang": lang, "voice": self.voice})
            return None
        t0 = time.perf_counter()
        first = None
        for part in split_for_speech(text):
            try:
                # Hebrew reads gendered words by who speaks and who listens (1 male, 2 female).
                body = {"text": part, "lang": lang, "voice": self.voice,
                        "speaker": GENDER.get(self.voice, 0), "listener": GENDER.get(self.listener, 0)}
                try:
                    r = await http.post(f"{VOICE_URL}/tts", json=body, timeout=60)
                except httpx.TransportError:    # a pooled connection the server had closed
                    r = await http.post(f"{VOICE_URL}/tts", json=body, timeout=60)
                r.raise_for_status()
                url = media.put(r.content, "audio/wav", "wav")
                await self.send({"type": "audio", "url": url, "text": part})
                first = first if first is not None else time.perf_counter() - t0
                self.clip_log.append((round(time.perf_counter() - t0, 2),
                                      r.headers.get("X-Synth-Seconds", "")))
            except Exception as e:  # noqa: BLE001 - the browser speaks instead
                log.warning("text-to-speech failed (%s: %s); browser voice fallback",
                            type(e).__name__, e)
                await self.send({"type": "speak", "text": part, "lang": lang, "voice": self.voice})
        return first

    def request_pictures(self, scene: dict, added: list, backdrop: str = "", everything: bool = False):
        """With the imagery service up, every new thing, the dreamer's new form and every new
        place get painted; the page turns each painting into brush strokes in space."""
        if not (IMAGERY_URL and monitor.ok("imagery")):
            return
        for t in scene.get("things", []):
            if (everything or t["id"] in added) and t.get("count", 1) <= 3 and t["kind"] not in SPRITE_SKIP:
                asyncio.create_task(self.paint(describe(t.get("label"), t["kind"], t.get("color")), "thing",
                                               t["id"], 0 if t["id"] in added else 1))
        me = scene.get("self") or {}
        key = f"{me.get('form')}|{me.get('label')}|{me.get('color')}"
        if me.get("form") not in (None, "wisp") and (everything or key != self.self_key):
            asyncio.create_task(self.paint(describe(me.get("label"), me["form"], me.get("color")), "thing", "self", 1))
        self.self_key = key
        setting = scene.get("setting") or {}
        place = f"{setting.get('kind')}|{setting.get('time')}|{setting.get('weather')}"
        if everything or backdrop.strip() or place != self.place_key:
            prompt = place_prompt(setting, backdrop)
            if prompt:
                asyncio.create_task(self.paint(prompt, "scene", "", 2, setting.get("kind") or ""))
        self.place_key = place

    async def paint(self, prompt: str, kind: str, thing_id: str = "", priority: int = 1, place: str = ""):
        """One painting from the imagery service, sent to the page as a picture to paint
        with (an older imagery service without /paint sends a plain picture instead)."""
        msg = {"type": "sprite" if kind == "thing" else "backdrop", "id": thing_id, "prompt": prompt,
               "relief": True, "place": place}
        cached = pictures.get((kind, prompt))
        if cached and media.get(cached.rsplit("/", 1)[-1]):
            await self.send({**msg, "url": cached})
            return
        try:
            r = await http.post(f"{IMAGERY_URL}/paint", timeout=240,
                                json={"prompt": prompt, "kind": kind, "priority": priority})
            if r.status_code == 404:
                await self.imagine(prompt, "sprite" if kind == "thing" else "backdrop", thing_id)
                return
            r.raise_for_status()
            url = media.put(r.content, "image/png", "png")
            pictures[(kind, prompt)] = url
            await self.send({**msg, "url": url})
        except Exception as e:  # noqa: BLE001
            log.info("painting skipped: %s", e)

    async def imagine(self, prompt: str, kind: str, thing_id: str = ""):
        cached = pictures.get((kind, prompt))
        if cached and media.get(cached.rsplit("/", 1)[-1]):
            await self.send({"type": kind, "url": cached, "id": thing_id, "prompt": prompt})
            return
        try:
            r = await http.post(f"{IMAGERY_URL}/imagine", json={"prompt": prompt, "kind": kind},
                                timeout=60)
            r.raise_for_status()
            url = media.put(r.content, "image/png", "png")
            pictures[(kind, prompt)] = url
            await self.send({"type": kind, "url": url, "id": thing_id, "prompt": prompt})
        except Exception as e:  # noqa: BLE001
            log.info("imagery skipped: %s", e)


# --------------------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(monitor.run())
    yield
    task.cancel()
    await http.aclose()


app = FastAPI(title="DreamScape web", lifespan=lifespan)


@app.middleware("http")
async def revalidate_page_files(request, call_next):
    """The page's files change while the game runs (live code): make the browser check
    for a newer version on every load instead of keeping a stale copy."""
    response = await call_next(request)
    if not request.url.path.startswith("/media/"):
        response.headers.setdefault("Cache-Control", "no-cache")
    return response


@app.get("/healthz")
async def healthz():
    return {"service": "web", "status": "ok", "clients": len(clients),
            "uptime_s": round(time.time() - STARTED)}


@app.get("/api/status")
async def status():
    return JSONResponse(monitor.snapshot())


@app.api_route("/api/debug/director/{path:path}", methods=["GET", "POST"])
async def debug_director(path: str, request: Request):
    """Pass-through to the director's /debug/* endpoints: model trials and downloads."""
    if path.split("/")[0] not in ("pull", "loaded", "turn", "recent", "unload", "release", "model",
                                    "models", "threads"):
        return JSONResponse({"error": "unknown debug path"}, status_code=404)
    url = f"{DIRECTOR_URL}/debug/{path}"
    try:
        if request.method == "GET":
            r = await http.get(url, timeout=10)
        else:
            r = await http.post(url, content=await request.body(), timeout=300,
                                headers={"content-type": "application/json"})
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=502)
    return Response(r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type", "application/json"))


@app.post("/api/debug/voice/device")
async def debug_voice_device(request: Request):
    """Pass-through to the voice service: move the voice between the GPU and the CPU."""
    try:
        r = await http.post(f"{VOICE_URL}/debug/device", content=await request.body(), timeout=180,
                            headers={"content-type": "application/json"})
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=502)
    return Response(r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type", "application/json"))


@app.get("/api/debug/voice/g2p")
async def debug_voice_g2p(text: str, speaker: int = 0, listener: int = 0):
    """Pass-through to the voice service: how it reads Hebrew, as IPA."""
    try:
        r = await http.get(f"{VOICE_URL}/debug/g2p", timeout=60,
                           params={"text": text, "speaker": speaker, "listener": listener})
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=502)
    return Response(r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type", "application/json"))


@app.get("/api/debug/tts")
async def debug_tts(text: str, lang: str = "he", voice: str = "neutral"):
    """Time one text-to-speech call (for tuning the voice on a live machine)."""
    t0 = time.perf_counter()
    try:
        r = await http.post(f"{VOICE_URL}/tts", json={"text": text[:600], "lang": lang, "voice": voice},
                            timeout=120)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=502)
    return {"status": r.status_code, "synth_s": r.headers.get("X-Synth-Seconds"),
            "wall_s": round(time.perf_counter() - t0, 3), "bytes": len(r.content)}


@app.get("/api/debug/gpu")
async def debug_gpu():
    """Graphics card memory, load, clock and heat, as the voice service sees them."""
    try:
        r = await http.get(f"{VOICE_URL}/debug/gpu", timeout=10)
        return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=502)


@app.get("/api/debug/imagery")
async def debug_imagery():
    """The painter's state: where it paints, how many it made and how long each stage took."""
    if not IMAGERY_URL:
        return JSONResponse({"error": "the imagery service is off"}, status_code=404)
    try:
        r = await http.get(f"{IMAGERY_URL}/healthz", timeout=5)
        return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=502)


@app.get("/api/debug/recent")
async def debug_recent():
    """The storyteller's last raw answers (for tuning the game on a live machine)."""
    try:
        r = await http.get(f"{DIRECTOR_URL}/debug/recent", timeout=5)
        return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=502)


@app.get("/media/{key}")
async def get_media(key: str):
    item = media.get(key)
    if not item:
        return Response(status_code=404)
    data, mime = item
    return Response(content=data, media_type=mime, headers={"Cache-Control": "max-age=3600"})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    d = Dreamer(ws)
    clients.add(d)
    await d.send({"type": "status", "services": monitor.snapshot()})
    tasks: set[asyncio.Task] = set()

    def spawn(coro):
        t = asyncio.create_task(coro)
        tasks.add(t)
        t.add_done_callback(tasks.discard)

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                meta, d.audio_meta = d.audio_meta or {}, None
                spawn(d.turn(audio=msg["bytes"], mime=meta.get("mime", ""), lang_hint=meta.get("lang")))
                continue
            try:
                data = json.loads(msg.get("text") or "{}")
            except ValueError:
                continue
            kind = data.get("type")
            if kind == "hello":
                spawn(d.hello(data))
            elif kind == "text":
                spawn(d.turn(text=str(data.get("text", "")), lang_hint=data.get("lang")))
            elif kind == "audio":
                d.audio_meta = data
            elif kind == "reset":
                spawn(d.reset(data))
            elif kind == "ping":
                await d.send({"type": "pong", "t": data.get("t")})
            elif kind == "more":
                d.more()
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(d)
        d.stop_directing()
        for t in tasks:
            t.cancel()


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
