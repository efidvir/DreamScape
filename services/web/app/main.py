"""Web: serves the dream page, runs each turn (speech -> director -> scene + voice),
and watches the heartbeat of every other service."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect, WebSocketState

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("web")
logging.getLogger("httpx").setLevel(logging.WARNING)

DIRECTOR_URL = os.environ.get("DIRECTOR_URL", "http://director:9010").rstrip("/")
VOICE_URL = os.environ.get("VOICE_URL", "http://voice:9020").rstrip("/")
IMAGERY_URL = os.environ.get("IMAGERY_URL", "").rstrip("/")
HEARTBEAT_S = float(os.environ.get("HEARTBEAT_S", "3"))
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
STARTED = time.time()

FALLBACK = {
    "he": "הקול שלך נבלע בערפל... אפשר לומר את זה שוב?",
    "en": "Your voice dissolves in the mist... could you say that again?",
}
NOT_HEARD = {
    "he": "רק רחש של רוח... לא שמעתי מילים.",
    "en": "Only a whisper of wind... I heard no words.",
}

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


# --------------------------------------------------------------------------- one connected player
class Dreamer:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.session = ""
        self.lang = "he"
        self.voice = "neutral"
        self.last_reply = ""
        self.busy = False
        self.audio_meta: dict | None = None
        self.self_key = ""

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
        self.session = str(data.get("session") or uuid.uuid4().hex)[:64]
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
        self.lang = d.get("lang", self.lang)
        self.last_reply = d.get("reply", "")
        await self.send({"type": "session", "session": self.session, "resumed": d.get("resumed"),
                         "scene": d.get("scene"), "lang": self.lang, "reply": self.last_reply,
                         "facts": d.get("facts", [])})
        if d.get("scene"):
            self.request_pictures(d["scene"], [], everything=True)
        if not d.get("resumed") and self.last_reply:
            await self.speak(self.last_reply, self.lang)

    # ----------------------------------------------------------------- the turn pipeline
    async def turn(self, text: str | None = None, audio: bytes | None = None, mime: str = "",
                   lang_hint: str | None = None):
        if self.busy:
            await self.send({"type": "notice", "key": "busy"})
            return
        self.busy = True
        t0 = time.perf_counter()
        timings = {}
        try:
            if audio is not None:
                await self.stage("hearing")
                text = await self.transcribe(audio, mime, lang_hint)
                timings["stt_s"] = round(time.perf_counter() - t0, 2)
                if not text:
                    lang = lang_hint if lang_hint in ("he", "en") else self.lang
                    await self.send({"type": "reply", "text": NOT_HEARD[lang], "lang": lang,
                                     "fallback": True})
                    await self.speak(NOT_HEARD[lang], lang)
                    return
            text = (text or "").strip()[:1500]
            if not text:
                return
            await self.send({"type": "transcript", "text": text})
            await self.stage("dreaming")
            t1 = time.perf_counter()
            d = await self.ask_director(text, lang_hint)
            timings["director_s"] = round(time.perf_counter() - t1, 2)
            lang = d.get("lang", self.lang)
            self.lang, self.voice = lang, d.get("voice", self.voice)
            self.last_reply = d.get("reply", "")
            if d.get("scene"):
                await self.send({"type": "scene", "scene": d["scene"], "added": d.get("added", []),
                                 "changed": d.get("changed", []), "removed": d.get("removed", []),
                                 "effect": d.get("effect", "none"), "ending": d.get("ending", False)})
            await self.send({"type": "reply", "text": self.last_reply, "lang": lang,
                             "fallback": d.get("fallback", False)})
            if d.get("scene"):
                self.request_pictures(d["scene"], d.get("added", []), d.get("backdrop", ""))
            t2 = time.perf_counter()
            await self.speak(self.last_reply, lang)
            timings["tts_s"] = round(time.perf_counter() - t2, 2)
            timings["total_s"] = round(time.perf_counter() - t0, 2)
            await self.send({"type": "timings", **timings})
        finally:
            self.busy = False
            await self.stage("idle")

    async def transcribe(self, audio: bytes, mime: str, lang_hint: str | None) -> str:
        files = {"audio": ("speech.webm", audio, mime or "application/octet-stream")}
        form = {"lang": lang_hint or "auto", "prompt": self.last_reply[:200]}
        try:
            r = await http.post(f"{VOICE_URL}/stt", files=files, data=form, timeout=60)
            r.raise_for_status()
            return (r.json().get("text") or "").strip()
        except Exception as e:  # noqa: BLE001
            log.warning("speech-to-text failed: %s", e)
            await self.send({"type": "notice", "key": "stt_down"})
            return ""

    async def ask_director(self, text: str, lang_hint: str | None) -> dict:
        try:
            r = await http.post(f"{DIRECTOR_URL}/turn",
                                json={"session_id": self.session, "text": text, "lang": lang_hint},
                                timeout=150)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            log.warning("director failed: %s", e)
            lang = "he" if any("֐" <= ch <= "׿" for ch in text) else "en"
            return {"reply": FALLBACK[lang], "lang": lang, "fallback": True, "scene": None}

    async def speak(self, text: str, lang: str):
        if not text:
            return
        try:
            r = await http.post(f"{VOICE_URL}/tts",
                                json={"text": text, "lang": lang, "voice": self.voice}, timeout=60)
            r.raise_for_status()
            url = media.put(r.content, "audio/wav", "wav")
            await self.send({"type": "audio", "url": url, "text": text})
        except Exception as e:  # noqa: BLE001 - the browser speaks instead
            log.info("text-to-speech unavailable (%s); browser voice fallback", e)
            await self.send({"type": "speak", "text": text, "lang": lang, "voice": self.voice})

    def request_pictures(self, scene: dict, added: list, backdrop: str = "", everything: bool = False):
        """With the imagery service up, every new thing (and the dreamer's new form) gets a
        generated picture that the page turns into its particle cloud."""
        if not (IMAGERY_URL and monitor.ok("imagery")):
            return
        if backdrop:
            asyncio.create_task(self.imagine(backdrop, "backdrop"))
        for t in scene.get("things", []):
            if (everything or t["id"] in added) and t.get("count", 1) <= 3 and t["kind"] not in SPRITE_SKIP:
                asyncio.create_task(self.imagine(describe(t.get("label"), t["kind"], t.get("color")), "sprite", t["id"]))
        me = scene.get("self") or {}
        key = f"{me.get('form')}|{me.get('label')}|{me.get('color')}"
        if me.get("form") not in (None, "wisp") and (everything or key != self.self_key):
            asyncio.create_task(self.imagine(describe(me.get("label"), me["form"], me.get("color")), "sprite", "self"))
        self.self_key = key

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


@app.get("/healthz")
async def healthz():
    return {"service": "web", "status": "ok", "clients": len(clients),
            "uptime_s": round(time.time() - STARTED)}


@app.get("/api/status")
async def status():
    return JSONResponse(monitor.snapshot())


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
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(d)
        for t in tasks:
            t.cancel()


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
