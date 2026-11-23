"""Director: the game master. Turns what the player says into the dreamer's reply
and a structured update of the dream scene."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import defaultdict, deque
from contextlib import aclosing, asynccontextmanager
from typing import Annotated

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import guard, hebrew, lexicon, prompt, vocab
from .llm import Ollama, parse_json_loose
from .schema import TurnOutput, parse_turn, turn_schema
from .story import SESSION_ID_RE, SessionStore

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("director")
logging.getLogger("httpx").setLevel(logging.WARNING)


class _QuietHealth(logging.Filter):
    """Heartbeats arrive every few seconds; keep them out of the access log."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "/healthz" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(_QuietHealth())

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma4:e4b")
DATA_DIR = os.environ.get("DATA_DIR", "/data/sessions")
HISTORY_TURNS = int(os.environ.get("HISTORY_TURNS", "8"))
# A turn that arrives while the model is still loading waits this long before the
# caller is told to come back later (HTTP 503).
READY_WAIT_S = float(os.environ.get("READY_WAIT_S", "20"))
FALLBACK_LINES = set(prompt.FALLBACK.values())

STARTED = time.time()
SCHEMAS = {True: turn_schema(pictures=True), False: turn_schema(pictures=False)}
llm = Ollama(OLLAMA_URL, LLM_MODEL,
             # Kept loaded while the game runs (the watchdog reloads it if Ollama restarts).
             keep_alive=os.environ.get("LLM_KEEP_ALIVE", "-1"),
             num_ctx=int(os.environ.get("LLM_NUM_CTX", "4096")),
             temperature=float(os.environ.get("LLM_TEMPERATURE", "0.8")),
             timeout=float(os.environ.get("LLM_TIMEOUT", "90")),
             num_thread=int(os.environ.get("LLM_NUM_THREAD", "0") or 0))
store = SessionStore(DATA_DIR)
locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
stats = {"turns": 0, "fallbacks": 0, "wrong_language": 0, "softened": 0}
# The last few raw model answers, for /debug/recent (tuning the prompt on a live game).
recent: deque[dict] = deque(maxlen=20)


# One tiny chat after loading, so the fixed prompt prefix is already cached for the first turn.
WARM_MESSAGES = prompt.build_messages(SessionStore.blank_summary(), [], "...", "he", HISTORY_TURNS,
                                      pictures=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(llm.watch(WARM_MESSAGES, SCHEMAS[False]))
    yield
    task.cancel()
    await llm.close()


app = FastAPI(title="DreamScape director", lifespan=lifespan)


# A dream's id names its file, so anything else is turned away before a turn starts.
SessionId = Annotated[str, Field(min_length=6, max_length=64, pattern=SESSION_ID_RE.pattern)]


class SessionIn(BaseModel):
    session_id: SessionId
    lang: str = "he"


class TurnIn(BaseModel):
    session_id: SessionId
    text: str = Field(min_length=1, max_length=2000)
    lang: str | None = None
    pictures: bool = False      # true when an imagery service will paint a backdrop
    model: str | None = None    # trials only (/debug): answer with this model instead


def _lang(v: str | None, default: str = "he") -> str:
    return v if v in ("he", "en") else default


def _load(session_id: str):
    """A stored dream, minus any exchange that ended in a fallback line."""
    sess = store.get(session_id)
    if sess and sess.forget_replies(FALLBACK_LINES):
        store.save(sess)
    return sess


async def _wait_ready(timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while llm.state != "ready":
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(0.5)
    return True


def _listener(sess, words: str) -> str:
    """Who the dreamer's words speak to: how they address the player, else how the player
    speaks of themselves. The voice reads "שלך" as shelach or shelcha from it."""
    return hebrew.addressed(words) or sess.player


def _payload(sess, reply: str, extra: dict | None = None) -> dict:
    out = {"session_id": sess.id, "lang": sess.lang, "reply": reply, "voice": sess.voice,
           "listener": _listener(sess, reply),
           "scene": sess.scene(), "added": [], "changed": [], "removed": [], "effect": "none",
           "backdrop": "", "facts": sess.facts}
    out.update(extra or {})
    return out


@app.get("/healthz")
async def healthz():
    status = {"ready": "ok", "error": "error"}.get(llm.state, "loading")
    return {"service": "director", "status": status, "ready": llm.state == "ready",
            "state": llm.state, "detail": llm.detail,
            "model": llm.model, "progress": llm.progress, "sessions": store.count(),
            "turns": stats["turns"], "fallbacks": stats["fallbacks"],
            "wrong_language": stats["wrong_language"],
            "last_latency_s": llm.last_latency, "tokens_per_s": llm.last_tokens_per_s,
            "uptime_s": round(time.time() - STARTED)}


class PullIn(BaseModel):
    model: str = Field(min_length=2, max_length=120)


@app.post("/debug/pull")
async def debug_pull(body: PullIn):
    """Start downloading another model in the background (the game keeps running)."""
    rec = llm.side_pulls.get(body.model)
    if not rec or rec["state"] == "error":
        asyncio.create_task(llm.pull_on_the_side(body.model))
    return {"model": body.model, "started": True}


@app.get("/debug/pull")
async def debug_pull_status():
    return llm.side_pulls


@app.get("/debug/models")
async def debug_models():
    """The models downloaded into Ollama, with their size on disk."""
    r = await llm._client.get(f"{llm.base_url}/api/tags", timeout=10)
    r.raise_for_status()
    return [{"name": m.get("name"), "size_gb": round((m.get("size") or 0) / 1e9, 2),
             "family": (m.get("details") or {}).get("family"),
             "quantization": (m.get("details") or {}).get("quantization_level")}
            for m in r.json().get("models", [])]


@app.get("/debug/loaded")
async def debug_loaded():
    """What Ollama holds in memory right now, with VRAM use."""
    return [{"name": m.get("name"), "size_gb": round((m.get("size") or 0) / 1e9, 2),
             "vram_gb": round((m.get("size_vram") or 0) / 1e9, 2),
             "context": m.get("context_length")} for m in await llm.loaded()]


@app.post("/debug/unload")
async def debug_unload(body: PullIn):
    """Free a trial model's memory. The game's own model comes back on the watchdog's
    next look once the trial hold is over (or at once if it is the one being unloaded)."""
    await llm.unload(body.model)
    if body.model == llm.model:
        llm.hold_until = 0
    return {"model": body.model, "unloaded": True, "loaded": [m.get("name") for m in await llm.loaded()]}


@app.post("/debug/model")
async def debug_model(body: PullIn):
    """Play with another (already downloaded) model until the director restarts; LLM_MODEL
    in .env is what makes a choice stick. The old model leaves the GPU first."""
    old = llm.model
    if body.model != old:
        llm.hold_until = time.time() + 600        # the watchdog stays out while it loads
        llm.model = body.model
        llm.state, llm.detail = "loading", f"switching to {body.model}"
        try:
            await llm.unload(old)
        except Exception as e:  # noqa: BLE001
            log.warning("could not unload %s: %s", old, e)

        async def switch():
            await llm.prepare(WARM_MESSAGES, SCHEMAS[False])
            llm.hold_until = 0
        asyncio.create_task(switch())
    return {"model": body.model, "previous": old}


class ThreadsIn(BaseModel):
    n: int = Field(ge=0, le=64)


@app.post("/debug/threads")
async def debug_threads(body: ThreadsIn):
    """Try another CPU thread count for the model (0: Ollama decides) until the director
    restarts; LLM_NUM_THREAD makes it stick. The model loads again with it."""
    llm.num_thread = body.n
    llm.hold_until = time.time() + 600
    llm.state, llm.detail = "loading", f"loading {llm.model} with {body.n or 'default'} threads"

    async def reload():
        try:
            await llm.unload(llm.model)
        except Exception as e:  # noqa: BLE001
            log.warning("could not unload %s: %s", llm.model, e)
        await llm.prepare(WARM_MESSAGES, SCHEMAS[False])
        llm.hold_until = 0
    asyncio.create_task(reload())
    return {"num_thread": body.n}


@app.post("/debug/release")
async def debug_release():
    """End the trial hold so the watchdog reloads the game's model right away."""
    llm.hold_until = 0
    return {"held": False}


@app.post("/debug/turn")
async def debug_turn(body: TurnIn):
    """A turn answered by any model (body.model), for side-by-side trials. The watchdog
    leaves Ollama alone for a few minutes so the trial model is not swapped out."""
    llm.hold_until = time.time() + 300
    t0 = time.perf_counter()
    first = None
    final = None
    asked_again = stats["wrong_language"]
    words = None
    async for kind, payload in _turn_events(body):
        if kind in ("first", "reply") and first is None:
            first = round(time.perf_counter() - t0, 2)
        if kind == "reply":
            words = round(time.perf_counter() - t0, 2)
        if kind == "final":
            final = payload
    return {**(final or {}), "first_words_s": first, "all_words_s": words,
            "total_s": round(time.perf_counter() - t0, 2),
            "model": body.model or llm.model, "tokens_per_s": llm.last_tokens_per_s,
            "prompt_tokens_per_s": llm.last_prompt_tokens_per_s,
            "asked_again_for_language": stats["wrong_language"] - asked_again}


@app.get("/debug/recent")
async def debug_recent():
    """The last raw model answers, newest first."""
    return list(reversed(recent))


@app.post("/session")
async def open_session(body: SessionIn):
    """Resume a dream if it exists, otherwise start a new one with an opening line."""
    sess = _load(body.session_id)
    if sess:
        last = next((h["text"] for h in reversed(sess.history) if h["role"] == "assistant"), "")
        return _payload(sess, last, {"resumed": True})
    try:
        sess = store.create(body.session_id, _lang(body.lang))
    except ValueError as e:
        raise HTTPException(400, str(e))
    line = prompt.opening_line(sess.lang)
    sess.remember("assistant", line)
    store.save(sess)
    return _payload(sess, line, {"resumed": False})


@app.post("/reset")
async def reset(body: SessionIn):
    try:
        sess = store.create(body.session_id, _lang(body.lang))
    except ValueError as e:
        raise HTTPException(400, str(e))
    line = prompt.opening_line(sess.lang)
    sess.remember("assistant", line)
    store.save(sess)
    return _payload(sess, line, {"resumed": False})


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    sess = _load(session_id)
    if not sess:
        raise HTTPException(404, "no such dream")
    return _payload(sess, "")


# The spoken reply inside a partly written answer: its first sentence as soon as that is
# complete, then all of it once the closing quote has arrived.
REPLY_OPEN = re.compile(r'"reply"\s*:\s*"((?:[^"\\]|\\.)*)')
REPLY_DONE = re.compile(r'"reply"\s*:\s*"((?:[^"\\]|\\.)*)"')
VOICE_DONE = re.compile(r'"voice"\s*:\s*"(\w+)"')
SENTENCE_END = re.compile(r"[.!?…]+[\"'”»)]*\s+")
FIRST_MIN = 25          # a shorter first sentence ("Snow...") waits for the one after it


def _json_text(raw: str) -> str:
    try:
        return json.loads(f'"{raw}"')
    except ValueError:      # cut inside an escape sequence; the next piece completes it
        return ""


def first_sentence(partial: str) -> str:
    """The first complete sentence of a reply that is still being written, once it is long
    enough to be spoken on its own."""
    for m in SENTENCE_END.finditer(partial):
        head = partial[:m.end()].strip()
        if len(head) >= FIRST_MIN:
            return head
    return ""


def _gentle(words: str, heard_as: guard.Heard, drift: str) -> str:
    """The dreamer's words as the player may hear them: listed words dissolved, and an
    answer that steps out of the dream ("as an AI...") turned into the dream drifting."""
    words = guard.clean(words)
    if not heard_as.care and guard.steps_out(words):
        return drift
    return words


def _unmarked(text: str) -> str:
    return " ".join(text.replace(guard.DISSOLVED, " ").split())


def _keep_gentle(out: TurnOutput, heard_as: guard.Heard, drift: str, stepped_out: bool = False) -> None:
    """The same for the whole answer: what the dream remembers and shows stays gentle.
    stepped_out: the model's own reply left the dream (it was replaced while streaming)."""
    stepped_out = stepped_out or (not heard_as.care and guard.steps_out(out.reply))
    out.reply = _gentle(out.reply, heard_as, drift)
    out.facts = [_unmarked(f) for f in out.facts if not guard.flagged(f) and _unmarked(f)]
    out.self_label = "" if guard.flagged(out.self_label) else _unmarked(out.self_label)
    for t in out.add:
        t.label = "" if guard.flagged(t.label) else _unmarked(t.label)
    if guard.flagged(out.backdrop):
        out.backdrop = ""
    if stepped_out:                      # nothing of a refused request reaches the scene
        out.facts, out.add, out.change, out.remove, out.backdrop = [], [], [], [], ""
    if out.effect == "none" and (heard_as.drift or stepped_out):
        out.effect = "ripple"            # the picture wavers as the dream moves on
    elif out.effect == "none" and heard_as.care:
        out.effect = "bloom"


async def _turn_events(body: TurnIn):
    """One turn as events: ("first", the reply's first sentence) as soon as the model has
    written it, ("reply", all the words) once they are complete, then ("final", the whole
    payload). Must be entered after the model is ready."""
    sess = _load(body.session_id) or store.create(body.session_id, _lang(body.lang))
    said = " ".join(body.text.split())
    # Words the dream leaves out dissolve before the storyteller hears them, and the page
    # shows the player the same dissolved line.
    heard_as = guard.hear(said)
    text = heard_as.text or guard.DISSOLVED
    if text != said:
        yield "heard", {"text": text}
    if heard_as.kinds:
        stats["softened"] += 1
    # Answer in the language the player actually used; the hint only breaks ties.
    lang = prompt.detect_lang(text, _lang(body.lang, sess.lang))
    heard = lexicon.read(text)
    sess.player = hebrew.self_gender(text) or sess.player     # "אני הולכת": the player is a she
    drift = guard.drift_line(lang)     # one per turn, so a replaced sentence and reply agree
    async with locks[sess.id]:
        sess.lang = lang
        # Hebrew verbs about the dreamer agree with its gender, known or just given.
        gender = heard.voice or (sess.voice if sess.voice in ("female", "male") else "")
        out, error, early, first, wrong_lang = None, "", None, None, False
        for attempt in range(2):
            # After an answer in the wrong language the second try sees only examples in the
            # right one, plus a plain reminder.
            msgs = prompt.build_messages(sess.summary(), sess.history, text, lang, HISTORY_TURNS,
                                         sess.now_line(), pictures=body.pictures,
                                         strict_lang=wrong_lang, gender=gender,
                                         note=guard.note(heard_as), player=sess.player)
            try:
                # The second try runs a little cooler, which usually fixes a malformed answer.
                written, wrong_lang = "", False
                async with aclosing(llm.chat_stream(msgs, SCHEMAS[body.pictures],
                                                    temperature=None if attempt == 0 else 0.5,
                                                    model=body.model)) as stream:
                    async for piece in stream:
                        written += piece
                        if early is not None:
                            continue             # the words are out; the scene is coming
                        done = REPLY_DONE.search(written)
                        if done:
                            words = _json_text(done.group(1)).strip()
                        elif first is None and (m := REPLY_OPEN.search(written)):
                            words = first_sentence(_json_text(m.group(1)))
                        else:
                            continue
                        words = _gentle(words, heard_as, drift)
                        if not words:
                            continue
                        said_in = prompt.detect_lang(words, lang)
                        if said_in != lang and attempt == 0 and first is None:
                            # Stop here and ask again before the player hears a word.
                            wrong_lang = True
                            break
                        v = VOICE_DONE.search(written)
                        voice = v.group(1) if v and v.group(1) in vocab.VOICES else None
                        if not voice and sess.voice == "neutral":
                            # what the player made it, else how it speaks of itself ("אני מרגישה")
                            voice = heard.voice or hebrew.self_gender(words) or None
                        # The voice reads the words in the language they are written in.
                        said = {"lang": said_in, "voice": voice or sess.voice,
                                "listener": _listener(sess, words)}
                        if done:
                            early = words
                            yield "reply", {"reply": early, **said}
                        else:
                            first = words
                            yield "first", {"text": first, **said}
                if wrong_lang:
                    stats["wrong_language"] += 1
                    error = "the model answered in the wrong language"
                    log.warning("turn attempt %d: %s", attempt + 1, error)
                    continue
                raw = parse_json_loose(written)
                recent.append({"time": round(time.time()), "session": sess.id, "said": text,
                               "lang": lang, "attempt": attempt + 1, "latency_s": llm.last_latency,
                               "tokens_per_s": llm.last_tokens_per_s,
                               "prompt_tokens_per_s": llm.last_prompt_tokens_per_s,
                               "model": body.model or llm.model, "raw": raw})
                got = parse_turn(raw)
                stepped_out = not heard_as.care and guard.steps_out(got.reply)
                if early:
                    got.reply = early            # the player is already hearing these words
                _keep_gentle(got, heard_as, drift, stepped_out)
                if got.reply:
                    out = got
                    break
                error = "the model gave an empty reply"
            except httpx.TimeoutException:
                error = "the model took too long"
                log.warning("turn attempt %d failed: %s", attempt + 1, error)
                break
            except Exception as e:  # noqa: BLE001 - the dream must never stall
                error = f"{type(e).__name__}: {e}"[:200]
            if first and early is None:
                # Its first sentence is already being spoken: that is the reply now, and
                # another try only shapes the scene around it.
                early = first
            log.warning("turn attempt %d failed: %s", attempt + 1, error)
        if out is None and (early or first):
            # The words were spoken but the scene got lost: the player's own words shape it.
            out = TurnOutput(reply=early or first)
        if out is None:
            stats["fallbacks"] += 1
            # Nothing is recorded, so a failed turn never shapes the story or the next answer.
            yield "final", _payload(sess, prompt.FALLBACK[lang],
                                    {"fallback": True, "error": error, "effect": "ripple"})
            return
        # Fill in what the player named but the model left out of the scene.
        notes = lexicon.nudge(out, text, sess.setting["kind"], sess.self_form, sess.voice,
                              {t["kind"] for t in sess.things.values()},
                              [t["label"].lower() for t in sess.things.values()],
                              current_label=sess.self_label)
        if recent:
            recent[-1]["nudged"] = notes
        sess.remember("user", text)
        change = sess.apply(out)
        if sess.voice == "neutral":
            sess.voice = hebrew.self_gender(out.reply) or "neutral"   # its words chose a gender
        reply = out.reply
        if sess.ended and out.ending:
            reply = f"{reply} {prompt.ENDINGS[lang]}"
        sess.remember("assistant", reply)
        store.save(sess)
        stats["turns"] += 1
        yield "final", _payload(sess, reply, {**change, "backdrop": out.backdrop,
                                              "ending": out.ending, "fallback": False,
                                              "latency_s": llm.last_latency})


async def _ready_or_503():
    if not await _wait_ready(READY_WAIT_S):
        # Nothing is recorded: the caller keeps the player's words and tries again later.
        raise HTTPException(503, detail={"state": llm.state, "detail": llm.detail,
                                         "progress": llm.progress})


async def _whole_turn(body: TurnIn):
    """The turn's events, always ending with one "final". Whatever goes wrong on the way
    ends the answer with a fallback the caller can show, so a half-sent answer never looks
    like the storyteller going away (which would hold every player's words)."""
    ended = False
    try:
        async with aclosing(_turn_events(body)) as events:
            async for kind, payload in events:
                ended = ended or kind == "final"
                yield kind, payload
    except Exception as e:  # noqa: BLE001 - the dream must never stall
        if ended:
            raise
        stats["fallbacks"] += 1
        log.exception("turn failed")
        yield "final", {"session_id": body.session_id, "lang": _lang(body.lang),
                        "reply": prompt.FALLBACK[_lang(body.lang)], "fallback": True,
                        "error": f"{type(e).__name__}: {e}"[:200], "effect": "ripple"}


@app.post("/turn")
async def turn(body: TurnIn):
    await _ready_or_503()
    final = None
    async for kind, payload in _whole_turn(body):
        if kind == "final":
            final = payload
    return final


@app.post("/turn/stream")
async def turn_stream(body: TurnIn):
    """Newline-delimited JSON: {"type": "reply", ...} as soon as the dreamer's words are
    written, then {"type": "final", ...} with the scene."""
    await _ready_or_503()

    async def lines():
        async for kind, payload in _whole_turn(body):
            yield json.dumps({"type": kind, **payload}, ensure_ascii=False) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson")
