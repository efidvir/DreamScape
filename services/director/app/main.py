"""Director: the game master. Turns what the player says into the dreamer's reply
and a structured update of the dream scene."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import prompt
from .llm import Ollama
from .schema import TurnOutput, parse_turn, turn_schema
from .story import SessionStore

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("director")
logging.getLogger("httpx").setLevel(logging.WARNING)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma3:4b")
DATA_DIR = os.environ.get("DATA_DIR", "/data/sessions")
HISTORY_TURNS = int(os.environ.get("HISTORY_TURNS", "8"))

STARTED = time.time()
SCHEMA = turn_schema()
llm = Ollama(OLLAMA_URL, LLM_MODEL,
             keep_alive=os.environ.get("LLM_KEEP_ALIVE", "2h"),
             num_ctx=int(os.environ.get("LLM_NUM_CTX", "4096")),
             temperature=float(os.environ.get("LLM_TEMPERATURE", "0.8")),
             timeout=float(os.environ.get("LLM_TIMEOUT", "90")))
store = SessionStore(DATA_DIR)
locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
stats = {"turns": 0, "fallbacks": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(llm.prepare())
    yield
    task.cancel()
    await llm.close()


app = FastAPI(title="DreamScape director", lifespan=lifespan)


class SessionIn(BaseModel):
    session_id: str = Field(min_length=6, max_length=64)
    lang: str = "he"


class TurnIn(BaseModel):
    session_id: str = Field(min_length=6, max_length=64)
    text: str = Field(min_length=1, max_length=2000)
    lang: str | None = None


def _lang(v: str | None, default: str = "he") -> str:
    return v if v in ("he", "en") else default


def _payload(sess, reply: str, extra: dict | None = None) -> dict:
    out = {"session_id": sess.id, "lang": sess.lang, "reply": reply, "voice": sess.voice,
           "scene": sess.scene(), "added": [], "changed": [], "removed": [], "effect": "none",
           "backdrop": "", "facts": sess.facts}
    out.update(extra or {})
    return out


@app.get("/healthz")
async def healthz():
    status = {"ready": "ok", "error": "error"}.get(llm.state, "loading")
    return {"service": "director", "status": status, "state": llm.state, "detail": llm.detail,
            "model": LLM_MODEL, "progress": llm.progress, "sessions": store.count(),
            "turns": stats["turns"], "fallbacks": stats["fallbacks"],
            "last_latency_s": llm.last_latency, "tokens_per_s": llm.last_tokens_per_s,
            "uptime_s": round(time.time() - STARTED)}


@app.post("/session")
async def open_session(body: SessionIn):
    """Resume a dream if it exists, otherwise start a new one with an opening line."""
    sess = store.get(body.session_id)
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
    sess = store.get(session_id)
    if not sess:
        raise HTTPException(404, "no such dream")
    return _payload(sess, "")


@app.post("/turn")
async def turn(body: TurnIn):
    sess = store.get(body.session_id) or store.create(body.session_id, _lang(body.lang))
    text = " ".join(body.text.split())
    # Answer in the language the player actually used; the hint only breaks ties.
    lang = prompt.detect_lang(text, _lang(body.lang, sess.lang))
    async with locks[sess.id]:
        sess.lang = lang
        msgs = prompt.build_messages(sess.summary(), sess.history, text, lang, HISTORY_TURNS)
        out: TurnOutput
        fallback = False
        try:
            if llm.state != "ready":
                raise RuntimeError(f"model not ready ({llm.state})")
            out = parse_turn(await llm.chat_json(msgs, SCHEMA))
            if not out.reply:
                raise ValueError("empty reply")
        except Exception as e:  # noqa: BLE001 - the dream must never stall
            log.warning("turn fell back: %s", e)
            fallback = True
            stats["fallbacks"] += 1
            out = TurnOutput(reply=prompt.FALLBACK[lang], effect="ripple")
        sess.remember("user", text)
        change = sess.apply(out) if not fallback else {"added": [], "changed": [], "removed": [],
                                                       "effect": "ripple"}
        reply = out.reply
        if sess.ended and out.ending:
            reply = f"{reply} {prompt.ENDINGS[lang]}"
        sess.remember("assistant", reply)
        store.save(sess)
        stats["turns"] += 1
        return _payload(sess, reply, {**change, "backdrop": out.backdrop, "ending": out.ending,
                                      "fallback": fallback, "latency_s": llm.last_latency})
