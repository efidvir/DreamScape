"""Voice: local speech-to-text and text-to-speech for Hebrew and English."""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .stt import SpeechToText
from .tts import TextToSpeech

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("voice")

STARTED = time.time()
stt = SpeechToText()
tts = TextToSpeech()


def _load_all():
    # Speech-to-text first: the player can type while text-to-speech is still loading.
    stt.load()
    tts.load()


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_load_all, daemon=True).start()
    yield


app = FastAPI(title="DreamScape voice", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    states = {stt.state, tts.state}
    if states == {"ready"}:
        status = "ok"
    elif "error" in states and "ready" in states:
        status = "degraded"
    elif states == {"error"}:
        status = "error"
    else:
        status = "loading"
    return {"service": "voice", "status": status,
            "stt": {"state": stt.state, "detail": stt.detail, "device": stt.active_device},
            "tts": {"state": tts.state, "detail": tts.detail},
            "uptime_s": round(time.time() - STARTED)}


@app.post("/stt")
async def speech_to_text(audio: UploadFile = File(...), lang: str = Form("auto"),
                         prompt: str = Form("")):
    data = await audio.read()
    if not data:
        raise HTTPException(400, "empty audio")
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(413, "audio too long")
    try:
        return await asyncio.to_thread(stt.transcribe, data, lang, prompt[:400])
    except RuntimeError as e:
        raise HTTPException(503, str(e))


class SpeakIn(BaseModel):
    text: str = Field(min_length=1, max_length=1200)
    lang: str = "he"
    voice: str = "neutral"


@app.post("/tts")
async def text_to_speech(body: SpeakIn):
    try:
        wav, took = await asyncio.to_thread(tts.synthesize, body.text, body.lang, body.voice)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    return Response(content=wav, media_type="audio/wav", headers={"X-Synth-Seconds": str(took)})
