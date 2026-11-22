"""Voice: local speech-to-text and text-to-speech for Hebrew and English."""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from . import stt as stt_mod
from . import tts as tts_mod
from .stt import SpeechToText
from .tts import TextToSpeech

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("voice")


class _QuietHealth(logging.Filter):
    """Heartbeats arrive every few seconds; keep them out of the access log."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "/healthz" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(_QuietHealth())

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
    except stt_mod.NotReady as e:
        raise HTTPException(503, str(e))
    except Exception as e:  # noqa: BLE001 - answer cleanly so the connection stays usable
        if type(e).__module__.startswith("av"):
            log.warning("could not decode %d bytes of %s: %s", len(data), audio.content_type, e)
            raise HTTPException(422, f"could not decode the recording ({type(e).__name__})")
        log.exception("speech-to-text failed")
        raise HTTPException(500, f"speech-to-text failed: {type(e).__name__}: {e}"[:300])


class SpeakIn(BaseModel):
    text: str = Field(min_length=1, max_length=1200)
    lang: str = "he"
    voice: str = "neutral"


@app.post("/tts")
async def text_to_speech(body: SpeakIn):
    try:
        wav, took = await asyncio.to_thread(tts.synthesize, body.text, body.lang, body.voice)
    except tts_mod.NotReady as e:
        raise HTTPException(503, str(e))
    except Exception as e:  # noqa: BLE001 - answer cleanly so the connection stays usable
        log.exception("text-to-speech failed")
        raise HTTPException(500, f"text-to-speech failed: {type(e).__name__}: {e}"[:300])
    return Response(content=wav, media_type="audio/wav", headers={"X-Synth-Seconds": str(took)})


GPU_FIELDS = ("name", "memory.used", "memory.total", "utilization.gpu", "temperature.gpu",
              "clocks.sm", "clocks.max.sm", "power.draw", "power.limit")


def _gpu_query(fields: tuple[str, ...]) -> list[str]:
    out = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(fields)}",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=5, check=True)
    return [v.strip() for v in out.stdout.strip().splitlines()[0].split(",")]


@app.get("/debug/gpu")
async def debug_gpu():
    """What the graphics card is doing right now: memory, load, clock and heat. Memory is
    the whole card's, so it covers the storyteller's model and the voices together."""
    if not shutil.which("nvidia-smi"):
        return {"ok": False, "error": "nvidia-smi is not available in this container"}
    try:
        values = await asyncio.to_thread(_gpu_query, GPU_FIELDS)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
    gpu = dict(zip(GPU_FIELDS, values))
    try:   # the processors the containers share (Docker Desktop gives them all to one VM)
        gpu["cpus"] = len(os.sched_getaffinity(0))
        gpu["load_1m"] = round(os.getloadavg()[0], 2)
    except (AttributeError, OSError):
        pass
    try:   # why the clock is held back, if it is (newer drivers renamed the field)
        gpu["clock_limits"] = (await asyncio.to_thread(_gpu_query, ("clocks_event_reasons.active",)))[0]
    except Exception:  # noqa: BLE001
        try:
            gpu["clock_limits"] = (await asyncio.to_thread(
                _gpu_query, ("clocks_throttle_reasons.active",)))[0]
        except Exception:  # noqa: BLE001
            pass
    return {"ok": True, "gpu": gpu}
