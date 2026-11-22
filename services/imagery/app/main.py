"""Imagery: dreamy backdrops and sprites from a fast diffusion model.

Work goes to GPU workers on other machines through Dask (the scheduler runs in this
container, port 8786). With IMAGERY_LOCAL=1 it can also render here.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from . import render

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("imagery")

SCHEDULER = os.environ.get("DASK_SCHEDULER", "tcp://127.0.0.1:8786")
LOCAL = os.environ.get("IMAGERY_LOCAL", "0") == "1"
STARTED = time.time()
state = {"client": None, "local": "off", "detail": "starting", "renders": 0, "last_s": None}


async def _connect():
    from dask.distributed import Client
    while True:
        try:
            state["client"] = await Client(SCHEDULER, asynchronous=True, timeout="5s")
            log.info("connected to Dask scheduler at %s", SCHEDULER)
            return
        except Exception as e:  # noqa: BLE001
            state["detail"] = f"waiting for the Dask scheduler ({type(e).__name__})"
            await asyncio.sleep(3)


async def _warm_local():
    state["local"] = "loading"
    try:
        dev = await asyncio.to_thread(render.warm)
        state["local"] = f"ready on {dev}"
    except Exception as e:  # noqa: BLE001
        state["local"] = f"error: {type(e).__name__}: {e}"[:200]
        log.exception("local imagery failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks = [asyncio.create_task(_connect())]
    if LOCAL:
        tasks.append(asyncio.create_task(_warm_local()))
    yield
    for t in tasks:
        t.cancel()
    if state["client"] is not None:
        await state["client"].close()


app = FastAPI(title="DreamScape imagery", lifespan=lifespan)


def _workers() -> int:
    c = state["client"]
    if c is None:
        return 0
    try:
        return len(c.scheduler_info().get("workers", {}))
    except Exception:  # noqa: BLE001
        return 0


@app.get("/healthz")
async def healthz():
    n = _workers()
    local_ready = state["local"].startswith("ready")
    if n or local_ready:
        status, detail = "ok", (f"{n} GPU worker(s)" if n else f"local GPU, {state['local']}")
    elif LOCAL and state["local"] == "loading":
        status, detail = "loading", "loading the image model"
    else:
        status, detail = "loading", "waiting for a GPU worker to join"
    return {"service": "imagery", "status": status, "detail": detail, "workers": n,
            "local": state["local"], "renders": state["renders"], "last_s": state["last_s"],
            "uptime_s": round(time.time() - STARTED)}


class ImagineIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=400)
    kind: str = "backdrop"


@app.post("/imagine")
async def imagine(body: ImagineIn):
    kind = body.kind if body.kind in render.STYLE else "backdrop"
    seed = int(hashlib.sha1(body.prompt.encode()).hexdigest()[:8], 16)
    t0 = time.perf_counter()
    try:
        if _workers():
            fut = state["client"].submit(render.render_png, body.prompt, kind, seed, pure=False)
            png = await asyncio.wait_for(fut, timeout=60)
        elif state["local"].startswith("ready"):
            png = await asyncio.to_thread(render.render_png, body.prompt, kind, seed)
        else:
            raise HTTPException(503, "no GPU available for images")
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        log.warning("render failed: %s", e)
        raise HTTPException(502, f"render failed: {type(e).__name__}")
    state["renders"] += 1
    state["last_s"] = round(time.perf_counter() - t0, 2)
    return Response(content=png, media_type="image/png")
