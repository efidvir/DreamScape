"""Imagery: the dream's things and places, painted.

/paint paints one thing or one place and returns it packed for the page (see paint.py);
/imagine returns a plain picture. Work goes to GPU workers on other machines through Dask
when any have joined (the scheduler runs in this container, port 8786); with
IMAGERY_LOCAL=1 it is also done here, one painting at a time, the player's newest things
first. Finished paintings are kept on disk, so a thing dreamt before appears at once.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from contextlib import asynccontextmanager
from functools import partial

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from . import paint, render

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("imagery")

SCHEDULER = os.environ.get("DASK_SCHEDULER", "tcp://127.0.0.1:8786")
LOCAL = os.environ.get("IMAGERY_LOCAL", "0") == "1"
STARTED = time.time()
state = {"client": None, "local": "off", "detail": "starting", "renders": 0, "last_s": None,
         "paintings": 0, "reused": 0, "paint_s": None}
KEEP = int(os.environ.get("IMAGERY_KEEP", "400"))      # paintings kept on disk


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
    """Load the painter's models (downloading them on the first start), trying again a
    minute later if the network or the disk let it down."""
    wait = 30
    while True:
        state["local"] = "loading"
        try:
            dev = await asyncio.to_thread(paint.warm)
            state["local"] = f"ready on {dev}"
            return
        except Exception as e:  # noqa: BLE001
            state["local"] = f"error: {type(e).__name__}: {e}"[:200]
            log.exception("loading the painter failed; trying again in %d s", wait)
            await asyncio.sleep(wait)
            wait = min(wait * 2, 600)


class Painter:
    """Paintings made on this machine: one at a time, the lowest priority number first."""

    def __init__(self, limit: int = 16):
        self.queue: asyncio.PriorityQueue | None = None
        self.limit = limit
        self.n = 0

    def start(self) -> asyncio.Task:
        self.queue = asyncio.PriorityQueue()
        return asyncio.create_task(self._run())

    def waiting(self) -> int:
        return self.queue.qsize() if self.queue else 0

    async def _run(self):
        while True:
            _, _, job, fut = await self.queue.get()
            if fut.done():          # nobody waits for it any more
                continue
            try:
                result = await asyncio.to_thread(job)
                if not fut.done():
                    fut.set_result(result)
            except Exception as e:  # noqa: BLE001
                if not fut.done():
                    fut.set_exception(e)

    async def run(self, job, priority: int):
        if self.waiting() >= self.limit:
            raise HTTPException(503, "too many paintings waiting")
        fut = asyncio.get_running_loop().create_future()
        self.n += 1
        await self.queue.put((priority, self.n, job, fut))
        return await fut


painter = Painter()
inflight: dict[str, asyncio.Future] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks = [asyncio.create_task(_connect()), painter.start()]
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
        status, detail = "ok", (f"{n} GPU worker(s)" if n else f"painting here, {state['local']}")
    elif LOCAL and state["local"] == "loading":
        status, detail = "loading", "loading the painter's models (the first start downloads them)"
    elif LOCAL and state["local"].startswith("error"):
        status, detail = "loading", f"the painter could not load ({state['local'][7:]}); trying again"
    else:
        status, detail = "loading", "waiting for a GPU worker to join"
    return {"service": "imagery", "status": status, "detail": detail, "workers": n,
            "local": state["local"], "renders": state["renders"], "last_s": state["last_s"],
            "paint": True, "paintings": state["paintings"], "reused": state["reused"],
            "paint_s": state["paint_s"], "stages": paint.last, "waiting": painter.waiting(),
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


# --------------------------------------------------------------------------- paintings
class PaintIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=400)
    kind: str = "thing"          # "thing" (one thing, on its own) or "scene" (a wide view)
    priority: int = 1            # lower is sooner: 0 for what the player just named


def _shelf() -> str:
    return os.path.join(paint.models_dir(), "paintings")


def _keep(path: str, png: bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.part"
    with open(tmp, "wb") as f:
        f.write(png)
    os.replace(tmp, path)
    names = [os.path.join(_shelf(), n) for n in os.listdir(_shelf()) if n.endswith(".png")]
    if len(names) > KEEP:      # the least recently used go first
        names.sort(key=os.path.getmtime)
        for old in names[:len(names) - KEEP]:
            try:
                os.remove(old)
            except OSError:
                pass


def _reuse(path: str) -> bytes | None:
    try:
        with open(path, "rb") as f:
            png = f.read()
        os.utime(path)
        return png
    except OSError:
        return None


async def _make(prompt: str, kind: str, priority: int, path: str) -> bytes:
    seed = paint.seed_of(prompt, kind)
    t0 = time.perf_counter()
    if _workers():
        fut = state["client"].submit(paint.paint_png, prompt, kind, seed, pure=False)
        png = await asyncio.wait_for(fut, timeout=120)
    elif state["local"].startswith("ready"):
        png = await painter.run(partial(paint.paint_png, prompt, kind, seed), priority)
    else:
        raise HTTPException(503, "nowhere to paint yet")
    state["paintings"] += 1
    state["paint_s"] = round(time.perf_counter() - t0, 2)
    await asyncio.to_thread(_keep, path, png)
    return png


@app.post("/paint")
async def paint_one(body: PaintIn):
    kind = body.kind if body.kind in paint.STYLE else "thing"
    prompt = " ".join(body.prompt.split())
    key = hashlib.sha1(f"{paint.VERSION}|{kind}|{prompt}".encode()).hexdigest()
    path = os.path.join(_shelf(), f"{key}.png")
    png = await asyncio.to_thread(_reuse, path)
    if png is not None:
        state["reused"] += 1
        return Response(content=png, media_type="image/png", headers={"X-Painted": "kept"})
    task = inflight.get(key)
    if task is None:
        task = asyncio.ensure_future(_make(prompt, kind, body.priority, path))
        inflight[key] = task
        task.add_done_callback(lambda _: inflight.pop(key, None))
    try:
        png = await asyncio.shield(task)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        log.warning("painting failed: %s: %s", type(e).__name__, e)
        raise HTTPException(502, f"painting failed: {type(e).__name__}")
    return Response(content=png, media_type="image/png", headers={"X-Painted": "new"})
