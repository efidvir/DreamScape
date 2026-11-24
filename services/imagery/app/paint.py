"""Painting the dream's things and places.

The image model paints the thing (or the place, as a wide view); a small matting model
finds the thing in its picture; a depth model reads how near every part of it is; and
the brush direction at every pixel comes from the painting itself. All of it is packed in
one PNG (see relief.py) that the page turns into brush strokes standing in space.

Importable on the API container and on remote Dask workers alike: each model loads once
per process. Without the matting model (no network on first use) the thing is told from
its background by colour and depth instead.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
import urllib.request

import numpy as np
from PIL import Image

from . import relief, render

log = logging.getLogger("imagery.paint")

VERSION = 1      # bump when the look of the paintings changes, so old ones are not reused

STYLE = {
    "thing": ("{p}, alone in the middle of the picture, whole and in full view, painterly oil "
              "painting, soft visible brush strokes, dreamlike glow, plain dark background"),
    "scene": ("{p}, wide panoramic view, impressionist oil painting, soft visible brush strokes, "
              "luminous dreamlike colours, deep atmospheric distance"),
}

DEPTH_MODEL = os.environ.get("IMAGERY_DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")
MATTES = {   # the matting model rembg calls silueta (u2net, Apache-2.0), from rembg's releases
    "silueta": ("https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx",
                "55e59e0d8062d2f5d013f4725ee84782", 320),
}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

_load = threading.Lock()
_depth = None
_matte = None
_matte_failed_at = 0.0
last = {}        # seconds per stage of the last painting


def models_dir() -> str:
    return os.environ.get("MODELS_DIR", "/models")


def sizes(kind: str) -> tuple[int, int]:
    if kind == "scene":
        w, h = os.environ.get("IMAGERY_SCENE_SIZE", "1024x512").lower().split("x")
        return int(w), int(h)
    s = int(os.environ.get("IMAGERY_THING_SIZE", "512"))
    return s, s


def steps() -> int:
    n = int(os.environ.get("IMAGERY_PAINT_STEPS", "0") or 0)
    return n if n > 0 else (2 if render.device() == "cuda" else 1)


def seed_of(prompt: str, kind: str) -> int:
    return int(hashlib.sha1(f"{kind}|{prompt}".encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------ depth
def depth_model():
    global _depth
    if _depth is None:
        with _load:
            if _depth is None:
                from transformers import AutoModelForDepthEstimation
                dev = render.device()
                model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).float().to(dev).eval()
                _depth = (model, dev)
    return _depth


def depth_of(img: Image.Image) -> np.ndarray:
    """Relative nearness of every pixel (bigger is nearer, any scale)."""
    import torch
    model, dev = depth_model()
    w, h = img.size
    s = 518 / min(w, h)                     # the model sees multiples of its 14-pixel patches
    tw, th = max(14, round(w * s / 14) * 14), max(14, round(h * s / 14) * 14)
    x = (np.asarray(img.convert("RGB").resize((tw, th), Image.BICUBIC), np.float32) / 255.0 - MEAN) / STD
    t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(dev)
    with torch.inference_mode():
        d = model(pixel_values=t).predicted_depth
        d = torch.nn.functional.interpolate(d[:, None].float(), size=(h, w), mode="bilinear", align_corners=False)
    return d[0, 0].cpu().numpy()


# ------------------------------------------------------------------ matting
def _fetch(url: str, md5: str, path: str) -> str:
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part, digest = path + ".part", hashlib.md5()
    with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
        for chunk in iter(lambda: r.read(1 << 20), b""):
            digest.update(chunk)
            f.write(chunk)
    if md5 and digest.hexdigest() != md5:
        os.remove(part)
        raise ValueError(f"{os.path.basename(path)} arrived damaged")
    os.replace(part, path)
    return path


def matte_session():
    """The matting model, or None while it can't be had (tried again after ten minutes)."""
    global _matte, _matte_failed_at
    if _matte is None and time.time() - _matte_failed_at > 600:
        with _load:
            if _matte is None and time.time() - _matte_failed_at > 600:
                name = os.environ.get("IMAGERY_MATTE", "silueta")
                try:
                    import onnxruntime as ort
                    url, md5, size = MATTES[name]
                    path = _fetch(url, md5, os.path.join(models_dir(), "matte", f"{name}.onnx"))
                    opts = ort.SessionOptions()
                    opts.intra_op_num_threads = render.threads()
                    _matte = (ort.InferenceSession(path, sess_options=opts,
                                                   providers=["CPUExecutionProvider"]), size)
                except Exception as e:  # noqa: BLE001 - colour and depth will do
                    _matte_failed_at = time.time()
                    log.warning("matting model unavailable (%s: %s); telling things apart by "
                                "colour and depth", type(e).__name__, e)
    return _matte


def matte_of(img: Image.Image) -> np.ndarray | None:
    """How surely each pixel belongs to the picture's subject, 0 to 1 (as rembg reads it)."""
    s = matte_session()
    if s is None:
        return None
    sess, size = s
    im = np.asarray(img.convert("RGB").resize((size, size), Image.LANCZOS), np.float32)
    im = (im / max(float(im.max()), 1e-6) - MEAN) / STD
    out = sess.run(None, {sess.get_inputs()[0].name: im.transpose(2, 0, 1)[None].astype(np.float32)})[0]
    pred = np.squeeze(out[:, 0]).astype(np.float32)
    pred = (pred - pred.min()) / max(float(pred.max() - pred.min()), 1e-6)
    mask = Image.fromarray((pred * 255).astype(np.uint8), "L").resize(img.size, Image.LANCZOS)
    return np.asarray(mask, np.float32) / 255.0


# ------------------------------------------------------------------ the painting
def warm() -> str:
    render.pipeline()
    depth_model()
    matte_session()
    return render.device()


def paint_png(prompt: str, kind: str = "thing", seed: int = 0) -> bytes:
    """Paint one thing (on its own) or one place (a wide view) and pack it for the page."""
    kind = kind if kind in STYLE else "thing"
    t0 = time.perf_counter()
    w, h = sizes(kind)
    img = render.generate(STYLE[kind].format(p=prompt.strip()[:300]), w, h, steps(), seed or seed_of(prompt, kind))
    t1 = time.perf_counter()
    rgb = np.asarray(img.convert("RGB"), np.float32) / 255.0
    raw = depth_of(img)
    t2 = time.perf_counter()
    if kind == "thing":
        matte = matte_of(img)
        alpha = relief.firm(matte) if matte is not None else relief.guess_mask(rgb, relief.normalize_depth(raw))
        near = relief.normalize_depth(raw, alpha)
    else:
        alpha = np.ones(raw.shape, np.float32)
        near = relief.normalize_depth(raw)
    t3 = time.perf_counter()
    angle, detail = relief.orientation(rgb)
    png = relief.pack(rgb, alpha, near, angle, detail)
    t4 = time.perf_counter()
    last.clear()
    last.update(kind=kind, paint_s=round(t1 - t0, 2), depth_s=round(t2 - t1, 2), matte_s=round(t3 - t2, 2),
                strokes_s=round(t4 - t3, 2), total_s=round(t4 - t0, 2), device=render.device())
    return png
