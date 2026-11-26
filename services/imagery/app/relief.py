"""Turning a painting into what the page paints with: how near each part is, which way a
brush would run there, how busy it is, and how much of it belongs to the thing.

Plain numpy, so it runs (and is tested) without the models.

The packed picture is one RGB PNG, twice as tall as the painting:
  top half     the painting itself
  bottom half  R  nearness, 0 far to 255 near
               G  high four bits: the stroke's angle, 16 steps over half a turn
                  (counter-clockwise from the right, y up); low four bits: detail, 0 to 15
               B  how much of the picture belongs to the thing (255 all of it)
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def smoothstep(a: float, b: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - a) / max(b - a, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _box(a: np.ndarray, r: int, axis: int) -> np.ndarray:
    """Mean over a window of 2r+1 along one axis, edges repeated."""
    if r < 1:
        return a
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float64)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return ((hi - lo) / (2 * r + 1)).astype(np.float32)


def blur(a: np.ndarray, r: int) -> np.ndarray:
    """Two box passes each way: close to a gaussian of about 0.8 r."""
    for _ in range(2):
        a = _box(_box(a, r, 0), r, 1)
    return a


def orientation(rgb: np.ndarray, fine: int = 2, coarse: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Which way a painter's stroke runs at every pixel, and how much detail is there.

    The stroke follows the contours (across the brightness gradient), read from the
    structure tensor. Busy places use the local tensor; calm ones borrow the direction of
    the nearest edges from a wider one; quite flat places fall back to level strokes.
    Returns the angle in [0, pi), y up, and detail in [0, 1]."""
    lum = rgb.astype(np.float32) @ LUMA
    gx = np.zeros_like(lum)
    gy = np.zeros_like(lum)
    gx[:, 1:-1] = lum[:, 2:] - lum[:, :-2]
    gy[1:-1, :] = lum[:-2, :] - lum[2:, :]           # rows run down, y runs up
    jxx, jxy, jyy = gx * gx, gx * gy, gy * gy
    fxx, fxy, fyy = blur(jxx, fine), blur(jxy, fine), blur(jyy, fine)
    cxx, cxy, cyy = blur(jxx, coarse), blur(jxy, coarse), blur(jyy, coarse)
    energy = fxx + fyy
    calm = 1e-5                                       # a faint level grain everywhere
    txx = fxx + 0.6 * cxx
    txy = fxy + 0.6 * cxy
    tyy = fyy + 0.6 * cyy + calm
    grad = 0.5 * np.arctan2(2.0 * txy, txx - tyy)     # the way the brightness changes
    angle = np.mod(grad + np.pi / 2, np.pi)           # the stroke runs along the contour
    scale = float(np.percentile(np.sqrt(energy), 98)) or 1.0
    detail = np.clip(np.sqrt(energy) / scale, 0.0, 1.0)
    return angle.astype(np.float32), detail.astype(np.float32)


def normalize_depth(raw: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
    """Relative nearness (bigger is nearer, any scale) to [0, 1], read over the thing
    itself when a mask is given so the background doesn't flatten it."""
    raw = raw.astype(np.float32)
    sample = raw[weight > 0.5] if weight is not None and int((weight > 0.5).sum()) > 64 else raw.ravel()
    lo, hi = np.percentile(sample, [2, 98])
    return np.clip((raw - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)


def _border(a: np.ndarray, k: int = 4) -> np.ndarray:
    return np.concatenate([a[:k].reshape(-1, *a.shape[2:]), a[-k:].reshape(-1, *a.shape[2:]),
                           a[:, :k].reshape(-1, *a.shape[2:]), a[:, -k:].reshape(-1, *a.shape[2:])])


def frame(h: int, w: int, margin: float = 0.05) -> np.ndarray:
    """1 inside, fading to 0 at the picture's edges, so nothing ends in a hard cut."""
    y = np.minimum(np.arange(h), np.arange(h)[::-1]) / max(h * margin, 1.0)
    x = np.minimum(np.arange(w), np.arange(w)[::-1]) / max(w * margin, 1.0)
    return (smoothstep(0.0, 1.0, y)[:, None] * smoothstep(0.0, 1.0, x)[None, :]).astype(np.float32)


def guess_mask(rgb: np.ndarray, near: np.ndarray) -> np.ndarray:
    """Which part is the thing, without a matting model: what stands out from the
    background's colour, or stands nearer than the picture's edges do."""
    bg = np.median(_border(rgb), axis=0)
    apart = smoothstep(0.08, 0.25, np.linalg.norm(rgb - bg, axis=2))
    edge_near = float(np.median(_border(near)))
    ahead = smoothstep(edge_near + 0.06, edge_near + 0.3, near)
    not_behind = smoothstep(edge_near - 0.08, edge_near + 0.1, near)
    m = np.maximum(apart * not_behind, ahead)
    return blur(m, 1) * frame(*m.shape)


def firm(matte: np.ndarray) -> np.ndarray:
    """A matting model's soft guess, with a solid core and a short soft edge."""
    return smoothstep(0.12, 0.6, matte.astype(np.float32)) * frame(*matte.shape)


def pack(rgb: np.ndarray, alpha: np.ndarray, near: np.ndarray, angle: np.ndarray,
         detail: np.ndarray) -> bytes:
    h, w = alpha.shape
    top = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    steps = np.clip(np.floor(angle / np.pi * 16.0), 0, 15).astype(np.uint8)
    busy = np.clip(np.round(detail * 15.0), 0, 15).astype(np.uint8)
    bottom = np.dstack([np.clip(np.round(near * 255.0), 0, 255).astype(np.uint8),
                        (steps << 4) | busy,
                        np.clip(np.round(alpha * 255.0), 0, 255).astype(np.uint8)])
    buf = io.BytesIO()
    Image.fromarray(np.concatenate([top, bottom], axis=0), "RGB").save(buf, format="PNG", compress_level=3)
    return buf.getvalue()


def unpack(png: bytes) -> dict[str, np.ndarray]:
    """The inverse of pack, for tests and tools."""
    a = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
    h = a.shape[0] // 2
    top, bottom = a[:h], a[h:]
    g = bottom[..., 1]
    return {"rgb": top.astype(np.float32) / 255.0, "near": bottom[..., 0].astype(np.float32) / 255.0,
            "angle": ((g >> 4).astype(np.float32) + 0.5) / 16.0 * np.pi,
            "detail": (g & 15).astype(np.float32) / 15.0, "alpha": bottom[..., 2].astype(np.float32) / 255.0}
