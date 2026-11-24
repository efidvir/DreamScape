import numpy as np
import pytest

from app import relief


def stripes(fn, n=96):
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    lum = 0.5 + 0.4 * np.sin(fn(x, n - 1 - y))          # n-1-y: y runs up
    return np.repeat(lum[..., None], 3, axis=2)


def mid(a):
    return a[24:-24, 24:-24]


def angle_error(a, want):
    d = np.abs(np.mod(a - want + np.pi / 2, np.pi) - np.pi / 2)   # strokes have no head
    return float(np.median(d))


def test_strokes_follow_the_contours():
    across = stripes(lambda x, y: y * 0.5)        # light changes going up: level contours
    up = stripes(lambda x, y: x * 0.5)            # light changes going right: upright contours
    slant = stripes(lambda x, y: (x + y) * 0.35)  # contours run up-left, at 135 degrees
    assert angle_error(mid(relief.orientation(across)[0]), 0.0) < 0.1
    assert angle_error(mid(relief.orientation(up)[0]), np.pi / 2) < 0.1
    assert angle_error(mid(relief.orientation(slant)[0]), 3 * np.pi / 4) < 0.1


def test_flat_paint_gets_level_strokes_and_no_detail():
    angle, detail = relief.orientation(np.full((40, 40, 3), 0.4, np.float32))
    assert angle_error(angle, 0.0) < 1e-3
    assert detail.max() == 0


def test_detail_is_where_the_picture_is_busy():
    img = np.full((80, 80, 3), 0.3, np.float32)
    img[:, 40:] = stripes(lambda x, y: x * 1.3, 80)[:, 40:]
    _, detail = relief.orientation(img)
    assert detail[:, 50:70].mean() > 0.5 > detail[:, 5:25].mean() + 0.3


def test_blur_keeps_the_mean_and_flat_fields():
    rng = np.random.default_rng(1)
    a = rng.random((50, 70)).astype(np.float32)
    assert abs(relief.blur(a, 3).mean() - a.mean()) < 0.01
    assert np.allclose(relief.blur(np.full((20, 20), 0.7, np.float32), 4), 0.7, atol=1e-5)


def test_depth_is_read_over_the_thing():
    raw = np.zeros((40, 40), np.float32)
    raw[10:30, 10:30] = np.linspace(5, 6, 20)[None, :]      # the thing, gently curved
    mask = np.zeros_like(raw)
    mask[10:30, 10:30] = 1
    over_all = relief.normalize_depth(raw)
    over_thing = relief.normalize_depth(raw, mask)
    span = lambda d: float(d[10:30, 10:30].max() - d[10:30, 10:30].min())
    assert span(over_thing) > 0.9 and span(over_all) < 0.2


def test_the_thing_is_told_from_a_dark_background():
    n = 64
    y, x = np.mgrid[0:n, 0:n]
    disc = (x - 32) ** 2 + (y - 32) ** 2 < 14 ** 2
    rgb = np.full((n, n, 3), 0.04, np.float32)
    rgb[disc] = [0.2, 0.5, 0.9]
    near = np.where(disc, 0.9, 0.1).astype(np.float32)
    m = relief.guess_mask(rgb, near)
    assert m[32, 32] > 0.9 and m[3, 3] < 0.05 and m[32, 60] < 0.05
    dark = np.full((n, n, 3), 0.04, np.float32)                     # a dark thing: depth tells it
    assert relief.guess_mask(dark, near)[32, 32] > 0.9


def test_mattes_get_a_solid_core_and_no_hard_frame():
    matte = np.zeros((60, 60), np.float32)
    matte[10:50, 10:50] = 0.8
    a = relief.firm(matte)
    assert a[30, 30] == pytest.approx(1.0, abs=1e-3)
    assert a[0, 30] == 0 and a[5, 5] == 0


def test_pack_round_trip():
    rng = np.random.default_rng(2)
    h, w = 24, 40
    rgb = rng.random((h, w, 3)).astype(np.float32)
    alpha, near = rng.random((h, w)).astype(np.float32), rng.random((h, w)).astype(np.float32)
    angle = (rng.random((h, w)) * np.pi).astype(np.float32)
    detail = rng.random((h, w)).astype(np.float32)
    png = relief.pack(rgb, alpha, near, angle, detail)
    out = relief.unpack(png)
    assert out["rgb"].shape == (h, w, 3)
    assert np.abs(out["rgb"] - rgb).max() <= 0.5 / 255 + 1e-6
    assert np.abs(out["alpha"] - alpha).max() <= 0.5 / 255 + 1e-6
    assert np.abs(out["near"] - near).max() <= 0.5 / 255 + 1e-6
    assert np.abs(out["detail"] - detail).max() <= 0.5 / 15 + 1e-6
    d = np.abs(np.mod(out["angle"] - angle + np.pi / 2, np.pi) - np.pi / 2)
    assert d.max() <= np.pi / 32 + 1e-5
