import asyncio
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main, paint, relief


def tiny(prompt: str) -> bytes:
    rng = np.random.default_rng(abs(hash(prompt)) % 1000)
    h = w = 8
    return relief.pack(rng.random((h, w, 3)), np.ones((h, w)), rng.random((h, w)),
                       np.zeros((h, w)), np.zeros((h, w)))


@pytest.fixture
def api(tmp_path, monkeypatch):
    calls = []

    def fake_paint(prompt, kind="thing", seed=0):
        calls.append((prompt, kind))
        time.sleep(0.05)
        return tiny(prompt + kind)

    async def no_scheduler():
        return None

    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(paint, "paint_png", fake_paint)
    monkeypatch.setattr(main, "_connect", no_scheduler)
    monkeypatch.setattr(main, "LOCAL", False)
    main.state.update(client=None, local="ready on cpu", paintings=0, reused=0)
    with TestClient(main.app) as client:
        yield client, calls


def test_health_says_it_paints(api):
    client, _ = api
    h = client.get("/healthz").json()
    assert h["status"] == "ok" and h["paint"] is True and h["waiting"] == 0


def test_a_painting_is_made_once_and_kept(api):
    client, calls = api
    a = client.post("/paint", json={"prompt": "a  blue whale", "kind": "thing"})
    b = client.post("/paint", json={"prompt": "a blue whale", "kind": "thing"})
    assert a.status_code == b.status_code == 200
    assert a.headers["X-Painted"] == "new" and b.headers["X-Painted"] == "kept"
    assert a.content == b.content and calls == [("a blue whale", "thing")]
    out = relief.unpack(a.content)
    assert out["rgb"].shape == (8, 8, 3)
    scene = client.post("/paint", json={"prompt": "a blue whale", "kind": "scene"})
    assert scene.status_code == 200 and len(calls) == 2
    assert client.get("/healthz").json()["reused"] == 1


def test_the_same_painting_asked_twice_at_once_is_made_once(api):
    client, calls = api
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(3) as pool:
        rs = list(pool.map(lambda _: client.post("/paint", json={"prompt": "a red fox"}), range(3)))
    assert [r.status_code for r in rs] == [200, 200, 200]
    assert calls == [("a red fox", "thing")]


def test_nothing_to_paint_with_yet(api):
    client, _ = api
    main.state["local"] = "loading"
    r = client.post("/paint", json={"prompt": "a moth"})
    assert r.status_code == 503


def test_the_players_things_are_painted_first():
    order, gate = [], threading.Event()

    def job(name):
        def run():
            if name == "first":
                gate.wait(5)
            order.append(name)
            return name
        return run

    async def scenario():
        p = main.Painter()
        worker = p.start()
        first = asyncio.ensure_future(p.run(job("first"), 1))
        await asyncio.sleep(0.05)                 # the painter is busy with it
        rest = [asyncio.ensure_future(p.run(job(n), pr)) for n, pr in
                (("scene", 2), ("self", 1), ("named", 0))]
        await asyncio.sleep(0.05)
        gate.set()
        await asyncio.gather(first, *rest)
        worker.cancel()

    asyncio.run(scenario())
    assert order == ["first", "named", "self", "scene"]


def test_old_paintings_make_room(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(main, "KEEP", 3)
    shelf = main._shelf()
    for i in range(5):
        main._keep(f"{shelf}/{i}.png", b"x")
        t = time.time() - 100 + i
        import os
        os.utime(f"{shelf}/{i}.png", (t, t))
    main._keep(f"{shelf}/new.png", b"x")
    import os
    assert sorted(os.listdir(shelf)) == ["3.png", "4.png", "new.png"]
