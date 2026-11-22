import re
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main

ROOT = Path(__file__).resolve().parents[2]


def fake_backend(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/healthz":
        return httpx.Response(200, json={"status": "ok", "detail": "fake"})
    if path in ("/session", "/reset"):
        return httpx.Response(200, json={"reply": "Where am I?", "lang": "en", "voice": "neutral",
                                         "resumed": False, "scene": {"setting": {"kind": "void"}, "things": []}})
    if path == "/turn":
        return httpx.Response(200, json={"reply": "A fairy... yes.", "lang": "en", "voice": "female",
                                         "scene": {"setting": {"kind": "forest"}, "things": [{"id": "t", "kind": "tree"}]},
                                         "added": ["t"], "changed": [], "removed": [], "effect": "sparkle"})
    if path == "/tts":
        return httpx.Response(200, content=b"RIFF....WAVE", headers={"content-type": "audio/wav"})
    if path == "/stt":
        return httpx.Response(200, json={"text": "you are a fairy"})
    return httpx.Response(404)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "http", httpx.AsyncClient(transport=httpx.MockTransport(fake_backend)))
    with TestClient(main.app) as c:
        yield c


def _until(ws, kind, limit=12):
    seen = []
    for _ in range(limit):
        m = ws.receive_json()
        seen.append(m["type"])
        if m["type"] == kind:
            return m
    raise AssertionError(f"no {kind} in {seen}")


def test_typed_turn_flows_to_scene_reply_and_voice(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "hello", "session": "abcdef123", "lang": "en"})
        assert _until(ws, "session")["reply"] == "Where am I?"
        _until(ws, "audio")
        ws.send_json({"type": "text", "text": "you are a fairy", "lang": None})
        assert _until(ws, "scene")["scene"]["setting"]["kind"] == "forest"
        assert _until(ws, "reply")["text"] == "A fairy... yes."
        audio = _until(ws, "audio")
        assert client.get(audio["url"]).headers["content-type"] == "audio/wav"


def test_spoken_turn_is_transcribed_first(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "hello", "session": "abcdef456", "lang": "en"})
        _until(ws, "audio")
        ws.send_json({"type": "audio", "mime": "audio/webm", "lang": "en"})
        ws.send_bytes(b"\x1aE\xdf\xa3fake-webm")
        assert _until(ws, "transcript")["text"] == "you are a fairy"
        _until(ws, "reply")


def test_page_and_vendored_three_are_served(client):
    assert "importmap" in client.get("/").text
    assert client.get("/vendor/three/three.module.js").status_code == 200


def test_every_director_word_has_a_builder():
    """The director's vocabulary and the 3D renderer must stay in step."""
    vocab = (ROOT / "director" / "app" / "vocab.py").read_text()
    kinds = re.findall(r'^\s+"([a-z]+)": "', vocab.split("THINGS = {")[1].split("}")[0], re.M)
    settings = re.findall(r'^\s+"([a-z]+)",', vocab.split("SETTINGS = [")[1].split("]")[0], re.M)
    js = (ROOT / "web" / "static" / "js" / "dream")
    builders = (js / "things.js").read_text().split("export const BUILDERS = {")[1].split("};")[0]
    places = (js / "settings.js").read_text().split("const BUILD = {")[1]
    assert len(kinds) == 40 and len(settings) == 13
    for k in kinds:
        assert re.search(rf"\b{k}\b", builders), f"no 3D builder for '{k}'"
    for s in settings:
        assert re.search(rf"^  {s}\(r", places, re.M), f"no landscape for '{s}'"


def test_pictures_are_requested_for_new_things_and_the_dreamer(client, monkeypatch):
    monkeypatch.setattr(main, "IMAGERY_URL", "http://imagery")
    main.monitor.status["imagery"] = {"status": "ok"}
    asked = []

    def backend(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/imagine":
            asked.append(request.read().decode())
            return httpx.Response(200, content=b"\x89PNG fake", headers={"content-type": "image/png"})
        if request.url.path == "/turn":
            return httpx.Response(200, json={
                "reply": "A cheese!", "lang": "en", "voice": "female", "backdrop": "",
                "scene": {"setting": {"kind": "moon"}, "self": {"form": "fairy", "label": "silver fairy", "color": "silver"},
                          "things": [{"id": "cheese", "kind": "cheese", "label": "glowing cheese", "color": "gold", "count": 1},
                                     {"id": "trees", "kind": "tree", "label": "", "color": "", "count": 20}]},
                "added": ["cheese", "trees"], "changed": [], "removed": [], "effect": "none"})
        return fake_backend(request)

    monkeypatch.setattr(main, "http", httpx.AsyncClient(transport=httpx.MockTransport(backend)))
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "hello", "session": "abcdef789", "lang": "en"})
        _until(ws, "audio")
        ws.send_json({"type": "text", "text": "there is cheese", "lang": None})
        got = {_until(ws, "sprite")["id"], _until(ws, "sprite")["id"]}
    assert got == {"cheese", "self"}
    assert any("gold cheese" in a or "glowing cheese" in a for a in asked)
    assert not any('"trees"' in a for a in asked)          # groves stay procedural
    main.monitor.status["imagery"] = {"status": "off"}
