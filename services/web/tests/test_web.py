import json
import re
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main

ROOT = Path(__file__).resolve().parents[2]


STATE = {"director": "ok", "stt_text": "you are a fairy", "turn": None}


def ndjson(final: dict) -> httpx.Response:
    """The director's streamed answer: the words first (unless it failed), then the scene."""
    lines = []
    if STATE.get("first") and not final.get("fallback"):
        lines.append({"type": "first", "text": STATE["first"], "lang": final.get("lang", "en"),
                      "voice": final.get("voice", "neutral")})
    if not final.get("fallback"):
        lines.append({"type": "reply", "reply": final["reply"], "lang": final.get("lang", "en"),
                      "voice": final.get("voice", "neutral")})
    lines.append({"type": "final", **final})
    return httpx.Response(200, content="".join(json.dumps(x) + "\n" for x in lines).encode(),
                          headers={"content-type": "application/x-ndjson"})


def fake_backend(request: httpx.Request) -> httpx.Response:
    path, host = request.url.path, request.url.host
    if path == "/healthz":
        if host == "voice":
            return httpx.Response(200, json={"status": "ok", "stt": {"state": "ready"},
                                             "tts": {"state": "ready"}})
        return httpx.Response(200, json={"status": STATE["director"], "detail": "fake"})
    if path in ("/session", "/reset"):
        return httpx.Response(200, json={"reply": "Where am I?", "lang": "en", "voice": "neutral",
                                         "resumed": False, "scene": {"setting": {"kind": "void"}, "things": []}})
    if path in ("/turn", "/turn/stream"):
        if STATE.get("drop"):                          # the director restarts mid-turn
            STATE["drop"] -= 1
            raise httpx.RemoteProtocolError("peer closed connection", request=request)
        if STATE["director"] != "ok":
            return httpx.Response(503, json={"detail": {"state": "pulling"}})
        final = STATE["turn"] or {"reply": "A fairy... yes.", "lang": "en", "voice": "female",
                                  "scene": {"setting": {"kind": "forest"}, "things": [{"id": "t", "kind": "tree"}]},
                                  "added": ["t"], "changed": [], "removed": [], "effect": "sparkle"}
        if path == "/turn":
            return httpx.Response(200, json=final)
        return ndjson(final)
    if path == "/tts":
        return httpx.Response(200, content=b"RIFF....WAVE", headers={"content-type": "audio/wav"})
    if path == "/stt":
        return httpx.Response(200, json={"text": STATE["stt_text"]})
    return httpx.Response(404)


@pytest.fixture
def client(monkeypatch):
    STATE.update(director="ok", stt_text="you are a fairy", turn=None, drop=0, first=None)
    monkeypatch.setattr(main, "http", httpx.AsyncClient(transport=httpx.MockTransport(fake_backend)))
    monkeypatch.setattr(main, "HEARTBEAT_S", 0.05)
    with TestClient(main.app) as c:
        import time
        for _ in range(100):           # wait for the first heartbeat
            if main.monitor.ready("director") and main.monitor.ready("stt"):
                break
            time.sleep(0.02)
        yield c


def _collect(ws, last, limit=60):
    """Every message up to and including the first of type `last`."""
    out = []
    for _ in range(limit):
        m = ws.receive_json()
        out.append(m)
        if m["type"] == last:
            return out
    raise AssertionError(f"no {last} in {[m['type'] for m in out]}")


def _until(ws, kind, limit=40):
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
        got = _collect(ws, "timings")
        kinds = [m["type"] for m in got]
        # the words come first and the dreamer starts talking; the scene follows
        assert kinds.index("reply") < kinds.index("scene")
        assert next(m for m in got if m["type"] == "reply")["text"] == "A fairy... yes."
        assert next(m for m in got if m["type"] == "scene")["scene"]["setting"]["kind"] == "forest"
        audio = next(m for m in got if m["type"] == "audio")
        assert client.get(audio["url"]).headers["content-type"] == "audio/wav"


def test_spoken_turn_is_transcribed_first(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "hello", "session": "abcdef456", "lang": "en"})
        _until(ws, "audio")
        ws.send_json({"type": "audio", "mime": "audio/webm", "lang": "en"})
        ws.send_bytes(b"\x1aE\xdf\xa3fake-webm")
        assert _until(ws, "transcript")["text"] == "you are a fairy"
        _until(ws, "reply")


def test_words_wait_for_a_sleeping_storyteller_and_then_flow(client):
    STATE["director"] = "loading"
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "hello", "session": "abcdef789", "lang": "en"})
        _until(ws, "session")
        ws.send_json({"type": "text", "text": "you are a fairy", "lang": None})
        assert _until(ws, "waiting")["for"] == "director"
        STATE["director"] = "ok"                       # the model finished loading
        assert _until(ws, "reply")["text"] == "A fairy... yes."


def test_silence_and_failures_become_notices(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "hello", "session": "abcdef999", "lang": "en"})
        _until(ws, "audio")
        STATE["stt_text"] = ""
        ws.send_json({"type": "audio", "mime": "audio/webm", "lang": "en"})
        ws.send_bytes(b"\x1aE\xdf\xa3fake-webm")
        assert _until(ws, "notice")["key"] == "not_heard"
        STATE["turn"] = {"reply": "Your voice dissolves in the mist...", "fallback": True, "error": "bad json"}
        ws.send_json({"type": "text", "text": "fly to the moon", "lang": None})
        n = _until(ws, "notice")
        assert n["key"] == "stumbled" and n["detail"] == "bad json"


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
        if request.url.path == "/turn/stream":
            return ndjson({
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


def test_first_sentence_is_split_off_when_long_enough():
    assert main.split_first_sentence("Snow... it feels so cold on my fur. A fox? But why this place?") == [
        "Snow... it feels so cold on my fur.", "A fox? But why this place?"]
    assert main.split_first_sentence("קול... שמעתי קול בתוך החושך. מי שם? ומי... מי אני?") == [
        "קול... שמעתי קול בתוך החושך.", "מי שם? ומי... מי אני?"]
    assert main.split_first_sentence("Short. Tiny.") == ["Short. Tiny."]
    assert main.split_first_sentence("One long sentence without any ending at all") == [
        "One long sentence without any ending at all"]


def test_long_reply_arrives_as_two_clips(client):
    STATE["turn"] = {"reply": "Snow... it feels so cold on my fur. A fox? But why this place?",
                     "lang": "en", "voice": "female", "scene": None}
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "hello", "session": "abcdef555", "lang": "en"})
        _until(ws, "audio")
        ws.send_json({"type": "text", "text": "you are a fox", "lang": None})
        first = _until(ws, "audio")
        second = _until(ws, "audio")
        assert first["text"] == "Snow... it feels so cold on my fur." and second["text"].startswith("A fox?")
        assert _until(ws, "timings")["to_first_sound_s"] >= 0


def test_a_storyteller_restart_mid_turn_only_delays_the_answer(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "hello", "session": "abcdef321", "lang": "en"})
        _until(ws, "session")
        STATE["drop"] = 1
        ws.send_json({"type": "text", "text": "you are a fairy", "lang": None})
        assert _until(ws, "waiting")["for"] == "director"
        assert _until(ws, "reply")["text"] == "A fairy... yes."


def test_first_sentence_is_spoken_before_the_rest_is_written(client):
    STATE["first"] = "A fairy of silver light, yes."
    STATE["turn"] = {"reply": "A fairy of silver light, yes. My wings hum softly in the dark.",
                     "lang": "en", "voice": "female", "scene": {"setting": {"kind": "forest"}, "things": []},
                     "added": [], "changed": [], "removed": [], "effect": "sparkle"}
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "hello", "session": "abcdef654", "lang": "en"})
        _until(ws, "audio")
        ws.send_json({"type": "text", "text": "you are a fairy", "lang": None})
        got = _collect(ws, "timings")
        replies = [m["text"] for m in got if m["type"] == "reply"]
        clips = [m["text"] for m in got if m["type"] == "audio"]
        assert replies == [STATE["first"], STATE["turn"]["reply"]]
        assert clips == ["A fairy of silver light, yes.", "My wings hum softly in the dark."]


def test_a_long_first_sentence_is_cut_at_its_first_comma():
    he = "דרקון כחול נוחת לידי, והכתר שלו משתקף בעיניים שלי. אני מרגישה כוח חדש."
    assert main.split_for_speech(he) == ["דרקון כחול נוחת לידי,", "והכתר שלו משתקף בעיניים שלי.",
                                         "אני מרגישה כוח חדש."]
    short = "A fairy, yes. My wings hum softly in the dark night."
    assert main.split_for_speech(short) == [short]
    too_short_a_clause = "Snow falls, and the long night is quiet and cold and white."
    assert main.split_for_speech(too_short_a_clause) == [too_short_a_clause]


def test_words_said_while_the_dreamer_answers_come_next(client):
    import time
    seen = []
    real = main.Dreamer.director_events

    async def slow(self, text, lang_hint):
        seen.append(text)
        if len(seen) == 1:
            import asyncio
            await asyncio.sleep(0.3)            # the first answer is still being written
        async for ev in real(self, text, lang_hint):
            yield ev

    main.Dreamer.director_events = slow
    try:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "hello", "session": "abcdef777", "lang": "en"})
            _until(ws, "audio")
            ws.send_json({"type": "text", "text": "you are a fairy", "lang": None})
            time.sleep(0.05)
            ws.send_json({"type": "text", "text": "fly to", "lang": None})
            ws.send_json({"type": "text", "text": "the moon", "lang": None})
            assert _until(ws, "notice")["key"] == "busy"
            _until(ws, "timings")
            _until(ws, "timings")
    finally:
        main.Dreamer.director_events = real
    assert seen == ["you are a fairy", "fly to the moon"]


def test_a_bad_dream_id_gets_a_fresh_one_the_page_can_keep(client):
    sent = []
    real = main.Dreamer.director_events

    async def spy(self, text, lang_hint):
        sent.append(self.session)
        async for ev in real(self, text, lang_hint):
            yield ev

    main.Dreamer.director_events = spy
    try:
        for bad in ("../../etc/passwd", "<script>", "x", "abcdefg\n", 12345678, None):
            with client.websocket_connect("/ws") as ws:
                ws.send_json({"type": "hello", "session": bad, "lang": "en"})
                got = _until(ws, "session")
                assert got["session"] != bad and re.fullmatch(r"[A-Za-z0-9_-]{6,64}", got["session"])
                ws.send_json({"type": "text", "text": "you are a fairy", "lang": None})
                assert _until(ws, "reply")["text"] == "A fairy... yes."
                assert sent[-1] == got["session"]
        with client.websocket_connect("/ws") as ws:            # a good id is kept as it is
            ws.send_json({"type": "hello", "session": "good_id-123", "lang": "en"})
            assert _until(ws, "session")["session"] == "good_id-123"
    finally:
        main.Dreamer.director_events = real


def test_words_before_hello_still_reach_the_storyteller(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "text", "text": "you are a fairy", "lang": None})
        assert _until(ws, "reply")["text"] == "A fairy... yes."


def test_a_flood_of_words_while_busy_stays_one_turn_of_words(client):
    seen = []
    real = main.Dreamer.director_events

    async def slow(self, text, lang_hint):
        seen.append(text)
        if len(seen) == 1:
            import asyncio
            await asyncio.sleep(0.3)
        async for ev in real(self, text, lang_hint):
            yield ev

    main.Dreamer.director_events = slow
    try:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "hello", "session": "abcdef888", "lang": "en"})
            _until(ws, "audio")
            ws.send_json({"type": "text", "text": "you are a fairy", "lang": None})
            for _ in range(20):
                ws.send_json({"type": "text", "text": "moon " * 400, "lang": None})
            _until(ws, "timings")
            _until(ws, "timings")
    finally:
        main.Dreamer.director_events = real
    assert len(seen) == 2 and len(seen[1]) <= main.MAX_CHARS
