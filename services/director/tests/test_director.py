import json
import os
import tempfile

import jsonschema
import pytest

os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="dreams-")

from fastapi.testclient import TestClient  # noqa: E402

from app import main, prompt  # noqa: E402
from app.schema import parse_turn, turn_schema  # noqa: E402
from app.story import SessionStore  # noqa: E402

FAIRY = json.loads(prompt.EXAMPLES[1][1])
MOON = json.loads(prompt.EXAMPLES[2][1])


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def __call__(self, messages, schema, max_tokens=420):
        self.calls.append(messages)
        return self.replies.pop(0)


@pytest.fixture
def client(monkeypatch):
    async def no_prepare():
        return None
    monkeypatch.setattr(main.llm, "prepare", no_prepare)
    main.llm.state = "ready"
    with TestClient(main.app) as c:
        yield c


def test_examples_match_schema():
    schema = turn_schema()
    for _, out in prompt.EXAMPLES:
        jsonschema.validate(json.loads(out), schema)


def test_parse_turn_survives_garbage():
    out = parse_turn({"reply": 5, "add": [{"kind": "unicornz", "count": 999}, "x"],
                      "setting": {"kind": "mars"}, "clarity": "high", "remove": [None, "A B"]})
    assert out.reply == ""
    assert out.add[0].kind == "other" and out.add[0].count <= 6
    assert out.setting_kind == "same"
    assert out.clarity is None
    assert out.remove == ["a-b"]


def test_story_flow(client, monkeypatch):
    fake = FakeLLM([FAIRY, MOON])
    monkeypatch.setattr(main.llm, "chat_json", fake)
    sid = "test-session-1"
    r = client.post("/session", json={"session_id": sid, "lang": "he"}).json()
    assert r["resumed"] is False and r["reply"] in prompt.OPENINGS["he"]
    assert r["scene"]["setting"]["kind"] == "void"

    r = client.post("/turn", json={"session_id": sid, "text": "את פיה, ואת נמצאת ביער חשוך"}).json()
    assert r["lang"] == "he" and r["voice"] == "female"
    assert r["scene"]["self"]["form"] == "fairy"
    assert r["scene"]["setting"]["kind"] == "forest"
    assert set(r["added"]) == {"tall-trees", "fireflies"}
    assert r["effect"] == "sparkle"

    r = client.post("/turn", json={"session_id": sid,
                                   "text": "You build a spaceship and fly to the moon"}).json()
    kinds = {t["kind"] for t in r["scene"]["things"]}
    assert r["lang"] == "en"
    assert r["scene"]["setting"]["kind"] == "moon"
    assert "tree" not in kinds and "spaceship" in kinds and "planet" in kinds
    assert set(r["removed"]) == {"tall-trees", "fireflies"}
    assert r["effect"] == "warp"
    assert r["scene"]["self"]["form"] == "fairy"      # the dreamer travels along

    # the prompt carries the state and keeps a stable, cacheable prefix
    first, second = fake.calls
    assert first[: 1 + 2 * len(prompt.EXAMPLES)] == second[: 1 + 2 * len(prompt.EXAMPLES)]
    assert "tall-trees" in second[-1]["content"]

    # resuming returns the same dream
    r = client.post("/session", json={"session_id": sid, "lang": "he"}).json()
    assert r["resumed"] is True and r["scene"]["setting"]["kind"] == "moon"
    again = SessionStore(os.environ["DATA_DIR"]).get(sid)
    assert again and again.setting["kind"] == "moon" and again.turn == 2


def test_fallback_when_model_not_ready(client):
    main.llm.state = "pulling"
    r = client.post("/turn", json={"session_id": "test-session-2", "text": "hello there"}).json()
    assert r["fallback"] is True and r["reply"] == prompt.FALLBACK["en"]
    assert r["effect"] == "ripple"
    main.llm.state = "ready"


def test_dream_forgets_oldest_things(client, monkeypatch):
    many = []
    for i in range(6):
        many.append({**FAIRY, "add": [{"id": f"rock-{i}-{j}", "kind": "rock", "label": "", "color": "",
                                       "size": "small", "count": 1, "where": "ground",
                                       "motion": "still"} for j in range(4)],
                     "setting": {"kind": "same", "time": "same", "weather": "same"}})
    monkeypatch.setattr(main.llm, "chat_json", FakeLLM(many))
    sid = "test-session-3"
    client.post("/session", json={"session_id": sid, "lang": "en"})
    for _ in range(6):
        r = client.post("/turn", json={"session_id": sid, "text": "more rocks"}).json()
    assert len(r["scene"]["things"]) <= 16
    assert all(t["id"].startswith("rock-5") or t["born"] >= 3 for t in r["scene"]["things"])


def test_bad_session_id(client):
    assert client.post("/session", json={"session_id": "../../etc", "lang": "en"}).status_code in (400, 422)


def test_healthz(client):
    h = client.get("/healthz").json()
    assert h["service"] == "director" and h["status"] == "ok"
