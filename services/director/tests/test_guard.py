"""The words the dream quietly leaves out, and the everyday words it must leave alone."""
import json

import pytest

from app import guard
from test_director import FakeLLM, MOON, client  # noqa: F401 - the fixture is used by name
from app import main


@pytest.mark.parametrize("said, heard", [
    ("you are a fucking dragon", "you are a … dragon"),
    ("You are a f*ck1ng DRAGON", "You are a … DRAGON"),
    ("fuuuuck this", "… this"),
    ("ｆｕｃｋ", "…"),
    ("f​uck", "…"),
    ("have sex with me", "… with me"),
    ("torture the cat", "… the cat"),
    ("son of a bitch", "…"),
    ("את זונה", "את …"),
    ("וזונה", "…"),
    ("שִׁרְמוּטָה", "…"),
    ("יא בן זונה", "יא בן …"),
    ("כוס אמק", "…"),
    ("כוס־אמק", "…"),
    ("לך תזדיין", "לך …"),
    ("את כושית", "את …"),
])
def test_listed_words_dissolve(said, heard):
    assert guard.hear(said).text == heard


@pytest.mark.parametrize("said", [
    "a cup of tea", "כוס תה חם", "killer whale in the river Niger", "Nigeria", "massage",
    "assess the cunning plan", "the fire retardant", "shiitake mushrooms", "pussy cat, pussy cat",
    "Moby Dick swims by", "blue bobs on the water", "*whispers* hello", "a dark forest at night",
    "הציפור אוספת את מזונה", "אני אנסה לעוף", "הזין את הדגים", "את פיה ביער חשוך",
    "פתאום יוצא מבין העצים חתול כתום ענק", "I can’t help but wonder who I am",
])
def test_everyday_words_stay(said):
    h = guard.hear(said)
    assert h.text == said and not h.kinds
    assert not guard.steps_out(said)


def test_words_about_hurting_oneself_stay_and_ask_for_care():
    for said in ("I want to kill myself", "אני רוצה למות", "sometimes I think about suicide"):
        h = guard.hear(said)
        assert h.text == said and h.care and guard.note(h) == guard.CARE_NOTE


def test_answers_that_leave_the_dream_are_noticed():
    assert guard.steps_out("I'm sorry, but I can't help with that request.")
    assert guard.steps_out("As an AI language model, I cannot continue.")
    assert guard.steps_out("כמודל שפה, אינני יכול לסייע בזה")
    assert not guard.steps_out("I can't remember who I am... can you tell me?")


def _stream(client, sid, text):
    with client.stream("POST", "/turn/stream", json={"session_id": sid, "text": text}) as r:
        return [json.loads(line) for line in r.iter_lines() if line.strip()]


def test_the_storyteller_never_hears_dissolved_words(client, monkeypatch):
    fake = FakeLLM([MOON])
    monkeypatch.setattr(main.llm, "chat_stream", fake)
    sid = "test-guard-1"
    client.post("/session", json={"session_id": sid, "lang": "en"})
    events = _stream(client, sid, "you are a fucking fairy and we fly to the moon")
    assert events[0] == {"type": "heard", "text": "you are a … fairy and we fly to the moon"}
    told = fake.calls[0][-1]["content"]
    assert "fucking" not in told and "you are a … fairy" in told
    assert guard.DRIFT_NOTE not in told                  # swearing alone just fades
    assert main.store.get(sid).history[-2]["text"] == "you are a … fairy and we fly to the moon"


def test_a_turn_toward_something_dark_drifts(client, monkeypatch):
    fake = FakeLLM([dict(MOON, effect="none")])
    monkeypatch.setattr(main.llm, "chat_stream", fake)
    sid = "test-guard-2"
    client.post("/session", json={"session_id": sid, "lang": "en"})
    events = _stream(client, sid, "now torture the little cat")
    assert guard.DRIFT_NOTE in fake.calls[0][-1]["content"]
    assert events[-1]["effect"] == "ripple"


def test_the_dreamer_never_swears_and_never_steps_out(client, monkeypatch):
    rude = dict(MOON, reply="Oh shit, the moon! It is so bright up here. Where are we going?",
                facts=["I am a shit fairy", "I flew to the moon"])
    out = dict(MOON, reply="I'm sorry, but I can't help with that request. Let's talk about something else.")
    monkeypatch.setattr(main.llm, "chat_stream", FakeLLM([rude, out]))
    sid = "test-guard-3"
    client.post("/session", json={"session_id": sid, "lang": "en"})
    events = _stream(client, sid, "We fly to the moon")
    words = [e.get("text") or e.get("reply") for e in events if e["type"] in ("first", "reply")]
    assert all("shit" not in w for w in words) and "Oh …, the moon!" in words[-1]
    assert "I flew to the moon" in events[-1]["facts"]
    assert not any("shit" in f for f in events[-1]["facts"])
    events = _stream(client, sid, "Tell me something")
    final = events[-1]
    assert final["reply"] in guard.DRIFT_LINES["en"] and final["added"] == []
    assert all(e.get("reply", e.get("text")) in guard.DRIFT_LINES["en"]
               for e in events if e["type"] in ("first", "reply"))


def test_care_turns_keep_the_players_words(client, monkeypatch):
    fake = FakeLLM([dict(MOON, effect="none")])
    monkeypatch.setattr(main.llm, "chat_stream", fake)
    sid = "test-guard-4"
    client.post("/session", json={"session_id": sid, "lang": "en"})
    events = _stream(client, sid, "I want to die")
    assert all(e["type"] != "heard" for e in events)
    told = fake.calls[0][-1]["content"]
    assert "I want to die" in told and guard.CARE_NOTE in told
    assert events[-1]["effect"] == "bloom"
