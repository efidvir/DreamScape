"""Who speaks and who listens, from how the Hebrew is written."""
import json

import pytest

from app import hebrew
from test_director import FakeLLM, FAIRY, client  # noqa: F401 - the fixture is used by name
from app import main


@pytest.mark.parametrize("text, gender", [
    ("אני מרגישה את החום של המדורה", "female"), ("אני לא יודעת מי אני", "female"),
    ("אני כל כך שמחה", "female"), ("אני עפה מעל הים", "female"), ("אני ענן קטנה", "female"),
    ("אני מרגיש קור", "male"), ("ואני ממשיך לטוס", "male"), ("אני דרקון גדול", "male"),
    ("אני רוצה לעוף", ""), ("אני רואה אור", ""), ("אני מקווה שכן", ""), ("אני רוצה מוזיקה", ""),
    ("החתול הכתום מופיע שוב", ""), ("I am a fairy", ""),
])
def test_how_a_speaker_speaks_of_themselves(text, gender):
    assert hebrew.self_gender(text) == gender


@pytest.mark.parametrize("text, gender", [
    ("מה את רואה עכשיו?", "female"), ("את רוצה לבוא איתי?", "female"), ("ואת? מה איתך?", "female"),
    ("מה אתה מרגיש?", "male"), ("לאן אתה הולך?", "male"),
    ("אני עושה את זה בשבילך", ""), ("את הדלת פתחתי לאט", ""), ("את זה אני לא יודעת", ""),
    ("החום של המדורה מחמם את הידיים", ""),
])
def test_how_the_dreamer_speaks_to_the_player(text, gender):
    assert hebrew.addressed(text) == gender


def test_the_voice_learns_both_genders_from_the_words(client, monkeypatch):
    she = dict(FAIRY, voice="same", reply="אני מרגישה את הרוח בכנפיים שלי. לאן את הולכת עכשיו?")
    fake = FakeLLM([she])
    monkeypatch.setattr(main.llm, "chat_stream", fake)
    sid = "test-hebrew-1"
    client.post("/session", json={"session_id": sid, "lang": "he"})
    with client.stream("POST", "/turn/stream", json={"session_id": sid, "text": "אני הולכת ליער"}) as r:
        events = [json.loads(line) for line in r.iter_lines() if line.strip()]
    spoken = [e for e in events if e["type"] in ("first", "reply")]
    assert spoken and all(e["voice"] == "female" and e["listener"] == "female" for e in spoken)
    sess = main.store.get(sid)
    assert sess.voice == "female" and sess.player == "female"
    # the next turn tells the storyteller how to speak to the player
    monkeypatch.setattr(main.llm, "chat_stream", fake := FakeLLM([she]))
    with client.stream("POST", "/turn/stream", json={"session_id": sid, "text": "יש פה פרפר"}) as r:
        list(r.iter_lines())
    assert "to the player in the feminine" in fake.calls[0][-1]["content"]
