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

    async def __call__(self, messages, schema, max_tokens=420, temperature=None, model=None):
        """Streams the next answer in small pieces, like Ollama does."""
        self.calls.append(messages)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        text = json.dumps(item, ensure_ascii=False)
        for i in range(0, len(text), 7):
            yield text[i:i + 7]


@pytest.fixture
def client(monkeypatch):
    async def no_prepare(*args, **kwargs):
        return None
    monkeypatch.setattr(main.llm, "prepare", no_prepare)
    monkeypatch.setattr(main.llm, "watch", no_prepare)
    main.llm.state = "ready"
    with TestClient(main.app) as c:
        yield c


def test_examples_match_schema():
    for pictures in (True, False):
        schema = turn_schema(pictures)
        for _, out in prompt.examples(pictures):
            jsonschema.validate(json.loads(out), schema)
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
    monkeypatch.setattr(main.llm, "chat_stream", fake)
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

    # the prompt carries the state and keeps a cacheable prefix: the same system prompt and
    # examples every turn, in an order that only changes with the language
    first, second = fake.calls
    n = 1 + 2 * len(prompt.EXAMPLES)
    assert first[0] == second[0]
    assert sorted(m["content"] for m in first[1:n]) == sorted(m["content"] for m in second[1:n])
    assert (prompt.build_messages("a", [], "x", "en")[:n]
            == prompt.build_messages("b", [{"role": "user", "text": "z"}], "y", "en")[:n])
    assert "tall-trees" in second[-1]["content"]

    # resuming returns the same dream
    r = client.post("/session", json={"session_id": sid, "lang": "he"}).json()
    assert r["resumed"] is True and r["scene"]["setting"]["kind"] == "moon"
    again = SessionStore(os.environ["DATA_DIR"]).get(sid)
    assert again and again.setting["kind"] == "moon" and again.turn == 2


def test_turn_waits_then_asks_to_come_back_when_model_not_ready(client, monkeypatch):
    monkeypatch.setattr(main, "READY_WAIT_S", 0.2)
    main.llm.state, main.llm.detail = "pulling", "downloading gemma3:4b: 40%"
    sid = "test-session-2"
    client.post("/session", json={"session_id": sid, "lang": "en"})
    r = client.post("/turn", json={"session_id": sid, "text": "hello there"})
    assert r.status_code == 503 and r.json()["detail"]["state"] == "pulling"
    main.llm.state = "ready"
    sess = main.store.get(sid)
    assert [h["role"] for h in sess.history] == ["assistant"]     # only the opening line


def test_failed_turn_retries_and_is_never_recorded(client, monkeypatch):
    sid = "test-session-4"
    client.post("/session", json={"session_id": sid, "lang": "en"})
    # first try fails, the retry succeeds
    monkeypatch.setattr(main.llm, "chat_stream", FakeLLM([ValueError("bad json"), FAIRY]))
    r = client.post("/turn", json={"session_id": sid, "text": "you are a fairy"}).json()
    assert r["fallback"] is False and r["reply"] == FAIRY["reply"]
    # both tries fail: a fallback line comes back, and the story does not change
    monkeypatch.setattr(main.llm, "chat_stream", FakeLLM([{"reply": ""}, ValueError("still bad")]))
    before = list(main.store.get(sid).history)
    r = client.post("/turn", json={"session_id": sid, "text": "fly to the moon"}).json()
    assert r["fallback"] is True and r["reply"] == prompt.FALLBACK["en"] and r["error"]
    assert main.store.get(sid).history == before


def test_old_fallback_exchanges_are_forgotten(client):
    sid = "test-session-5"
    store = SessionStore(os.environ["DATA_DIR"])
    sess = store.create(sid, "he")
    sess.remember("assistant", prompt.opening_line("he"))
    for _ in range(3):
        sess.remember("user", "את פיה")
        sess.remember("assistant", prompt.FALLBACK["he"])
    store.save(sess)
    main.store._mem.pop(sid, None)
    r = client.post("/session", json={"session_id": sid, "lang": "he"}).json()
    assert r["resumed"] is True and r["reply"] not in prompt.FALLBACK.values()
    assert [h["role"] for h in main.store.get(sid).history] == ["assistant"]


def test_dream_forgets_oldest_things(client, monkeypatch):
    many = []
    for i in range(6):
        many.append({**FAIRY, "reply": "More stones gather around me.", "add": [{"id": f"rock-{i}-{j}", "kind": "rock", "label": "", "color": "",
                                       "size": "small", "count": 1, "where": "ground",
                                       "motion": "still"} for j in range(4)],
                     "setting": {"kind": "same", "time": "same", "weather": "same"}})
    monkeypatch.setattr(main.llm, "chat_stream", FakeLLM(many))
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


def test_lexicon_reads_hebrew_and_english():
    from app.lexicon import read
    h = read("את פיה ואת נמצאת ביער חשוך")
    assert (h.form, h.setting, h.voice) == ("fairy", "forest", "female")
    h = read("את בונה ספינת חלל מעלים וטסה לירח")
    assert h.setting == "moon" and h.things[0][0] == "spaceship" and h.form is None
    assert read("את אוכלת את הגבינה").form is None          # "את ה..." is the object marker
    assert read("יש פה מים").things[0][0] == "water"         # water, not "ים" (sea)
    assert read("ספר לי מה את רואה").things == []            # "tell me", not a book
    h = read("You are a tiny fairy in a dark forest")
    assert (h.form, h.setting) == ("fairy", "forest")
    assert read("you are holding a sword").form is None
    assert read("You become the god of the whole world").effect == "ascend"


def test_scene_follows_the_player_when_the_model_changes_nothing(client, monkeypatch):
    lazy = {"facts": [], "reply": "אני מרגישה את האוויר הסמיך... מה עוד?", "voice": "same",
            "self": {"form": "same", "label": "", "color": ""},
            "setting": {"kind": "same", "time": "same", "weather": "same"},
            "add": [], "change": [], "remove": [], "effect": "none", "clarity": 0.2,
            "backdrop": "", "ending": False}
    monkeypatch.setattr(main.llm, "chat_stream", FakeLLM([lazy, dict(lazy)]))
    sid = "test-session-6"
    client.post("/session", json={"session_id": sid, "lang": "he"})
    r = client.post("/turn", json={"session_id": sid, "text": "את פיה ואת נמצאת ביער חשוך"}).json()
    assert r["scene"]["self"]["form"] == "fairy" and r["scene"]["setting"]["kind"] == "forest"
    assert r["voice"] == "female" and r["effect"] in ("warp", "sparkle")
    r = client.post("/turn", json={"session_id": sid, "text": "את בונה ספינת חלל וטסה לירח"}).json()
    assert r["scene"]["setting"]["kind"] == "moon"
    assert any(t["kind"] == "spaceship" for t in r["scene"]["things"])
    recent = client.get("/debug/recent").json()
    assert recent[0]["nudged"] and recent[0]["said"].startswith("את בונה")


def test_model_kinds_are_refined_and_not_doubled():
    from app.lexicon import nudge
    from app.schema import ThingSpec, TurnOutput
    out = TurnOutput(reply="...", add=[ThingSpec(id="giant-cheese", kind="food",
                                                 label="a huge block of white cheese")])
    nudge(out, "על הירח את מוצאת גבינה ענקית", "moon", "fairy", "female", set(), [])
    assert [t.kind for t in out.add] == ["cheese"]


def test_the_dream_only_moves_when_the_player_travels():
    from app.lexicon import nudge
    from app.schema import ThingSpec, TurnOutput
    out = TurnOutput(reply="...", setting_kind="city",
                     add=[ThingSpec(id="castle", kind="castle", label="a castle with a crown")])
    nudge(out, "A castle appears with a golden crown inside", "clouds", "dragon", "male", set(), [])
    assert out.setting_kind == "same" and out.add[0].kind == "castle"
    out = TurnOutput(reply="...", setting_kind="clouds")
    nudge(out, "You wake up and fly above the clouds at dawn", "cave", "dragon", "male", set(), [])
    assert out.setting_kind == "clouds"
    out = TurnOutput(reply="...", setting_kind="moon")
    nudge(out, "את טסה לכוכב רחוק", "forest", "fairy", "female", set(), [])
    assert out.setting_kind == "moon"
    out = TurnOutput(reply="...", setting_kind="cave",
                     add=[ThingSpec(id="dark-cave", kind="other", label="a small dark cave")])
    nudge(out, "You are a tiny dragon sleeping in a dark cave", "void", "wisp", "neutral", set(), [])
    assert out.setting_kind == "cave" and out.add == []


def test_watchdog_reloads_a_model_ollama_dropped(monkeypatch):
    import asyncio
    import httpx
    from app.llm import Ollama
    loaded = {"models": []}
    calls = []

    def backend(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/api/ps":
            return httpx.Response(200, json=loaded)
        return httpx.Response(200, json={})

    llm = Ollama("http://ollama:11434", "gemma3:4b", keep_alive="-1")
    assert llm.keep_alive == -1
    llm._client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    prepared = []

    async def fake_prepare(warm=None, schema=None):
        prepared.append(1)
        loaded["models"] = [{"name": "gemma3:4b", "model": "gemma3:4b"}]

    llm.prepare = fake_prepare

    async def run():
        task = asyncio.create_task(llm.watch(None, every=0.01))
        await asyncio.sleep(0.05)
        loaded["models"] = []                      # Ollama restarted: the model is gone
        await asyncio.sleep(0.05)
        task.cancel()

    asyncio.run(run())
    assert len(prepared) >= 2 and "/api/ps" in calls


def test_the_dreamer_is_never_also_a_thing():
    from app.lexicon import nudge
    from app.schema import ThingSpec, TurnOutput
    out = TurnOutput(reply="...", add=[ThingSpec(id="fox", kind="dog", label="a small, fluffy fox"),
                                       ThingSpec(id="moon", kind="moon", label="a pale moon")])
    nudge(out, "You are a little fox in a snowy forest at night", "void", "wisp", "neutral", set(), [])
    assert out.self_form == "wolf" and [t.id for t in out.add] == ["moon"]


def test_now_line_reads_naturally():
    from app.story import Session
    s = Session(id="abcdef")
    assert s.now_line() == "You are a nameless spark of light in a dark mist, at night."
    s.self_form, s.setting["kind"] = "fairy", "moon"
    assert s.now_line() == "You are a fairy on the moon, at night."


def test_reply_streams_before_the_scene(client, monkeypatch):
    monkeypatch.setattr(main.llm, "chat_stream", FakeLLM([FAIRY]))
    sid = "test-session-7"
    client.post("/session", json={"session_id": sid, "lang": "he"})
    with client.stream("POST", "/turn/stream", json={"session_id": sid, "text": "את פיה ביער חשוך"}) as r:
        events = [json.loads(line) for line in r.iter_lines() if line.strip()]
    assert [e["type"] for e in events] == ["first", "reply", "final"]
    assert events[0]["text"] == "פיה... כן. אני מרגישה את הכנפיים שלי רועדות."
    assert events[0]["voice"] == "female" and FAIRY["reply"].startswith(events[0]["text"])
    assert events[1]["reply"] == FAIRY["reply"] and events[1]["voice"] == "female"
    assert events[2]["scene"]["self"]["form"] == "fairy" and events[2]["reply"] == FAIRY["reply"]


def test_spoken_words_survive_a_broken_scene(client, monkeypatch):
    class Broken:
        async def __call__(self, *a, **k):
            yield '{"facts": [], "voice": "same", "reply": "A dragon... I feel my wings.", "self": {"form": "dra'
            raise ValueError("connection dropped")

    monkeypatch.setattr(main.llm, "chat_stream", Broken())
    sid = "test-session-8"
    client.post("/session", json={"session_id": sid, "lang": "en"})
    with client.stream("POST", "/turn/stream", json={"session_id": sid, "text": "You are a dragon"}) as r:
        events = [json.loads(line) for line in r.iter_lines() if line.strip()]
    assert events[0]["reply"] == "A dragon... I feel my wings."
    final = events[-1]
    assert final["fallback"] is False and final["scene"]["self"]["form"] == "dragon"


def test_kinds_follow_their_own_labels():
    from app.lexicon import refine_kinds
    from app.schema import ThingSpec, TurnOutput
    out = TurnOutput(reply="...", add=[
        ThingSpec(id="k", kind="gem", label="a small golden key"),
        ThingSpec(id="g", kind="person", label="a young girl holding a glowing flashlight"),
        ThingSpec(id="f", kind="dog", label="a small, fluffy fox with white fur"),
        ThingSpec(id="m", kind="rock", label="a jagged mountain peak shrouded in mist"),
        ThingSpec(id="p", kind="portal", label="a shimmering portal to space")])
    refine_kinds(out)
    assert [t.kind for t in out.add] == ["key", "person", "wolf", "rock", "portal"]


def test_things_that_only_repeat_the_place_are_dropped():
    from app.lexicon import nudge
    from app.schema import ThingSpec, TurnOutput
    out = TurnOutput(reply="...", setting_kind="cave", add=[
        ThingSpec(id="mouth", kind="house", label="the mouth of a dark cave"),
        ThingSpec(id="girl", kind="person", label="a girl with a lantern")])
    nudge(out, "Then you fly back home to your cave", "space", "dragon", "male", set(), [])
    assert [t.id for t in out.add] == ["girl"]
    out = TurnOutput(reply="...", add=[ThingSpec(id="peak", kind="rock", label="a tall snow-capped mountain")])
    nudge(out, "You are a dragon sleeping on a mountain", "mountains", "dragon", "male", set(), [])
    assert out.add == []


def test_trial_turn_uses_the_chosen_model_and_holds_the_watchdog(client, monkeypatch):
    seen = {}

    class Fake(FakeLLM):
        async def __call__(self, messages, schema, max_tokens=420, temperature=None, model=None):
            seen["model"] = model
            async for piece in FakeLLM.__call__(self, messages, schema, max_tokens, temperature, model):
                yield piece

    monkeypatch.setattr(main.llm, "chat_stream", Fake([FAIRY]))
    sid = "test-trial-1"
    client.post("/session", json={"session_id": sid, "lang": "he"})
    r = client.post("/debug/turn", json={"session_id": sid, "text": "את פיה", "model": "gemma4:e4b"}).json()
    assert seen["model"] == "gemma4:e4b" and r["model"] == "gemma4:e4b"
    assert r["first_words_s"] is not None and r["scene"]["self"]["form"] == "fairy"
    import time as _t
    assert main.llm.hold_until > _t.time()
    main.llm.hold_until = 0


def test_prompt_stays_small_for_long_stories():
    from app.story import Session
    sess = Session(id="long-one")
    for i in range(40):
        sess.remember("user", "מילה " * 400)
        sess.remember("assistant", "word " * 400)
    msgs = prompt.build_messages(sess.summary(), sess.history, "x " * 3000, "he", 8, sess.now_line())
    last = msgs[-1]["content"]
    assert len(last) < 4000 and last.count("…") >= 8


def test_a_fox_stays_a_fox():
    from app.lexicon import nudge
    from app.schema import TurnOutput
    out = TurnOutput(reply="...")
    nudge(out, "You are a little fox in a snowy forest at night", "void", "wisp", "neutral", set(), [])
    assert out.self_form == "wolf" and out.self_label == "fox" and "I am a fox" in out.facts
    out = TurnOutput(reply="...", self_form="wolf", self_label="grey wolf with snowy fur")
    nudge(out, "אתה שועל קטן ביער מושלג", "void", "wisp", "neutral", set(), [])
    assert out.self_label.startswith("fox")


def test_examples_in_the_answer_language_come_last():
    for lang in ("he", "en"):
        exs = prompt.examples(True, lang)
        assert len(exs) == len(prompt.EXAMPLES)
        assert prompt.detect_lang(json.loads(exs[-1][1])["reply"], "x") == lang
        only = prompt.examples(False, lang, only_lang=True)
        assert only and all(prompt.detect_lang(json.loads(a)["reply"], "x") == lang for _, a in only)
    msgs = prompt.build_messages("Nothing is known yet.", [], "You are a dragon", "en", strict_lang=True)
    assert msgs[-1]["content"].endswith(prompt.STRICT_LANG["en"])


def test_an_answer_in_the_wrong_language_is_asked_again(client, monkeypatch):
    dragon_he = {**FAIRY, "reply": "דרקון... אני מרגיש את הכנפיים שלי.", "voice": "same"}
    dragon_en = {**FAIRY, "reply": "A dragon... my wings stir on the cold stone.", "voice": "same"}
    fake = FakeLLM([dragon_he, dragon_en])
    monkeypatch.setattr(main.llm, "chat_stream", fake)
    sid = "test-session-lang"
    client.post("/session", json={"session_id": sid, "lang": "he"})
    before = main.stats["wrong_language"]
    with client.stream("POST", "/turn/stream",
                       json={"session_id": sid, "text": "You are a small red dragon"}) as r:
        events = [json.loads(line) for line in r.iter_lines() if line.strip()]
    # a single sentence comes out whole, once its closing quote arrives
    assert [e["type"] for e in events] == ["reply", "final"]
    assert events[0]["reply"] == dragon_en["reply"] and events[0]["lang"] == "en"
    assert events[1]["reply"] == dragon_en["reply"]
    assert main.stats["wrong_language"] == before + 1
    assert len(fake.calls) == 2 and fake.calls[1][-1]["content"].endswith(prompt.STRICT_LANG["en"])


def test_others_of_the_dreamers_kind_still_appear():
    from app.lexicon import nudge
    from app.schema import TurnOutput
    out = TurnOutput(reply="...")
    nudge(out, "It starts to snow and three wolves appear", "forest", "wolf", "neutral", set(), [],
          current_label="little fox")
    assert [(t.kind, t.count) for t in out.add] == [("wolf", 3)]
    out = TurnOutput(reply="...")
    nudge(out, "The fox runs to the frozen lake", "forest", "wolf", "neutral", set(), [],
          current_label="little fox")
    assert "wolf" not in [t.kind for t in out.add]
    out = TurnOutput(reply="...")
    nudge(out, "A grey wolf watches you", "forest", "wolf", "neutral", set(), [],
          current_label="little fox")
    assert [t.kind for t in out.add] == ["wolf"]


def test_hebrew_answers_are_asked_to_agree_with_the_dreamers_gender(client, monkeypatch):
    fake = FakeLLM([FAIRY])
    monkeypatch.setattr(main.llm, "chat_stream", fake)
    sid = "test-session-gender"
    client.post("/session", json={"session_id": sid, "lang": "he"})
    client.post("/turn", json={"session_id": sid, "text": "את דג קטן בים עמוק"})
    assert fake.calls[0][-1]["content"].endswith(
        "answer in Hebrew only, with no English words" + prompt.HE_GENDER["female"]
        + ", then put all of it into the scene fields.")
    english = prompt.build_messages("x", [], "You are a fish", "en", gender="female")
    assert "feminine" not in english[-1]["content"]


def test_first_sentence_is_spoken_even_when_the_rest_is_lost(client, monkeypatch):
    class Cut:
        def __init__(self):
            self.calls = 0

        async def __call__(self, *a, **k):
            self.calls += 1
            if self.calls == 1:
                yield ('{"facts": [], "voice": "same", "reply": "A long, cold wind runs across '
                       'the stone. Then the')
                raise ValueError("connection dropped")
            yield json.dumps({**MOON, "reply": "Something else entirely.", "add": []})

    monkeypatch.setattr(main.llm, "chat_stream", Cut())
    sid = "test-session-cut"
    client.post("/session", json={"session_id": sid, "lang": "en"})
    with client.stream("POST", "/turn/stream", json={"session_id": sid, "text": "You fly to the moon"}) as r:
        events = [json.loads(line) for line in r.iter_lines() if line.strip()]
    assert [e["type"] for e in events] == ["first", "final"]
    assert events[0]["text"] == "A long, cold wind runs across the stone."
    # the second try only shaped the scene: the words stay the ones already spoken
    assert events[1]["reply"] == events[0]["text"] and events[1]["scene"]["setting"]["kind"] == "moon"


def test_first_sentence_waits_for_a_long_enough_one():
    assert main.first_sentence("Snow... ") == ""
    assert main.first_sentence("Snow... it falls and falls on everything. And") == \
        "Snow... it falls and falls on everything."
    assert main.first_sentence("A single sentence that never ends") == ""


def test_the_named_place_wins_when_the_model_repeats_the_old_one():
    from app.lexicon import nudge
    from app.schema import TurnOutput
    out = TurnOutput(reply="...", setting_kind="mountains")
    nudge(out, "את עולה על הדרקון ועפה איתו מעל העננים", "mountains", "person", "female", set(), [])
    assert out.setting_kind == "clouds"
    out = TurnOutput(reply="...", setting_kind="mountains")
    nudge(out, "בעננים אתם מוצאים ארמון עשוי מקרח", "mountains", "person", "female", set(), [])
    assert out.setting_kind == "clouds"
    out = TurnOutput(reply="...")
    nudge(out, "far away you see a mountain", "forest", "fairy", "female", set(), [])
    assert out.setting_kind == "same"
    out = TurnOutput(reply="...")
    nudge(out, "את רואה הר רחוק", "forest", "fairy", "female", set(), [])
    assert out.setting_kind == "same"


def test_facts_are_not_repeated_and_grow_more_precise():
    from app.schema import TurnOutput
    from app.story import Session
    sess = Session(id="facts-1")
    sess.apply(TurnOutput(reply=".", facts=["I am a witch", "I live in a cabin"]))
    sess.apply(TurnOutput(reply=".", facts=["I am a young witch", "I live in a cabin.", "I am a cat"]))
    sess.apply(TurnOutput(reply=".", facts=["I am a catfish", "i am a witch"]))
    assert sess.facts == ["I live in a cabin", "I am a young witch", "I am a cat", "I am a catfish"]


def test_a_companion_of_the_dreamers_kind_stays_in_the_scene():
    from app.lexicon import nudge
    from app.schema import ThingSpec, TurnOutput
    out = TurnOutput(reply="...", self_form="person", add=[ThingSpec(id="snow-queen", kind="person",
                                                                    label="ice queen")])
    nudge(out, "בתוך הארמון יושבת מלכת השלג ומחכה לך", "mountains", "person", "female", set(), [],
          current_label="witch")
    assert "snow-queen" in [t.id for t in out.add]


def test_a_dreamer_filed_as_other_takes_the_shape_it_names():
    from app.lexicon import nudge
    from app.schema import TurnOutput
    out = TurnOutput(reply="...", self_form="other", self_label="old wise owl")
    nudge(out, "אתה ינשוף זקן וחכם שגר בעץ ענק ביער", "void", "wisp", "neutral", set(), [])
    assert out.self_form == "bird"
    out = TurnOutput(reply="...", self_form="other", self_label="a shimmering being")
    nudge(out, "you are a robot", "void", "wisp", "neutral", set(), [])
    assert out.self_form == "person"


def test_weather_stays_behind_when_the_dream_leaves_for_the_moon():
    from app.schema import TurnOutput
    from app.story import Session
    sess = Session(id="weather-1")
    sess.apply(TurnOutput(reply=".", setting_kind="forest", setting_weather="snow"))
    sess.apply(TurnOutput(reply=".", setting_kind="moon"))
    assert sess.setting["weather"] == "clear"
    sess.apply(TurnOutput(reply=".", setting_kind="meadow", setting_weather="rain"))
    sess.apply(TurnOutput(reply=".", setting_kind="mountains"))
    assert sess.setting["weather"] == "rain"


def test_waking_up_in_bed_ends_the_dream():
    from app.lexicon import nudge
    from app.schema import TurnOutput
    for text in ("אתה מתעורר במיטה שלך", "You wake up in your bed", "החלום נגמר"):
        out = TurnOutput(reply="...")
        nudge(out, text, "forest", "fairy", "female", set(), [])
        assert out.ending, text
    out = TurnOutput(reply="...")
    nudge(out, "You wake up and fly to a city made of crystal", "mountains", "dragon", "neutral", set(), [])
    assert not out.ending


def test_things_the_model_drew_under_another_kind_are_not_added_again():
    from app.lexicon import nudge
    from app.schema import ThingSpec, TurnOutput
    out = TurnOutput(reply="...", add=[ThingSpec(id="tree", kind="tree", label="giant tree with glowing fruit")])
    nudge(out, "באי יש עץ ענק עם פירות זוהרים", "ocean", "fish", "female", set(), [])
    assert [t.kind for t in out.add] == ["tree"]
