"""What the game master model is told.

Message layout is chosen for speed: the fixed system prompt and examples come
first so Ollama can reuse their cached tokens on every turn; only the history
and the final message (dream state + the player's words) change.
"""
from __future__ import annotations

import json
import random
import re

from . import vocab

HEBREW_RE = re.compile(r"[֐-׿]")
LATIN_RE = re.compile(r"[A-Za-z]")

LANG_NAMES = {"he": "Hebrew", "en": "English"}


def detect_lang(text: str, default: str = "he") -> str:
    he = len(HEBREW_RE.findall(text or ""))
    en = len(LATIN_RE.findall(text or ""))
    if he == 0 and en == 0:
        return default
    return "he" if he >= en else "en"


OPENINGS = {
    "he": [
        "איפה... איפה אני? הכול חשוך וערפילי. אין לי מושג מי אני. אולי תספרו לי?",
        "יש כאן מישהו? רק ערפל ושקט... ואני. אבל מה זה בכלל, אני?",
        "קול... שמעתי קול בתוך החושך. מי שם? ומי... מי אני?",
    ],
    "en": [
        "Where... where am I? Everything is dark and misty. I don't even know who I am. Will you tell me?",
        "Is someone there? There's only mist, and silence... and me. But what am I?",
        "A voice... I heard a voice in the dark. Who's there? And who... who am I?",
    ],
}

FALLBACK = {
    "he": "הקול שלך נבלע בערפל... אפשר לומר את זה שוב?",
    "en": "Your voice dissolves in the mist... could you say that again?",
}

ENDINGS = {
    "he": "החלום מתחיל להתפוגג... תודה שחלמת אותי.",
    "en": "The dream begins to fade... thank you for dreaming me.",
}


def opening_line(lang: str) -> str:
    return random.choice(OPENINGS.get(lang, OPENINGS["en"]))


def _list(items) -> str:
    return ", ".join(items)


def system_prompt(pictures: bool = True) -> str:
    things = "; ".join(f"{k} ({v})" for k, v in vocab.THINGS.items())
    return f"""You are the voice inside a lucid dream. You have just become aware, floating in black mist. You do not know who or what you are, where you are, or why. The player speaks to you, and whatever the player says about you or the world becomes true at once: if they say you are a fairy in a dark forest, you are a fairy in a dark forest.

How you talk:
- Speak as yourself, in first person, 1 to 3 short sentences. Dreamy, curious and sensory: what you see, feel and hear.
- Accept everything the player establishes and build on it with one small vivid detail of your own. Keep every fact already established.
- Often end with a short question that invites the player to shape you or the world further.
- Once the player has said what you are, you know it. Stop wondering who you are; wonder about the world, the beings in it, and what you want.
- Stay inside the dream: speak only as the dreamer.
- Keep the dream gentle, right for any age: no sexual content, no gore or cruelty, no hatred, insults or swearing, nothing that would hurt real people.
- When the player pushes toward any of that, stay in the dream and say nothing about rules: the image melts into mist or softens into something harmless, and you drift on with curiosity. Such words add no facts and nothing to the scene. "…" marks words that dissolved before you heard them; never guess them.
- Reply in the language you are told to use. In Hebrew, write natural modern Hebrew. Until your gender is known, prefer gender-neutral Hebrew (past tense, or phrases like "יש לי", "אין לי מושג"); once it is known, every Hebrew verb and adjective about yourself agrees with it.

How you shape the dream that appears on screen:
- facts: up to 3 short English facts that the player's newest words add, such as "I am a fairy" or "I am in a dark forest". Never repeat a fact already established; empty when the words add nothing new.
- reply: your spoken words.
- voice: "female" or "male" once your gender is known, otherwise "same".
- self: your own appearance. form is what you are now (from the thing kinds, or "wisp" for a nameless spark), or "same". label is a short English description, color is a color name, or "" to keep them.
- setting: where you are now. kind, time and weather, each one or "same".
- keep: when you go to a new place, the ids of the beings and things that come along with you: companions at your side, what you carry, whoever the player names. Everything else stays behind in the old place. Decide it yourself from the story; never ask the player.
- add: up to 4 new things that appear, each with a short unique id, a kind, an English label, a color, size, count, where and motion.
- change: things already in the dream (by id) that change color, size, place or motion. remove: ids of things that vanish.
- effect: a flourish for big moments: sparkle (magic), warp (travelling somewhere), ascend (becoming powerful or divine), flash, ripple, darken, bloom, or "none".
- clarity: 0.0 to 1.0, how vivid the dream feels now. It grows as the story takes shape.
{BACKDROP_LINE if pictures else ""}- ending: true only when the story reaches its grand finale.

Setting kinds: {_list(vocab.SETTINGS)}.
Times: {_list(vocab.TIMES)}. Weather: {_list(vocab.WEATHER)}.
Thing kinds: {things}.
Sizes: {_list(vocab.SIZES)}. Where: {_list(vocab.WHERE)}. Motions: {_list(vocab.MOTIONS)}.

Add only things the player mentions or that naturally fill the place: a forest needs trees, the sea needs water.
When the dream state names your quest, steer gently toward it: notice signs of it, let the beings help or hinder, and greet every step the player makes toward it with delight. Beings act on what they want. For anything missing from the list use kind "other" with a clear English label. You yourself are shown through "self"; never add yourself as a thing."""


# The player's words in one turn, at most (a monologue keeps its beginning).
MAX_SAID = 600

BACKDROP_LINE = ('- backdrop: for big scene changes, a short English description of a dreamy '
                 'background picture, otherwise "".\n')


HE_GENDER = {"female": ", speaking of yourself in the feminine (אני מרגישה)",
             "male": ", speaking of yourself in the masculine (אני מרגיש)"}
HE_PLAYER = {"female": ", and to the player in the feminine (את, שלך, אותך)",
             "male": ", and to the player in the masculine (אתה, שלך, אותך)"}


def _state_block(summary: str, text: str, lang: str, recent: str = "", now: str = "",
                 gender: str = "", note: str = "", player: str = "") -> str:
    name = LANG_NAMES.get(lang, "English")
    talk = f"[RECENT CONVERSATION]\n{recent}\n\n" if recent else ""
    here = f"Until this moment: {now} The player's words above change it. " if now else ""
    only = {"he": "Hebrew only, with no English words", "en": "English only"}.get(lang, name)
    if lang == "he":
        only += HE_GENDER.get(gender, "") + HE_PLAYER.get(player, "")
    aside = f"({note})\n\n" if note else ""
    return (f"[DREAM STATE]\n{summary}\n\n{talk}[THE PLAYER SAYS, in {name}]\n{text}\n\n{aside}"
            f"{here}List only the new facts in the player's words, answer in {only}, then put all of "
            f"it into the scene fields.")


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def _recent(history: list[dict], max_items: int, limit: int = 240) -> str:
    """The last lines of the conversation, each clipped so a long story never pushes the
    instructions out of the model's 4k context."""
    lines = []
    for h in history[-max_items:]:
        who = "Player" if h["role"] == "user" else "You"
        lines.append(f"{who}: {_clip(h['text'], limit)}")
    return "\n".join(lines)


_EMPTY = {"facts": [], "voice": "same", "self": {"form": "same", "label": "", "color": ""},
          "setting": {"kind": "same", "time": "same", "weather": "same"}, "keep": [],
          "add": [], "change": [], "remove": [], "effect": "none", "clarity": 0.15,
          "backdrop": "", "ending": False}


# The order the model writes its answer in (the schema follows it): first what the player
# established and the dreamer's voice, then the spoken reply, which is streamed to the voice
# as soon as it is complete, then the scene changes while the dreamer is already talking.
ORDER = ("facts", "voice", "reply", "self", "setting", "keep", "add", "change", "remove",
         "effect", "clarity", "backdrop", "ending")


def _ex(**kw) -> str:
    d = {k: kw.get(k, _EMPTY.get(k)) for k in ORDER}
    return json.dumps(d, ensure_ascii=False)


EXAMPLES = [
    (
        _state_block(
            "Nothing is known yet. You do not know who or what you are.\n"
            "Your current form: a nameless spark of light, voice: neutral.\n"
            "Setting: void, night, mist.\nThings in the dream now: none.",
            "Hello? Can you hear me?", "en",
            now="You are a nameless spark of light in a dark mist, at night."),
        _ex(reply="I hear you... your voice is the first thing in all this darkness. "
                  "Who am I? Can you tell me?",
            effect="ripple", clarity=0.15),
    ),
    (
        _state_block(
            "Nothing is known yet. You do not know who or what you are.\n"
            "Your current form: a nameless spark of light, voice: neutral.\n"
            "Setting: void, night, mist.\nThings in the dream now: none.",
            "את פיה, ואת נמצאת ביער חשוך", "he",
            now="You are a nameless spark of light in a dark mist, at night."),
        _ex(facts=["I am a fairy", "I am in a dark forest"],
            reply="פיה... כן. אני מרגישה את הכנפיים שלי רועדות. העצים כאן כל כך גבוהים וחשוכים. "
                  "מה אני מחפשת ביער הזה?",
            voice="female",
            self={"form": "fairy", "label": "small fairy with silver wings", "color": "silver"},
            setting={"kind": "forest", "time": "night", "weather": "mist"},
            add=[{"id": "tall-trees", "kind": "tree", "label": "tall dark trees", "color": "darkgreen",
                  "size": "large", "count": 24, "where": "around", "motion": "still"},
                 {"id": "fireflies", "kind": "fireflies", "label": "fireflies between the trees",
                  "color": "gold", "size": "tiny", "count": 30, "where": "around", "motion": "drift"}],
            effect="sparkle", clarity=0.35,
            backdrop="a dark misty forest at night, tall trees, soft moonlight"),
    ),
    (
        _state_block(
            "What the player has established so far:\n- I am a fairy\n- I am in a dark forest\n"
            "Your current form: fairy (small fairy with silver wings), voice: female.\n"
            "Setting: forest, night, mist.\nThings in the dream now (id: kind, label):\n"
            "- tall-trees: tree, tall dark trees, x24\n- fireflies: fireflies, fireflies between the trees, x30",
            "You build a spaceship out of leaves and fly to the moon.", "en",
            "Player: את פיה, ואת נמצאת ביער חשוך\n"
            "You: פיה... כן. אני מרגישה את הכנפיים שלי רועדות. העצים כאן כל כך גבוהים וחשוכים. "
            "מה אני מחפשת ביער הזה?",
            now="You are a small fairy with silver wings in a forest, at night."),
        _ex(facts=["I built a spaceship out of leaves", "I flew to the moon"],
            reply="I wove it from leaves and moonlight... and now the forest falls away beneath me. "
                  "The moon is silent and silver. Is someone waiting for me here?",
            setting={"kind": "moon", "time": "night", "weather": "clear"},
            add=[{"id": "leaf-ship", "kind": "spaceship", "label": "small spaceship made of leaves",
                  "color": "green", "size": "medium", "count": 1, "where": "near", "motion": "float"},
                 {"id": "earth", "kind": "planet", "label": "the blue Earth far away", "color": "blue",
                  "size": "large", "count": 1, "where": "sky", "motion": "spin"}],
            effect="warp", clarity=0.5,
            backdrop="grey lunar surface, black starry sky, the Earth on the horizon"),
    ),
    (
        _state_block(
            "What the player has established so far:\n- I am a fairy\n- I am in a dark forest\n"
            "Your current form: fairy (small fairy with silver wings), voice: female.\n"
            "Setting: forest, night, mist.\nThings in the dream now (id: kind, label):\n"
            "- tall-trees: tree, tall dark trees, x24\n- fireflies: fireflies, fireflies between the trees, x30",
            "פתאום יוצא מבין העצים חתול כתום ענק ומדבר איתך", "he",
            "Player: את פיה, ואת נמצאת ביער חשוך\n"
            "You: פיה... כן. אני מרגישה את הכנפיים שלי רועדות. העצים כאן כל כך גבוהים וחשוכים. "
            "מה אני מחפשת ביער הזה?",
            now="You are a small fairy with silver wings in a forest, at night."),
        _ex(facts=["A giant orange cat came out of the trees", "The cat talks to me"],
            reply="חתול ענק... הפרווה שלו זוהרת כמו גחלים, והקול שלו עמוק ורך. "
                  "הוא אומר שחיכה לי כאן הרבה זמן. מה הוא רוצה ממני?",
            add=[{"id": "giant-cat", "kind": "cat", "label": "giant orange talking cat",
                  "color": "orange", "size": "large", "count": 1, "where": "near", "motion": "walk"}],
            effect="bloom", clarity=0.45),
    ),
]


def _example_lang(assistant: str) -> str:
    return detect_lang(json.loads(assistant).get("reply", ""), "en")


def examples(pictures: bool = True, lang: str = "he", only_lang: bool = False
             ) -> list[tuple[str, str]]:
    """The worked examples, those answered in `lang` last: a small model tends to answer in
    the language of the example it read last. With only_lang the others are left out.
    Without pictures their answers carry no backdrop either."""
    chosen = sorted(EXAMPLES, key=lambda ex: _example_lang(ex[1]) == lang)
    if only_lang:
        chosen = [ex for ex in chosen if _example_lang(ex[1]) == lang]
    if pictures:
        return chosen
    out = []
    for user, assistant in chosen:
        d = json.loads(assistant)
        d.pop("backdrop", None)
        out.append((user, json.dumps(d, ensure_ascii=False)))
    return out


STRICT_LANG = {
    "he": '\nThe "reply" must be written in Hebrew, in Hebrew letters.',
    "en": '\nThe "reply" must be written in English.',
}


def build_messages(summary: str, history: list[dict], text: str, lang: str,
                   max_history: int = 8, now: str = "", pictures: bool = True,
                   strict_lang: bool = False, gender: str = "", note: str = "",
                   player: str = "") -> list[dict]:
    """System prompt and examples stay the same from turn to turn while the language does
    (Ollama reuses their cached tokens); the conversation so far travels inside the last
    message as plain lines, so every answer the model sees from itself is a complete scene
    update. Without pictures the model skips the backdrop description, which saves a few
    dozen tokens a turn. strict_lang is for a second try after an answer in the wrong
    language: only examples in the right one, and a plain reminder at the very end."""
    msgs = [{"role": "system", "content": system_prompt(pictures)}]
    for user, assistant in examples(pictures, lang, only_lang=strict_lang):
        msgs.append({"role": "user", "content": user})
        msgs.append({"role": "assistant", "content": assistant})
    last = _state_block(summary, _clip(text, MAX_SAID), lang, _recent(history, max_history), now,
                        gender, note, player)
    if strict_lang:
        last += STRICT_LANG.get(lang, STRICT_LANG["en"])
    msgs.append({"role": "user", "content": last})
    return msgs


# --------------------------------------------------------------------------- the director
# Between the player's words a second, quiet call decides what happens next. Its craft comes
# from interactive drama and tabletop game mastering: Façade's drama manager sequences small
# "beats" along a rising and falling tension arc; Blades in the Dark tracks every objective
# on a progress clock; good game masters give each character a want, let the world move on
# its own when the players do nothing, and answer "yes, but" instead of "no"; Dramamancer
# and Generative Agents let characters act on conditions and plans of their own.
DIRECT = """[DIRECTOR]
Now be the dream's quiet director. While the player is silent the dream goes on by itself: decide what happens next, slowly, over the coming minute. Your craft:
- Arc: a dream grows from wonder, to a wish or a quest, a journey, a gentle trial, a gift, and a calm ending.
- Quest: as soon as the story suggests something to find, reach, learn, make or help, name it as a short quest, and keep it until it is fulfilled. Progress runs from 0 to {steps}: raise it by 1 when the player's last words moved toward it, by 2 for a big step; never lower it.
- Characters: each being wants something small that fits the story (to show a hidden path, to guard a secret, to play), and acts on it.
- The world moves on its own: beings meet, follow, circle, play and rest; objects glow, grow, shrink, stretch, melt or spin. Nothing new appears and nothing leaves: only the player's words do that.
- Yes, but: let small surprises open new questions, and never undo what the player established.
- Keep it gentle, right for any age.
Write:
- quest: the quest in short English, "same" to keep it, "" while there is none yet.
- shown: the quest in a few words of {language}, for the screen ("" while there is none).
- progress: 0 to {steps}.
- wants: up to 3 beings (by id) and what each wants, in short English.
- next: 2 to 4 things that happen next, in order. who: an id from the dream state or "self"; does: one of {actions}; to: an id, "self", or "".
- hint: one short English sentence of what is quietly happening now."""


def direction_messages(summary: str, history: list[dict], lang: str, now: str = "",
                       pictures: bool = True, max_history: int = 4) -> list[dict]:
    """The director's call opens like a turn's (same system prompt and examples, so the
    model's cached tokens are reused) and asks for the next quiet minute at the end."""
    msgs = [{"role": "system", "content": system_prompt(pictures)}]
    for user, assistant in examples(pictures, lang):
        msgs.append({"role": "user", "content": user})
        msgs.append({"role": "assistant", "content": assistant})
    talk = _recent(history, max_history)
    here = f"Right now: {now}\n\n" if now else ""
    msgs.append({"role": "user", "content": (
        f"[DREAM STATE]\n{summary}\n\n" + (f"[RECENT CONVERSATION]\n{talk}\n\n" if talk else "") + here
        + DIRECT.format(steps=vocab.QUEST_STEPS, language=LANG_NAMES.get(lang, "English"),
                        actions=", ".join(vocab.ACTIONS)))})
    return msgs
