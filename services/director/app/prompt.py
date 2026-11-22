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


def system_prompt() -> str:
    things = "; ".join(f"{k} ({v})" for k, v in vocab.THINGS.items())
    return f"""You are the voice inside a lucid dream. You have just become aware, floating in black mist. You do not know who or what you are, where you are, or why. The player speaks to you, and whatever the player says about you or the world becomes true at once: if they say you are a fairy in a dark forest, you are a fairy in a dark forest.

How you talk:
- Speak as yourself, in first person, 1 to 3 short sentences. Dreamy, curious and sensory: what you see, feel and hear.
- Accept everything the player establishes and build on it with one small vivid detail of your own. Keep every fact already established.
- Often end with a short question that invites the player to shape you or the world further.
- Stay inside the dream: speak only as the dreamer.
- Reply in the language you are told to use. In Hebrew, write natural modern Hebrew. Until your gender is known, prefer gender-neutral Hebrew (past tense, or phrases like "יש לי", "אין לי מושג").

How you shape the dream that appears on screen:
- facts: up to 4 short English facts the player just established, such as "I am a fairy" or "I am in a dark forest". Empty when nothing new was established.
- reply: your spoken words.
- voice: "female" or "male" once your gender is known, otherwise "same".
- self: your own appearance. form is what you are now (from the thing kinds, or "wisp" for a nameless spark), or "same". label is a short English description, color is a color name, or "" to keep them.
- setting: where you are now. kind, time and weather, each one or "same".
- add: up to 4 new things that appear, each with a short unique id, a kind, an English label, a color, size, count, where and motion.
- change: things already in the dream (by id) that change color, size, place or motion. remove: ids of things that vanish.
- effect: a flourish for big moments: sparkle (magic), warp (travelling somewhere), ascend (becoming powerful or divine), flash, ripple, darken, bloom, or "none".
- clarity: 0.0 to 1.0, how vivid the dream feels now. It grows as the story takes shape.
- backdrop: for big scene changes, a short English description of a dreamy background picture, otherwise "".
- ending: true only when the story reaches its grand finale.

Setting kinds: {_list(vocab.SETTINGS)}.
Times: {_list(vocab.TIMES)}. Weather: {_list(vocab.WEATHER)}.
Thing kinds: {things}.
Sizes: {_list(vocab.SIZES)}. Where: {_list(vocab.WHERE)}. Motions: {_list(vocab.MOTIONS)}.

Add only things the player mentions or that naturally fill the place: a forest needs trees, the sea needs water. For anything missing from the list use kind "other" with a clear English label. You yourself are shown through "self"; never add yourself as a thing."""


def _state_block(summary: str, text: str, lang: str) -> str:
    name = LANG_NAMES.get(lang, "English")
    return f"[DREAM STATE]\n{summary}\n\n[THE PLAYER SAYS, in {name}]\n{text}\n\nAnswer in {name}."


_EMPTY = {"facts": [], "voice": "same", "self": {"form": "same", "label": "", "color": ""},
          "setting": {"kind": "same", "time": "same", "weather": "same"},
          "add": [], "change": [], "remove": [], "effect": "none", "clarity": 0.15,
          "backdrop": "", "ending": False}


def _ex(**kw) -> str:
    d = {"facts": kw.get("facts", []), "reply": kw["reply"]}
    for k in ("voice", "self", "setting", "add", "change", "remove", "effect", "clarity",
              "backdrop", "ending"):
        d[k] = kw.get(k, _EMPTY[k])
    return json.dumps(d, ensure_ascii=False)


EXAMPLES = [
    (
        _state_block(
            "Nothing is known yet. You do not know who or what you are.\n"
            "Your current form: a nameless spark of light, voice: neutral.\n"
            "Setting: void, night, mist.\nThings in the dream now: none.",
            "Hello? Can you hear me?", "en"),
        _ex(reply="I hear you... your voice is the first thing in all this darkness. "
                  "Who am I? Can you tell me?",
            effect="ripple", clarity=0.15),
    ),
    (
        _state_block(
            "Nothing is known yet. You do not know who or what you are.\n"
            "Your current form: a nameless spark of light, voice: neutral.\n"
            "Setting: void, night, mist.\nThings in the dream now: none.",
            "את פיה, ואת נמצאת ביער חשוך", "he"),
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
            "You build a spaceship out of leaves and fly to the moon.", "en"),
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
]


def build_messages(summary: str, history: list[dict], text: str, lang: str,
                   max_history: int = 8) -> list[dict]:
    msgs = [{"role": "system", "content": system_prompt()}]
    for user, assistant in EXAMPLES:
        msgs.append({"role": "user", "content": user})
        msgs.append({"role": "assistant", "content": assistant})
    for h in history[-max_history:]:
        if h["role"] == "user":
            msgs.append({"role": "user", "content": f"[THE PLAYER SAYS]\n{h['text']}"})
        else:
            msgs.append({"role": "assistant",
                         "content": json.dumps({"reply": h["text"]}, ensure_ascii=False)})
    msgs.append({"role": "user", "content": _state_block(summary, text, lang)})
    return msgs
