"""JSON schema the LLM must follow (Ollama structured outputs) and a forgiving parser.

Every key is required and uses "same" / "" / [] for "no change", which keeps the
grammar simple enough for small local models. The parser still assumes nothing:
it clamps, defaults and drops anything malformed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import vocab

_ID_RE = re.compile(r"[^a-z0-9_-]+")


def _enum(values, extra=()):
    return {"type": "string", "enum": list(extra) + list(values)}


def turn_schema(pictures: bool = True) -> dict:
    thing = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "kind": _enum(vocab.THING_KINDS),
            "label": {"type": "string"},
            "color": {"type": "string"},
            "size": _enum(vocab.SIZES.keys()),
            "count": {"type": "integer", "minimum": 1, "maximum": 40},
            "where": _enum(vocab.WHERE),
            "motion": _enum(vocab.MOTIONS),
        },
        "required": ["id", "kind", "label", "color", "size", "count", "where", "motion"],
    }
    change = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "color": {"type": "string"},
            "size": _enum(vocab.SIZES.keys(), extra=["same"]),
            "where": _enum(vocab.WHERE, extra=["same"]),
            "motion": _enum(vocab.MOTIONS, extra=["same"]),
        },
        "required": ["id", "color", "size", "where", "motion"],
    }
    props = {
            "facts": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
            "reply": {"type": "string"},
            "voice": _enum(vocab.VOICES, extra=["same"]),
            "self": {
                "type": "object",
                "properties": {
                    "form": _enum(vocab.SELF_FORMS, extra=["same"]),
                    "label": {"type": "string"},
                    "color": {"type": "string"},
                },
                "required": ["form", "label", "color"],
            },
            "setting": {
                "type": "object",
                "properties": {
                    "kind": _enum(vocab.SETTINGS, extra=["same"]),
                    "time": _enum(vocab.TIMES, extra=["same"]),
                    "weather": _enum(vocab.WEATHER, extra=["same"]),
                },
                "required": ["kind", "time", "weather"],
            },
            "keep": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
            "add": {"type": "array", "items": thing, "maxItems": 4},
            "change": {"type": "array", "items": change, "maxItems": 4},
            "remove": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
            "effect": _enum(vocab.EFFECTS),
            "clarity": {"type": "number"},
            "backdrop": {"type": "string"},
            "ending": {"type": "boolean"},
    }
    # Facts and voice, then the spoken reply, then the scene (see prompt.ORDER).
    order = ["facts", "voice", "reply", "self", "setting", "keep", "add", "change", "remove",
             "effect", "clarity", "backdrop", "ending"]
    if not pictures:
        order.remove("backdrop")
    return {"type": "object", "properties": {k: props[k] for k in order}, "required": order}


def direction_schema() -> dict:
    """What the dream's quiet director writes between turns (see prompt.direction_message)."""
    beat = {"type": "object",
            "properties": {"who": {"type": "string"}, "does": _enum(vocab.ACTIONS), "to": {"type": "string"}},
            "required": ["who", "does", "to"]}
    want = {"type": "object", "properties": {"id": {"type": "string"}, "want": {"type": "string"}},
            "required": ["id", "want"]}
    return {"type": "object",
            "properties": {"quest": {"type": "string"}, "shown": {"type": "string"},
                           "progress": {"type": "integer", "minimum": 0, "maximum": vocab.QUEST_STEPS},
                           "wants": {"type": "array", "items": want, "maxItems": 3},
                           "next": {"type": "array", "items": beat, "minItems": 1, "maxItems": 4},
                           "hint": {"type": "string"}},
            "required": ["quest", "shown", "progress", "wants", "next", "hint"]}


# --------------------------------------------------------------------------- parsing

@dataclass
class ThingSpec:
    id: str
    kind: str
    label: str = ""
    color: str = ""
    size: str = "medium"
    count: int = 1
    where: str = "around"
    motion: str = "still"


@dataclass
class ChangeSpec:
    id: str
    color: str = ""
    size: str = "same"
    where: str = "same"
    motion: str = "same"


@dataclass
class TurnOutput:
    reply: str = ""
    facts: list[str] = field(default_factory=list)
    voice: str = "same"
    self_form: str = "same"
    self_label: str = ""
    self_color: str = ""
    setting_kind: str = "same"
    setting_time: str = "same"
    setting_weather: str = "same"
    keep: list[str] = field(default_factory=list)
    add: list[ThingSpec] = field(default_factory=list)
    change: list[ChangeSpec] = field(default_factory=list)
    remove: list[str] = field(default_factory=list)
    effect: str = "none"
    clarity: float | None = None
    backdrop: str = ""
    ending: bool = False


def _str(v, limit=200) -> str:
    if not isinstance(v, str):
        return ""
    return " ".join(v.split())[:limit]


def _pick(v, allowed, default):
    v = _str(v, 40).lower()
    return v if v in allowed else default


def clean_id(v) -> str:
    s = _ID_RE.sub("-", _str(v, 40).lower()).strip("-")
    return s[:32]


def parse_turn(raw: dict) -> TurnOutput:
    """Turn whatever the model produced into a safe TurnOutput."""
    if not isinstance(raw, dict):
        raw = {}
    out = TurnOutput()
    out.reply = _str(raw.get("reply"), 600)
    facts = raw.get("facts") if isinstance(raw.get("facts"), list) else []
    out.facts = [f for f in (_str(x, 160) for x in facts) if f][:3]
    out.voice = _pick(raw.get("voice"), vocab.VOICES + ["same"], "same")

    me = raw.get("self") if isinstance(raw.get("self"), dict) else {}
    out.self_form = _pick(me.get("form"), vocab.SELF_FORMS + ["same"], "same")
    out.self_label = _str(me.get("label"), 80)
    out.self_color = _str(me.get("color"), 30)

    st = raw.get("setting") if isinstance(raw.get("setting"), dict) else {}
    out.setting_kind = _pick(st.get("kind"), vocab.SETTINGS + ["same"], "same")
    out.setting_time = _pick(st.get("time"), vocab.TIMES + ["same"], "same")
    out.setting_weather = _pick(st.get("weather"), vocab.WEATHER + ["same"], "same")

    keep = raw.get("keep") if isinstance(raw.get("keep"), list) else []
    out.keep = [k for k in (clean_id(x) for x in keep) if k][:8]

    for item in (raw.get("add") if isinstance(raw.get("add"), list) else [])[:4]:
        if not isinstance(item, dict):
            continue
        kind = _pick(item.get("kind"), vocab.THING_KINDS, "other")
        tid = clean_id(item.get("id")) or kind
        try:
            count = int(item.get("count", 1))
        except (TypeError, ValueError):
            count = 1
        cap = vocab.MAX_COUNT.get(kind, vocab.DEFAULT_MAX_COUNT)
        out.add.append(ThingSpec(
            id=tid,
            kind=kind,
            label=_str(item.get("label"), 80),
            color=_str(item.get("color"), 30),
            size=_pick(item.get("size"), list(vocab.SIZES), "medium"),
            count=max(1, min(cap, count)),
            where=_pick(item.get("where"), vocab.WHERE, vocab.DEFAULT_WHERE.get(kind, "around")),
            motion=_pick(item.get("motion"), vocab.MOTIONS, vocab.DEFAULT_MOTION.get(kind, "still")),
        ))

    for item in (raw.get("change") if isinstance(raw.get("change"), list) else [])[:4]:
        if not isinstance(item, dict) or not clean_id(item.get("id")):
            continue
        out.change.append(ChangeSpec(
            id=clean_id(item.get("id")),
            color=_str(item.get("color"), 30),
            size=_pick(item.get("size"), list(vocab.SIZES) + ["same"], "same"),
            where=_pick(item.get("where"), vocab.WHERE + ["same"], "same"),
            motion=_pick(item.get("motion"), vocab.MOTIONS + ["same"], "same"),
        ))

    rem = raw.get("remove") if isinstance(raw.get("remove"), list) else []
    out.remove = [r for r in (clean_id(x) for x in rem) if r][:6]
    out.effect = _pick(raw.get("effect"), vocab.EFFECTS, "none")
    try:
        c = float(raw.get("clarity"))
        out.clarity = max(0.0, min(1.0, c))
    except (TypeError, ValueError):
        out.clarity = None
    out.backdrop = _str(raw.get("backdrop"), 200)
    out.ending = bool(raw.get("ending")) if isinstance(raw.get("ending"), bool) else False
    return out


@dataclass
class Direction:
    quest: str = "same"
    shown: str = ""
    progress: int | None = None
    wants: dict[str, str] = field(default_factory=dict)
    next: list[dict] = field(default_factory=list)
    hint: str = ""


def parse_direction(raw: dict, ids: set[str]) -> Direction:
    """The director's answer, kept to the beings and things that are really there."""
    if not isinstance(raw, dict):
        raw = {}
    d = Direction()
    q = _str(raw.get("quest"), 120)
    d.quest = "same" if q.lower() in ("same", "") and raw.get("quest") != "" else q
    d.shown = _str(raw.get("shown"), 80)
    try:
        d.progress = max(0, min(vocab.QUEST_STEPS, int(raw.get("progress"))))
    except (TypeError, ValueError):
        d.progress = None
    for w in (raw.get("wants") if isinstance(raw.get("wants"), list) else [])[:3]:
        if isinstance(w, dict) and (tid := clean_id(w.get("id"))) in ids and _str(w.get("want"), 100):
            d.wants[tid] = _str(w.get("want"), 100)
    seen = 0
    for b in (raw.get("next") if isinstance(raw.get("next"), list) else [])[:4]:
        if not isinstance(b, dict):
            continue
        who, to = clean_id(b.get("who")), clean_id(b.get("to"))
        does = _pick(b.get("does"), vocab.ACTIONS, "")
        if not does or who not in ids | {"self"}:
            continue
        if to not in ids | {"self"} or to == who:
            to = ""
        d.next.append({"who": who, "does": does, "to": to})
        seen += 1
    d.hint = _str(raw.get("hint"), 160)
    return d
