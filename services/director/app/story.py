"""Session state: what the dreamer knows about itself and what the dream contains."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import vocab
from .schema import Direction, TurnOutput

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,64}$")

# Places with no weather of their own.
NO_WEATHER = {"space", "moon", "underwater", "cave"}

# Things that belong to a place: when the dream moves somewhere else they stay behind
# unless the storyteller says they come along. Beings and objects come along only when they
# are kept, changed or named (see Session.apply).
SCENERY = {"tree", "flower", "mushroom", "rock", "water", "fireflies", "cloud", "house",
           "castle", "fire", "rainbow", "crystal", "portal", "sun", "moon", "planet", "star",
           "other"}


@dataclass
class Thing:
    id: str
    kind: str
    label: str = ""
    color: str = ""
    size: str = "medium"
    count: int = 1
    where: str = "around"
    motion: str = "still"
    born: int = 0          # turn number it appeared on


_QUEST_COMMON = {"find", "quest", "quiet", "the", "a", "an", "of", "for", "to", "and", "help", "reach",
                 "learn", "make", "lost", "hidden", "secret", "ancient", "magic", "small", "little", "great"}


def _same_quest(new: str, old: str) -> bool:
    """The director often says the same quest in new words ("a hidden stone" becomes "find
    the ancient moss stone"): they share a word that matters."""
    words = lambda q: {w for w in re.findall(r"[a-z]{3,}", q.lower()) if w not in _QUEST_COMMON}
    return bool(old) and bool(words(new) & words(old))


def _fact_words(fact: str) -> frozenset[str]:
    """A fact's words: one fact repeats another when all its words are in it."""
    return frozenset(re.sub(r"[^\w\s]", " ", fact.lower()).split())


@dataclass
class Session:
    id: str
    lang: str = "he"
    turn: int = 0
    facts: list[str] = field(default_factory=list)
    voice: str = "neutral"
    player: str = ""                 # how the player speaks of themselves: female, male or unknown
    self_form: str = "wisp"
    self_label: str = ""
    self_color: str = "#d8ecff"
    setting: dict = field(default_factory=lambda: {"kind": "void", "time": "night", "weather": "mist"})
    things: dict[str, dict] = field(default_factory=dict)
    clarity: float = 0.12
    ended: bool = False
    # The quiet director's view of the story (written between turns, see apply_direction).
    quest: str = ""                  # what the dream is about now, in short English
    quest_shown: str = ""            # the same in the player's language, for the screen
    progress: int = 0                # 0 .. vocab.QUEST_STEPS
    fulfilled: str = ""              # a quest just completed, until the storyteller celebrates it
    quests_done: int = 0
    wants: dict[str, str] = field(default_factory=dict)
    hint: str = ""                   # what is quietly happening, for the next answer
    directed: int = -1               # the turn the director last looked at
    history: list[dict] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    # ------------------------------------------------------------------ views
    def scene(self) -> dict:
        return {
            "self": {"form": self.self_form, "label": self.self_label, "color": self.self_color},
            "setting": dict(self.setting),
            "things": [dict(t) for t in self.things.values()],
            "clarity": round(self.clarity, 3),
            "turn": self.turn,
            "ended": self.ended,
        }

    def summary(self) -> str:
        """Compact state description for the prompt."""
        lines = []
        if self.facts:
            lines.append("What the player has established so far:")
            lines += [f"- {f}" for f in self.facts[-vocab.MAX_FACTS:]]
        else:
            lines.append("Nothing is known yet. You do not know who or what you are.")
        me = self.self_form if self.self_form != "wisp" else "a nameless spark of light"
        lines.append(f"Your current form: {me}" + (f" ({self.self_label})" if self.self_label else "")
                     + f", voice: {self.voice}.")
        s = self.setting
        lines.append(f"Setting: {s['kind']}, {s['time']}, {s['weather']}.")
        if self.things:
            lines.append("Things in the dream now (id: kind, label):")
            for t in self.things.values():
                want = self.wants.get(t["id"])
                lines.append(f"- {t['id']}: {t['kind']}" + (f", {t['label']}" if t["label"] else "")
                             + (f", x{t['count']}" if t["count"] > 1 else "")
                             + (f" (wants {want})" if want else ""))
        else:
            lines.append("Things in the dream now: none.")
        if self.fulfilled:
            lines.append(f"Your quest is fulfilled: {self.fulfilled}. Celebrate it; a new wish may appear.")
        elif self.quest:
            lines.append(f"Your quest: {self.quest} ({self.progress} of {vocab.QUEST_STEPS} done).")
        if self.hint:
            lines.append(f"Meanwhile: {self.hint}")
        return "\n".join(lines)

    def now_line(self) -> str:
        """One plain sentence of where and what the dreamer is right now."""
        me = self.self_form if self.self_form != "wisp" else "nameless spark of light"
        if self.self_label:
            me = self.self_label
        a = "an" if me[:1].lower() in "aeiou" else "a"
        kind = self.setting["kind"]
        where = {"void": "in a dark mist", "moon": "on the moon", "space": "in space",
                 "clouds": "above the clouds", "underwater": "under the water",
                 "snow": "in a snowy land", "mountains": "in the mountains",
                 "city": "in a city"}.get(kind, f"in a {kind}")
        return f"You are {a} {me} {where}, at {self.setting['time']}."

    # ------------------------------------------------------------------ updates
    def apply_direction(self, d: Direction) -> dict:
        """What the quiet director decided between turns. A quest only moves forward; when
        it is complete the next answer celebrates it and a new one may begin."""
        if d.quest == "" and self.quest and self.progress < vocab.QUEST_STEPS:
            pass                                       # a quest is never dropped half-way
        elif d.quest not in ("same", "") and _same_quest(d.quest, self.quest):
            self.quest = d.quest                       # the same quest, said more precisely
        elif d.quest not in ("same", "") and d.quest.lower() != self.quest.lower():
            if self.quest and self.progress >= vocab.QUEST_STEPS - 1:
                self.quests_done += 1                  # it was all but done: count it
            self.quest, self.progress = d.quest, 0
        if d.shown and self.quest:
            self.quest_shown = d.shown
        if d.progress is not None and self.quest:
            self.progress = max(self.progress, min(vocab.QUEST_STEPS, d.progress, self.progress + 2))
        done = False
        if self.quest and self.progress >= vocab.QUEST_STEPS:
            self.fulfilled, done = self.quest, True
            self.quests_done += 1
            self.quest, self.quest_shown, self.progress = "", "", 0
        self.wants.update({k: v for k, v in d.wants.items() if k in self.things})
        self.hint = d.hint
        self.directed = self.turn
        self.updated = time.time()
        return {"quest": self.quest, "shown": self.quest_shown, "progress": self.progress,
                "steps": vocab.QUEST_STEPS, "fulfilled": done, "next": d.next, "wants": dict(self.wants)}

    def remember(self, role: str, text: str) -> None:
        self.history.append({"role": role, "text": text, "turn": self.turn})
        self.history = self.history[-40:]

    def forget_replies(self, replies: set[str]) -> bool:
        """Drop every exchange that ended in one of these replies (old fallback lines),
        so the model never learns to keep repeating them."""
        kept: list[dict] = []
        changed = False
        for h in self.history:
            if h["role"] == "assistant" and h["text"] in replies:
                changed = True
                if kept and kept[-1]["role"] == "user":
                    kept.pop()
                continue
            kept.append(h)
        if changed:
            self.history = kept
        return changed

    def apply(self, out: TurnOutput, mentioned: set[str] = frozenset(), moved: bool = False) -> dict:
        """Apply one LLM turn. Returns what changed, for the renderer. `mentioned` holds the
        ids of things the player's words or the reply named (they come along to a new place);
        `moved`: the player went somewhere new that keeps the kind of place (another forest)."""
        self.turn += 1
        if self.fulfilled:              # the storyteller has had its turn to celebrate it
            self.fulfilled = ""
        added, changed, removed = [], [], []
        effect = out.effect

        for f in out.facts:
            new = _fact_words(f)
            if not new or any(new <= _fact_words(x) for x in self.facts):
                continue                    # already known, perhaps in more detail
            # a fact that says more replaces the one it extends ("I am a witch" -> "... young witch")
            self.facts = [x for x in self.facts if not _fact_words(x) <= new]
            self.facts.append(f)
        self.facts = self.facts[-vocab.MAX_FACTS:]

        if out.voice != "same":
            self.voice = out.voice

        if out.self_form != "same" and out.self_form != self.self_form:
            self.self_form = out.self_form
            # A new form starts from its own look: the wisp's pale glow or an old label
            # ("a fairy with silver wings") must not carry over to a dragon.
            self.self_label, self.self_color = "", ""
            if effect == "none":
                effect = "sparkle"
        if out.self_label:
            self.self_label = out.self_label
        if out.self_color:
            self.self_color = out.self_color

        new_kind = out.setting_kind
        travelled = new_kind != "same" and new_kind != self.setting["kind"]
        if travelled or moved:
            # A new place: what the storyteller keeps, changes or brings again comes along,
            # and so do the beings the words name; the rest stays behind in the old place.
            # Scenery (trees, houses, the sun) belongs to its place unless it is kept.
            told = set(out.keep) | {a.id for a in out.add}
            along = told | {c.id for c in out.change} | set(mentioned)
            for tid in list(self.things):
                t = self.things[tid]
                if tid in (told if t["kind"] in SCENERY else along):
                    continue
                del self.things[tid]
                removed.append(tid)
            self.hint = ""
            if effect == "none":
                effect = "warp"
        if travelled:
            self.setting["kind"] = new_kind
            # Rain and snow stay behind on the moon, in space, under water and in caves.
            if out.setting_weather == "same" and new_kind in NO_WEATHER:
                self.setting["weather"] = "clear"
        if out.setting_time != "same":
            self.setting["time"] = out.setting_time
        if out.setting_weather != "same":
            self.setting["weather"] = out.setting_weather

        for rid in out.remove:
            if rid in self.things:
                del self.things[rid]
                removed.append(rid)

        for spec in out.add:
            tid = spec.id
            existing = self.things.get(tid)
            if existing and existing["kind"] != spec.kind:
                n = 2
                while f"{tid}-{n}" in self.things:
                    n += 1
                tid = f"{tid}-{n}"
                existing = None
            data = asdict(Thing(id=tid, kind=spec.kind, label=spec.label, color=spec.color,
                                size=spec.size, count=spec.count, where=spec.where,
                                motion=spec.motion, born=self.turn))
            if existing:
                data["born"] = existing["born"]
                self.things[tid] = data
                changed.append(tid)
            else:
                self.things[tid] = data
                added.append(tid)

        for ch in out.change:
            t = self.things.get(ch.id)
            if not t:
                continue
            if ch.color:
                t["color"] = ch.color
            for key in ("size", "where", "motion"):
                val = getattr(ch, key)
                if val != "same":
                    t[key] = val
            if ch.id not in added:
                changed.append(ch.id)

        self.wants = {k: v for k, v in self.wants.items() if k in self.things}
        # The dream forgets its oldest things.
        if len(self.things) > vocab.MAX_THINGS:
            old = sorted((t for t in self.things.values() if t["id"] not in added),
                         key=lambda t: t["born"])
            for t in old[: len(self.things) - vocab.MAX_THINGS]:
                del self.things[t["id"]]
                removed.append(t["id"])

        base = min(0.8, 0.12 + 0.06 * self.turn)
        self.clarity = base if out.clarity is None else 0.5 * base + 0.5 * out.clarity
        if out.ending:
            self.ended = True
            if effect == "none":
                effect = "ascend"
        self.updated = time.time()
        return {"added": added, "changed": sorted(set(changed)), "removed": removed,
                "effect": effect}


class SessionStore:
    """Sessions live in memory and are mirrored to JSON files so a dream survives restarts."""

    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, Session] = {}
        self._lock = threading.Lock()

    def _path(self, sid: str) -> Path:
        return self.root / f"{sid}.json"

    def get(self, sid: str) -> Session | None:
        if not SESSION_ID_RE.fullmatch(sid or ""):
            return None
        with self._lock:
            if sid in self._mem:
                return self._mem[sid]
            p = self._path(sid)
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    sess = Session(**{k: v for k, v in data.items() if k in Session.__dataclass_fields__})
                    self._mem[sid] = sess
                    return sess
                except (OSError, ValueError, TypeError):
                    return None
        return None

    def create(self, sid: str, lang: str) -> Session:
        if not SESSION_ID_RE.fullmatch(sid or ""):
            raise ValueError("bad session id")
        sess = Session(id=sid, lang=lang)
        with self._lock:
            self._mem[sid] = sess
        self.save(sess)
        return sess

    def save(self, sess: Session) -> None:
        tmp = self._path(sess.id).with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(sess), ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path(sess.id))

    def count(self) -> int:
        return len(self._mem)

    @staticmethod
    def blank_summary() -> str:
        return Session(id="warmup").summary()
