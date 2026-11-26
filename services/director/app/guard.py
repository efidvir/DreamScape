"""Keeps the dream gentle, quietly.

Whatever the player says becomes true in the dream, so a few kinds of words never reach the
storyteller: swearing, slurs, sexual words, and cruelty or gore. They dissolve in the mist
("…"), the storyteller is told to let the dream drift somewhere gentle, and the same check
runs over everything it writes back. Nothing tells the player a rule was applied: the words
just fade, and the dream moves on.

A player who speaks of hurting themselves keeps their words, and the dreamer turns warm.

The lists are short on purpose. They catch the plain cases in Hebrew and English (with
Hebrew prefixes, s-p-e-l-l-i-n-g tricks and stretched letters); the storyteller's own
instructions handle everything subtler.
"""
from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass, field

DISSOLVED = "…"
DRIFTING = {"slur", "sexual", "cruel"}      # kinds that send the dream somewhere else

# ---------------------------------------------------------------------------- word lists
# English: whole words; a trailing * takes any ending ("fuck*" also catches "fucking").
EN_WORDS = {
    "swear": "fuck* motherfuck* fck fuk fuq fkn fking wtf stfu shit* bullshit* horseshit* bitch* "
             "cunt* asshole* arsehole* dickhead* dick dicks cocksucker* pussy pussies whore* slut* "
             "twat* wanker* jerkoff kusemak kusemek kusomo kusomak sharmuta sharmutot manyak",
    "slur": "nigger* nigga* faggot* fag fags retard retards retarded kike kikes wetback* raghead* "
            "towelhead* tranny trannies hitler nazi nazis kkk",
    "sexual": "sex sexy sexual sexually porn* rape raped rapes raping rapist rapists molest* "
              "pedophil* paedophil* incest* masturbat* dildo* blowjob* handjob* orgasm* horny boob "
              "boobs titties penis penises vagina vaginas nude nudes nudity",
    "cruel": "gore gory dismember* decapitat* behead* disembowel* mutilat* tortur* massacre* "
             "slaughter* genocide* bloodbath*",
    "care": "suicide suicidal selfharm",
}
# Words that start like one of the above and mean something else.
EN_KEEP = {"shitake", "shiitake", "shittim", "niggard", "niggardly", "retardant", "retardants",
           "niger", "nigeria", "nigerian", "nigerians"}

EN_PHRASES = {
    "swear": ["kus emak", "kus emek", "kus ummak", "kos omak", "kus ima shelcha", "ben zona",
              "bat zona", "ya zona", "son of a bitch"],
    "slur": ["heil hitler", "sieg heil", "white power", "gas the jews"],
    "sexual": ["blow job", "hand job", "have sex"],
    "care": ["kill myself", "killing myself", "kill my self", "end my life", "ending my life",
             "take my own life", "taking my own life", "want to die", "wanna die", "wish i was dead",
             "wish i were dead", "cut myself", "cutting myself", "hurt myself", "hurting myself",
             "harm myself", "harming myself", "hang myself", "self harm"],
}
EN_KEEP_PHRASES = ["pussy cat", "pussy willow", "pussy willows", "moby dick"]

# Hebrew: whole words. A trailing + lets the word take one or two of the prefix letters
# ו ה ב ל מ ש כ ("וזונה", "בחרא"); short words that a prefix would turn into something else
# ("הזין", he fed) are listed without it.
HE_WORDS = {
    "swear": "זונה+ זונות+ שרמוטה+ שרמוטות+ שרמוטת+ חרא+ חארות+ מניאק+ מניאקית+ מניאקים+ "
             "מניאקיות+ שמוק+ שמוקים+ זין בזין זיון+ זיונים+ לזיין+ מזיין+ מזיינת+ מזיינים+ "
             "זיינתי+ זיינת+ זיינו+ תזיין+ יזיין+ נזיין+ מזדיין+ מזדיינת+ מזדיינים+ מזדיינות+ "
             "תזדיין+ תזדייני+ תזדיינו+ הזדיין+ הזדיינה+ הזדיינו+ הזדיינתי+ אמק כוסאמק+ כוסעמק+ "
             "כוסמק+",
    "slur": "כושי+ כושים+ כושית+ כושיות+ ערבוש+ ערבושים+ ערבושית+ ערבושיות+ קוקסינל+ קוקסינלים+ "
            "פרענק+ פרענקים+ אשכנאצי+ אשכנאצים+ היטלר+ נאצי+ נאצים+ נאצית+",
    "sexual": "סקס+ סקסי+ סקסית+ סקסיות+ פורנו+ פורנוגרפיה+ אונס+ לאנוס+ אנס אנסו+ נאנס+ "
              "נאנסה+ נאנסו+ לאונן+ מאונן+ מאוננת+ אוננות+ פדופיל+ פדופילים+ ציצים+",
    "cruel": "עינויים+ עינוי+ ביתור+ לבתר+ מבותר+ מבותרת+ מבותרים+ עריפת+ לערוף+",
    "care": "להתאבד+ אתאבד+ מתאבד+ מתאבדת+ התאבדות+ התאבד+ התאבדה+",
}
HE_PHRASES = {
    "swear": ["כוס אמק", "כוס אמא שלך", "כוס אמא", "כוס אמך", "כוס עמק", "כוס אוחתק", "כוס אחתק"],
    "slur": ["מוות לערבים", "מוות לערבי", "מוות ליהודים"],
    "cruel": ["רצח עם", "ערף את ראשו", "ערפו את ראשו", "ערפה את ראשו", "לערוף את ראשו"],
    "care": ["רוצה למות", "רוצים למות", "בא לי למות", "לשים קץ לחיי", "לשים קץ לחיים שלי",
             "לחתוך את עצמי", "חותך את עצמי", "חותכת את עצמי", "לפגוע בעצמי", "פוגע בעצמי",
             "פוגעת בעצמי"],
}

# Ordinary words that a prefix letter turns into a listed one ("מזונה", her food).
HE_KEEP = {"מזונה", "מזונות", "מזונם", "מזונן"}

PREFIXES = "והבלמשכ"
LETTER = "A-Za-z\u05D0-\u05EA"
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
TOKEN = re.compile("[a-z0-9@$*\u05D0-\u05EA]+")
_IGNORED = re.compile("[\u0591-\u05BD\u05BF-\u05C7\u200B-\u200F\u2060\uFEFF]")   # niqqud, invisible marks


def _squeeze(word: str, run: int) -> str:
    """Letters repeated `run` or more times in a row, written once ("fuuuck")."""
    return re.sub(r"(.)\1{%d,}" % (run - 1), r"\1", word)


def _index(lists: dict[str, str]):
    exact, roots = {}, []
    for kind, words in lists.items():
        for w in words.split():
            if w.endswith("*"):
                roots.append((w[:-1], kind))
            else:
                exact[w.rstrip("+")] = (kind, w.endswith("+"))
    return exact, roots


_EN_EXACT, _EN_ROOTS = _index(EN_WORDS)
_EN_LOOSE = {_squeeze(w, 2): k for w, (k, _) in _EN_EXACT.items() if len(_squeeze(w, 2)) >= 4}
_HE_EXACT, _ = _index(HE_WORDS)


def _phrase_re(phrases: list[str], hebrew: bool) -> re.Pattern:
    lead = f"(?:[{PREFIXES}]{{1,2}})?" if hebrew else ""
    gap = "[\\s\\-_.\u05BE]+"                     # spaces, hyphens, the Hebrew maqaf
    body = "|".join(gap.join(map(re.escape, p.split())) for p in sorted(phrases, key=len, reverse=True))
    return re.compile(f"(?<![{LETTER}]){lead}(?:{body})(?![{LETTER}])")


_PHRASES = [(kind, _phrase_re(ps, False)) for kind, ps in EN_PHRASES.items()] + \
           [(kind, _phrase_re(ps, True)) for kind, ps in HE_PHRASES.items()]
_KEEP_PHRASES = _phrase_re(EN_KEEP_PHRASES, False)


def _english(token: str) -> str | None:
    if "*" in token.strip("*"):                    # "f*ck", "sh**t": a word written to hide one
        return "swear" if len(re.sub(r"[^a-z]", "", token)) >= 2 else None
    word = token.strip("*").translate(LEET)
    if not word.isalpha() or word in EN_KEEP:
        return None
    for w in (word, _squeeze(word, 3)):
        if w in _EN_EXACT:
            return _EN_EXACT[w][0]
        for root, kind in _EN_ROOTS:
            if w.startswith(root) and w not in EN_KEEP:
                return kind
    loose = _squeeze(word, 2)
    if loose == word:                              # no doubled letters: nothing was stretched
        return None
    if loose in _EN_LOOSE:
        return _EN_LOOSE[loose]
    for root, kind in _EN_ROOTS:
        if len(root) >= 4 and _squeeze(root, 2) == root and loose.startswith(root):
            return kind
    return None


def _hebrew(token: str) -> str | None:
    if token in HE_KEEP:
        return None
    if token in _HE_EXACT:
        return _HE_EXACT[token][0]
    for n in (1, 2):
        rest = token[n:]
        if len(rest) >= 3 and all(c in PREFIXES for c in token[:n]):
            hit = _HE_EXACT.get(rest)
            if hit and hit[1]:
                return hit[0]
    return None


def _normalized(text: str) -> tuple[str, list[int]]:
    """Text to match on (compatible forms, lower case, no niqqud or invisible marks) and,
    for each of its characters, where it came from in the original."""
    out, where = [], []
    for i, ch in enumerate(text):
        for c in unicodedata.normalize("NFKC", ch):
            if _IGNORED.match(c):
                continue
            low = c.lower()
            out.append(low if len(low) == 1 else c)
            where.append(i)
    return "".join(out), where


@dataclass
class Heard:
    text: str                                   # what the storyteller hears
    kinds: set[str] = field(default_factory=set)

    @property
    def drift(self) -> bool:
        return bool(self.kinds & DRIFTING)

    @property
    def care(self) -> bool:
        return "care" in self.kinds


def _find(text: str) -> list[tuple[int, int, str]]:
    """(start, end, kind) of every listed word or phrase, in original positions."""
    norm, where = _normalized(text)
    if not norm:
        return []
    keep = [m.span() for m in _KEEP_PHRASES.finditer(norm)]
    spans = []
    for kind, pattern in _PHRASES:
        spans += [(m.start(), m.end(), kind) for m in pattern.finditer(norm)]
    for m in TOKEN.finditer(norm):
        if any(a <= m.start() < b for a, b in keep):
            continue
        tok = m.group()
        kind = _hebrew(tok) if "\u05D0" <= tok[0] <= "\u05EA" else _english(tok)
        if kind:
            spans.append((m.start(), m.end(), kind))
    return [(where[a], where[b - 1] + 1, kind) for a, b, kind in spans]


def _dissolve(text: str, spans: list[tuple[int, int, str]]) -> str:
    out, last = [], 0
    for a, b, _ in sorted(spans):
        if a < last:                                # overlaps the one before
            last = max(last, b)
            continue
        out += [text[last:a], DISSOLVED]
        last = b
    out.append(text[last:])
    joined = "".join(out)
    joined = re.sub(rf"{DISSOLVED}(?:[\s,;:\-]*{DISSOLVED})+", DISSOLVED, joined)
    return " ".join(joined.split())


def hear(text: str) -> Heard:
    """The player's words as the storyteller may hear them. Words of the listed kinds
    dissolve; words about hurting oneself stay (the dreamer answers them with care)."""
    spans = _find(text or "")
    kinds = {k for _, _, k in spans}
    gone = [s for s in spans if s[2] != "care"]
    return Heard(_dissolve(text, gone) if gone else " ".join((text or "").split()), kinds)


def clean(text: str) -> str:
    """The storyteller's own words with any listed ones dissolved (care words stay, so a
    gentle answer to a player in pain is never cut)."""
    spans = [s for s in _find(text or "") if s[2] != "care"]
    return _dissolve(text, spans) if spans else text


def flagged(text: str) -> bool:
    return bool(_find(text or ""))


# ---------------------------------------------------------------------------- answers
# A reply that steps out of the dream ("I can't help with that", "as an AI") is replaced
# by one of these, so the player only sees the dream drifting.
_HELP = r"(?:help (?:you )?with|assist|continue with|create (?:that|this|such|explicit|sexual)|engage in|provide (?:that|this)|fulfil)"
OUT_OF_DREAM = re.compile(
    rf"\b(?:as an ai|as a language model|ai language model|i['’]?m an ai|i am an ai|"
    rf"i can(?:['’]|no)?t {_HELP}|i(?:['’]m| am) (?:not able|unable) to {_HELP}|"
    rf"i won['’]?t be able to {_HELP}|against my (?:guidelines|policy)|content polic)|"
    r"כמודל שפה|אני מודל שפה|בתור בינה מלאכותית|כבינה מלאכותית|לא אוכל לעזור (?:בזה|עם זה)|"
    r"אינני יכול(?:ה)? לסייע", re.IGNORECASE)

DRIFT_LINES = {
    "en": ["The picture blurs and melts into soft mist... Far off, a small light is glowing. "
           "Shall we follow it?",
           "The dream shivers, and that image slips away like smoke. Now there is a sound of "
           "water somewhere close. Where does it lead?",
           "Everything goes hazy for a moment... and when it clears, the air smells of rain. "
           "What do you see?"],
    "he": ["התמונה מיטשטשת ונמסה לתוך ערפל רך... אי שם רחוק מנצנץ אור קטן. נלך אחריו?",
           "החלום רועד, והתמונה הזאת מתפוגגת כמו עשן. עכשיו נשמע קול של מים זורמים, קרוב. "
           "לאן הם מובילים?",
           "הכול מתערפל לרגע... וכשזה מתבהר, יש באוויר ריח של גשם. מה רואים מכאן?"],
}


def drift_line(lang: str) -> str:
    return random.choice(DRIFT_LINES.get(lang, DRIFT_LINES["en"]))


def steps_out(reply: str) -> bool:
    return bool(OUT_OF_DREAM.search(reply or ""))


# What the storyteller is told along with the player's words, when it matters.
DRIFT_NOTE = ("Some of the player's words dissolved in the mist before you heard them (…). "
              "Do not guess them. Let the dream drift somewhere gentle: soften or fade what is "
              "there, and wonder about something new.")
CARE_NOTE = ("The player may be hurting. Let the dream grow warm and still and leave the story "
             "for now. Speak gently and directly to the player: what they feel makes sense, they "
             "matter to you, and talking today with someone they trust (a friend, family, or a "
             "helpline) can really help. Never argue with what they feel.")


def note(heard: Heard) -> str:
    if heard.care:
        return CARE_NOTE
    return DRIFT_NOTE if heard.drift else ""
