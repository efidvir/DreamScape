"""What the player names, in Hebrew or English, mapped onto the dream's vocabulary.

A small model sometimes answers beautifully in character and still leaves the scene
untouched. After every turn the director reads the player's own words with these lists
and fills in what the model left out: where the dreamer is, what the dreamer is, and
the things the player mentioned. The model's own choices always win; this only fills gaps.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schema import ThingSpec, TurnOutput
from . import vocab

# ---------------------------------------------------------------------------- word lists
# (surface word or phrase, kind, count). Hebrew entries are matched as whole words after
# removing up to two one-letter prefixes (ו ה ב ל מ ש כ); English ones as whole words.

THING_WORDS: list[tuple[str, str, int]] = [
    # multi-word phrases first: they are removed from the text before single words match
    ("ספינת חלל", "spaceship", 1), ("ספינות חלל", "spaceship", 2), ("חד קרן", "horse", 1),
    ("חד-קרן", "horse", 1), ("כדור הארץ", "planet", 1), ("כוכב לכת", "planet", 1),
    ("רוח רפאים", "ghost", 1), ("רוחות רפאים", "ghost", 3), ("אבן חן", "gem", 1),
    ("אבני חן", "gem", 5), ("כוכב נופל", "star", 1), ("קשת בענן", "rainbow", 1),
    ("כדור אור", "orb", 1), ("כדור קסם", "orb", 1), ("שרביט קסם", "sword", 1),
    ("space ship", "spaceship", 1), ("shooting star", "star", 1), ("the earth", "planet", 1),
    ("magic wand", "sword", 1), ("crystal ball", "orb", 1),
    ("הר געש", "volcano", 1), ("הרי געש", "volcano", 2), ("הר הגעש", "volcano", 1),
    # Hebrew
    ("פיה", "fairy", 1), ("פיות", "fairy", 3), ("פרפר", "butterfly", 1), ("פרפרים", "butterfly", 8),
    ("גחלילית", "fireflies", 12), ("גחליליות", "fireflies", 30), ("ציפור", "bird", 1),
    ("ציפורים", "bird", 6), ("עוף", "bird", 1), ("דג", "fish", 1), ("דגים", "fish", 10),
    ("חתול", "cat", 1), ("חתולה", "cat", 1), ("חתולים", "cat", 3), ("כלב", "dog", 1),
    ("כלבה", "dog", 1), ("כלבים", "dog", 3), ("זאב", "wolf", 1), ("זאבים", "wolf", 3),
    ("שועל", "wolf", 1), ("סוס", "horse", 1), ("סוסה", "horse", 1), ("סוסים", "horse", 3),
    ("צבי", "horse", 1), ("איילה", "horse", 1), ("דרקון", "dragon", 1), ("דרקונים", "dragon", 2),
    ("נחש", "dragon", 1), ("אדם", "person", 1), ("איש", "person", 1), ("אישה", "person", 1),
    ("ילד", "person", 1), ("ילדה", "person", 1), ("נסיך", "person", 1), ("נסיכה", "person", 1),
    ("מלך", "person", 1), ("מלכה", "person", 1), ("קוסם", "person", 1), ("מכשפה", "person", 1),
    ("עץ", "tree", 1), ("עצים", "tree", 14), ("פרח", "flower", 1),
    ("פרחים", "flower", 16), ("פטריה", "mushroom", 1), ("פטרייה", "mushroom", 1),
    ("פטריות", "mushroom", 10), ("סלע", "rock", 1), ("סלעים", "rock", 8), ("אבן", "rock", 1),
    ("אבנים", "rock", 8), ("קריסטל", "crystal", 1), ("קריסטלים", "crystal", 6), ("גביש", "crystal", 1),
    ("גבישים", "crystal", 6), ("ענן", "cloud", 1), ("עננים", "cloud", 5), ("אש", "fire", 1),
    ("מדורה", "fire", 1), ("אגם", "water", 1), ("בריכה", "water", 1), ("נהר", "water", 1),
    ("מים", "water", 1), ("ירח", "moon", 1), ("שמש", "sun", 1), ("כוכב", "star", 1),
    ("כוכבים", "star", 24), ("כוכבי", "star", 24), ("שבתאי", "planet", 1), ("מאדים", "planet", 1),
    ("קשת", "rainbow", 1), ("בית", "house", 1), ("בקתה", "house", 1),
    ("בתים", "house", 3), ("טירה", "castle", 1), ("ארמון", "castle", 1), ("מגדל", "castle", 1),
    ("דלת", "portal", 1), ("שער", "portal", 1), ("פורטל", "portal", 1), ("חללית", "spaceship", 1),
    ("רקטה", "spaceship", 1), ("טיל", "spaceship", 1), ("סירה", "boat", 1), ("ספינה", "boat", 1),
    ("פנס", "lantern", 1), ("מנורה", "lantern", 1), ("גבינה", "cheese", 1), ("גבינות", "cheese", 2),
    ("עוגה", "food", 1), ("תפוח", "food", 1), ("פרי", "food", 1), ("פירות", "food", 4),
    ("לחם", "food", 1), ("אוצר", "gem", 3), ("מטבע", "gem", 1), ("מטבעות", "gem", 6),
    ("יהלום", "gem", 1), ("כתר", "crown", 1), ("מפתח", "key", 1), ("ספרים", "book", 3), ("מגילה", "book", 1), ("חרב", "sword", 1), ("שרביט", "sword", 1), ("כדור", "orb", 1),
    ("לוויתן", "fish", 1), ("לווייתן", "fish", 1), ("דולפין", "fish", 1), ("כריש", "fish", 1),
    ("תמנון", "fish", 1), ("רובוט", "person", 1), ("מלאך", "person", 1), ("ינשוף", "bird", 1),
    ("מלכת", "person", 1), ("נסיכת", "person", 1),
    ("נשר", "bird", 1), ("אריה", "cat", 1), ("נמר", "cat", 1),
    # English
    ("fairy", "fairy", 1), ("fairies", "fairy", 3), ("butterfly", "butterfly", 1),
    ("butterflies", "butterfly", 8), ("firefly", "fireflies", 12), ("fireflies", "fireflies", 30),
    ("bird", "bird", 1), ("birds", "bird", 6), ("fish", "fish", 6), ("cat", "cat", 1),
    ("cats", "cat", 3), ("kitten", "cat", 1), ("dog", "dog", 1), ("dogs", "dog", 3),
    ("puppy", "dog", 1), ("wolf", "wolf", 1), ("wolves", "wolf", 3), ("fox", "wolf", 1),
    ("horse", "horse", 1), ("horses", "horse", 3), ("unicorn", "horse", 1), ("deer", "horse", 1),
    ("dragon", "dragon", 1), ("dragons", "dragon", 2), ("serpent", "dragon", 1), ("snake", "dragon", 1),
    ("person", "person", 1), ("man", "person", 1), ("woman", "person", 1), ("child", "person", 1),
    ("boy", "person", 1), ("girl", "person", 1), ("prince", "person", 1), ("princess", "person", 1),
    ("king", "person", 1), ("queen", "person", 1), ("wizard", "person", 1), ("witch", "person", 1),
    ("ghost", "ghost", 1), ("ghosts", "ghost", 3), ("spirit", "ghost", 1), ("tree", "tree", 1),
    ("trees", "tree", 14), ("flower", "flower", 1), ("flowers", "flower", 16), ("mushroom", "mushroom", 1),
    ("mushrooms", "mushroom", 10), ("rock", "rock", 1), ("rocks", "rock", 8), ("stone", "rock", 1),
    ("stones", "rock", 8), ("crystal", "crystal", 1), ("crystals", "crystal", 6), ("cloud", "cloud", 1),
    ("clouds", "cloud", 5), ("fire", "fire", 1), ("campfire", "fire", 1), ("lake", "water", 1),
    ("pond", "water", 1), ("river", "water", 1), ("moon", "moon", 1), ("sun", "sun", 1),
    ("planet", "planet", 1), ("saturn", "planet", 1), ("mars", "planet", 1), ("star", "star", 1),
    ("stars", "star", 24), ("rainbow", "rainbow", 1), ("house", "house", 1), ("hut", "house", 1),
    ("cabin", "house", 1), ("castle", "castle", 1), ("tower", "castle", 1), ("palace", "castle", 1),
    ("door", "portal", 1), ("gate", "portal", 1), ("portal", "portal", 1), ("spaceship", "spaceship", 1),
    ("rocket", "spaceship", 1), ("boat", "boat", 1), ("ship", "boat", 1), ("lantern", "lantern", 1),
    ("lamp", "lantern", 1), ("cheese", "cheese", 1), ("cake", "food", 1), ("apple", "food", 1),
    ("fruit", "food", 1), ("bread", "food", 1), ("gem", "gem", 1), ("gems", "gem", 5),
    ("treasure", "gem", 3), ("coin", "gem", 1), ("coins", "gem", 6), ("diamond", "gem", 1),
    ("crown", "crown", 1), ("key", "key", 1), ("book", "book", 1), ("scroll", "book", 1),
    ("sword", "sword", 1), ("wand", "sword", 1), ("orb", "orb", 1),
    ("whale", "fish", 1), ("dolphin", "fish", 1), ("shark", "fish", 1), ("octopus", "fish", 1),
    ("whales", "fish", 3), ("dolphins", "fish", 4), ("sharks", "fish", 3),
    ("volcano", "volcano", 1), ("volcanoes", "volcano", 2), ("lava", "volcano", 1),
    ("bear", "wolf", 1), ("bears", "wolf", 3), ("rabbit", "cat", 1), ("rabbits", "cat", 4),
    ("bunny", "cat", 1), ("לבה", "volcano", 1), ("דוב", "wolf", 1), ("ארנב", "cat", 1),
    ("ארנבת", "cat", 1), ("ארנבון", "cat", 1),
    ("robot", "person", 1), ("angel", "person", 1), ("owl", "bird", 1), ("eagle", "bird", 1),
    ("lion", "cat", 1), ("tiger", "cat", 1),
]

# Places. A trip "to the moon" moves the whole dream there; a moon in the sky is a thing.
SETTING_WORDS: list[tuple[str, str]] = [
    ("לירח", "moon"), ("על הירח", "moon"), ("בירח", "moon"), ("to the moon", "moon"),
    ("on the moon", "moon"), ("לחלל", "space"), ("בחלל", "space"), ("החלל", "space"),
    ("outer space", "space"), ("into space", "space"), ("in space", "space"), ("to space", "space"),
    ("מתחת למים", "underwater"), ("מתחת לים", "underwater"), ("מתחת לפני הים", "underwater"),
    ("בתוך הים", "underwater"), ("underwater", "underwater"), ("under the sea", "underwater"),
    ("under water", "underwater"), ("מעל העננים", "clouds"), ("בין העננים", "clouds"),
    ("above the clouds", "clouds"), ("in the clouds", "clouds"), ("among the clouds", "clouds"),
    ("בתוך העננים", "clouds"), ("בעננים", "clouds"), ("לעננים", "clouds"), ("אל העננים", "clouds"),
    ("into the clouds", "clouds"), ("to the clouds", "clouds"), ("inside the clouds", "clouds"),
    ("יער", "forest"), ("יערות", "forest"), ("ג'ונגל", "forest"), ("forest", "forest"),
    ("woods", "forest"), ("jungle", "forest"), ("אחו", "meadow"), ("שדה", "meadow"),
    ("גן", "meadow"), ("meadow", "meadow"), ("field", "meadow"), ("garden", "meadow"),
    ("מדבר", "desert"), ("desert", "desert"), ("dunes", "desert"), ("ים", "ocean"),
    ("אוקיינוס", "ocean"), ("חוף", "ocean"), ("ocean", "ocean"), ("sea", "ocean"),
    ("beach", "ocean"), ("הר", "mountains"), ("הרים", "mountains"), ("mountain", "mountains"),
    ("mountains", "mountains"), ("מערה", "cave"), ("cave", "cave"), ("cavern", "cave"),
    ("עיר", "city"), ("city", "city"), ("town", "city"), ("ארץ מושלגת", "snow"),
    ("snowy land", "snow"), ("קרח", "snow"), ("ice", "snow"), ("glacier", "snow"),
]

WEATHER_WORDS = {"שלג": "snow", "snow": "snow", "snowing": "snow", "גשם": "rain", "rain": "rain",
                 "raining": "rain", "סערה": "storm", "storm": "storm", "ברקים": "storm",
                 "lightning": "storm", "ערפל": "mist", "fog": "mist", "foggy": "mist"}
TIME_WORDS = {"לילה": "night", "night": "night", "midnight": "night", "צהריים": "day",
              "daytime": "day", "noon": "day", "שחר": "dawn", "זריחה": "dawn", "dawn": "dawn",
              "sunrise": "dawn", "בוקר": "dawn", "morning": "dawn", "שקיעה": "dusk", "ערב": "dusk",
              "sunset": "dusk", "dusk": "dusk", "evening": "dusk"}

# Things a place already draws by itself: no need to add them again.
SETTING_HAS = {"forest": {"tree"}, "meadow": {"tree", "flower"}, "desert": {"rock"},
               "ocean": {"water"}, "space": {"star"}, "clouds": {"cloud"}, "underwater": {"water"}}

FEMALE = {"פיה", "נסיכה", "מלכה", "מכשפה", "ילדה", "אישה", "חתולה", "כלבה", "סוסה", "מלכת", "נסיכת",
          "fairy", "princess", "queen", "witch", "girl", "woman", "mermaid", "goddess", "lady"}

# Big moments named outright.
EFFECT_WORDS = {"אלוהים": "ascend", "אלוהית": "ascend", "אלה": "ascend", "god": "ascend",
                "goddess": "ascend", "divine": "ascend", "קסם": "sparkle", "קסום": "sparkle",
                "קסומה": "sparkle", "magic": "sparkle", "magical": "sparkle", "פיצוץ": "flash",
                "explodes": "flash", "explosion": "flash"}
MALE = {"נסיך", "מלך", "קוסם", "ילד", "איש", "prince", "king", "wizard", "boy", "man", "god", "lord"}

# Hebrew words whose meaning is narrower than their kind: the label keeps what the player
# actually said (a fox stays a fox even though foxes are drawn by the wolf builder).
HE_GLOSS = {"שועל": "fox", "צבי": "deer", "איילה": "deer", "חד קרן": "unicorn", "חד-קרן": "unicorn",
            "נחש": "snake", "נסיך": "prince", "נסיכה": "princess", "מלך": "king", "מלכה": "queen",
            "קוסם": "wizard", "מכשפה": "witch", "ילד": "boy", "ילדה": "girl", "איש": "man",
            "אישה": "woman", "עוף": "bird", "רקטה": "rocket", "טיל": "rocket", "ספינה": "ship",
            "מגדל": "tower", "ארמון": "palace", "בקתה": "cabin", "שער": "gate", "דלת": "door",
            "לוויתן": "whale", "לווייתן": "whale", "דולפין": "dolphin", "כריש": "shark",
            "תמנון": "octopus", "רובוט": "robot", "מלאך": "angel", "ינשוף": "owl", "נשר": "eagle",
            "אריה": "lion", "נמר": "tiger", "מלכת": "queen", "נסיכת": "princess",
            "הר געש": "volcano", "הרי געש": "volcano", "הר הגעש": "volcano", "לבה": "lava",
            "דוב": "bear", "ארנב": "rabbit", "ארנבת": "rabbit", "ארנבון": "rabbit"}

PREFIXES = "והבלמשכ"
HE = re.compile(r"[א-ת'\"\-]+")
EN = re.compile(r"[a-z']+")
# "you are a ...", "you become ...". Hebrew: "את פיה", "ואתה דרקון", "הפכת ל..." at the start
# of a clause; "את ה..." is the object marker, so a word with the article never counts.
YOU_EN = re.compile(r"\b(you are|you're|youre|you become|you turn into|you've become)\b(.*)")
YOU_HE = re.compile(r"(?:^|[\s.,!?;:])(ו?(?:את|אתה|הנך|נהיית|תהיי|תהיה)|הפכת ל|הופכת ל|הופך ל)(?=\s)(.*)")
EN_SKIP = {"a", "an", "the", "now", "actually", "really", "just", "also", "still", "very", "so", "my"}
EN_STOP = {"in", "on", "at", "with", "near", "inside", "under", "from", "of", "to", "by", "and",
           "not", "no", "going", "flying", "holding", "standing", "sitting", "looking", "riding"}
HE_SKIP = {"בעצם", "עכשיו", "כבר", "גם", "רק", "היום", "פתאום"}


def _bare(word: str) -> list[str]:
    """The word and the same word without one or two prefix letters."""
    out = [word]
    for n in (1, 2):
        if len(word) - n >= 2 and all(c in PREFIXES for c in word[:n]):
            out.append(word[n:])
    return out


_THING_INDEX = {w: (k, c) for w, k, c in THING_WORDS if " " not in w}
_EN_WORDS: dict[str, set[str]] = {}
for _w, _k, _c in THING_WORDS:
    if _w.isascii():
        _EN_WORDS.setdefault(_k, set()).add(_w)
_THING_PHRASES = [(w, k, c) for w, k, c in THING_WORDS if " " in w]
_SETTING_INDEX = {w: k for w, k in SETTING_WORDS if " " not in w}
_SETTING_PHRASES = [(w, k) for w, k in SETTING_WORDS if " " in w]


@dataclass
class Heard:
    setting: str | None = None
    setting_phrase: bool = False      # named as a place ("above the clouds"), not a passing word
    weather: str | None = None
    time: str | None = None
    effect: str | None = None
    form: str | None = None
    form_word: str = ""
    voice: str | None = None
    things: list[tuple[str, int, str]] = field(default_factory=list)   # kind, count, word


def _words(text: str) -> list[str]:
    return HE.findall(text) + EN.findall(text)


FINAL = {"כ": "ך", "מ": "ם", "נ": "ן", "פ": "ף", "צ": "ץ"}


def _stems(word: str) -> list[str]:
    """The base form of a Hebrew plural or feminine word: דולפינים and דולפינה are דולפין,
    סירות is סירה. Bases shorter than three letters are left out ("ימים" is not "ים")."""
    def final(w: str) -> str:
        return w[:-1] + FINAL.get(w[-1], w[-1]) if w else w
    out = []
    if word.endswith("ים") and len(word) >= 5:
        out.append(final(word[:-2]))
    if word.endswith("ות") and len(word) >= 4:
        out += [word[:-2] + "ה", final(word[:-2])]
    if word.endswith("ית") and len(word) >= 5:
        out.append(final(word[:-2]))
    if word.endswith("ה") and len(word) >= 4:
        out.append(final(word[:-1]))
    return [w for w in out if len(w) >= 3]


def _lookup(word: str, *indexes: dict, stems: bool = False):
    """Exact matches first (so "מים", water, is never read as "ים", sea), then without prefixes,
    then, when asked, the base form of a plural or feminine word (only ever for things:
    "מדברים", talking, is not a desert)."""
    tiers = [[word], _bare(word)[1:]]
    if stems:
        tiers.append([s for w in _bare(word) for s in _stems(w)])
    for forms in tiers:
        for w in forms:
            for n, index in enumerate(indexes):
                if w in index:
                    return n, w, index[w]
    return None, None, None


def _form_after(lead: str, tail: str) -> tuple[str | None, str]:
    """The form word right after "you are" / "את", or None."""
    words = _words(tail)
    if not words:
        return None, ""
    if any("\u05d0" <= c <= "\u05ea" for c in lead):          # Hebrew: the noun comes first
        words = [w for w in words[:3] if w not in HE_SKIP][:1]
        if words and words[0].startswith("ה") and lead.lstrip("ו") == "את":
            return None, ""
    else:                                                     # English: adjectives may come first
        kept = []
        for w in words:
            if w in EN_STOP or w.endswith("ing"):
                break
            if w not in EN_SKIP:
                kept.append(w)
            if len(kept) >= 3:
                break
        words = kept
    for w in words:
        _, found, hit = _lookup(w, _THING_INDEX, stems=True)
        if hit and hit[0] in vocab.SELF_FORMS:
            return hit[0], found
    return None, ""


def read(text: str) -> Heard:
    """What the player's words name, before any judgement about the scene."""
    heard = Heard()
    low = " " + " ".join((text or "").lower().replace("־", " ").split()) + " "

    # 1. places and things named with a phrase ("to the moon", "ספינת חלל"), then removed
    for phrase, kind in _SETTING_PHRASES:
        if f" {phrase} " in low or f" {phrase}" in low and phrase.startswith(("מ", "ב", "ל")):
            if not heard.setting:
                heard.setting, heard.setting_phrase = kind, True
            low = low.replace(phrase, " ")
    for phrase, kind, count in _THING_PHRASES:
        if phrase in low:
            heard.things.append((kind, count, phrase))
            low = low.replace(phrase, " ")

    # 2. what the dreamer is
    m = YOU_EN.search(low) or YOU_HE.search(low)
    if m:
        lead = m.group(1).strip()
        heard.form, heard.form_word = _form_after(lead, m.group(2))
        if heard.form:
            if lead.lstrip("ו") in ("את", "תהיי") or lead == "הופכת ל":
                heard.voice = "female"
            elif lead.lstrip("ו") in ("אתה", "תהיה") or lead == "הופך ל":
                heard.voice = "male"
    if heard.form_word in FEMALE:
        heard.voice = "female"
    elif heard.form_word in MALE:
        heard.voice = "male"

    # 3. single words: places, weather, time and things (the dreamer's form is not a thing)
    form_taken = False
    for w in _words(low):
        which, found, hit = _lookup(w, _SETTING_INDEX, WEATHER_WORDS, TIME_WORDS, _THING_INDEX,
                                    EFFECT_WORDS)
        if which is None:
            _, found, hit = _lookup(w, _THING_INDEX, stems=True)     # דולפינים, ארנבות
            if hit and w.endswith(("ים", "ות")):
                hit = (hit[0], max(hit[1], 3))                      # a plural is a few
            which = 3 if hit else None
        if which is None:
            continue
        if which == 4:
            heard.effect = heard.effect or hit
            continue
        if which == 0:
            if not heard.setting:
                heard.setting = hit
                # "ביער", "לירח", "בעננים": in or to the place, the way a phrase names it
                heard.setting_phrase = w[0] in "בל" and (found != w or found[0] in "בל")
        elif which == 1:
            heard.weather = heard.weather or hit
        elif which == 2:
            heard.time = heard.time or hit
        elif heard.form and found == heard.form_word and not form_taken:
            form_taken = True
        else:
            heard.things.append((hit[0], hit[1], found))
    return heard


def kind_of(label: str) -> str | None:
    """The kind a free-text label names, if the dream can draw it ("erupting volcano" is a
    volcano, "דולפין מעופף" a fish), else None."""
    low = " " + " ".join((label or "").lower().replace("־", " ").split()) + " "
    for phrase, kind, _ in _THING_PHRASES:
        if phrase in low:
            return kind
    for w in _words(low):
        _, _, hit = _lookup(w, _THING_INDEX, stems=True)
        if hit and hit[0] != "other":
            return hit[0]
    return None


# Words for going somewhere: without one of these (or a place named outright) the dream
# stays where it is, so a castle "appearing" never drags the dreamer into a city.
TRAVEL = re.compile(
    r"\b(fly|flies|flew|flying|travel|travels|go to|goes to|went to|arrive|arrives|land|lands|"
    r"teleport|sail|sails|swim to|walk to|walks into|enter|enters|wake up in|wakes up in|"
    r"transported|fall into|falls into|dive|dives|climb|climbs|go outside|goes outside|"
    r"step out|steps out|run to|runs to|ran to|jump into|jumps into|jumped into|"
    r"takes you|take you|took you|carries you|carry you|carried you|brings you|bring you|"
    r"brought you|swallows|swallow|swallowed)\b"
    r"|(?:^|\s)ו?(טס|טסה|טסים|טסות|עף|עפה|עפים|הולך|הולכת|מגיע|מגיעה|נוחת|נוחתת|נכנס|נכנסת|"
    r"מפליג|מפליגה|צולל|צוללת|מטפס|מטפסת|נופל|נופלת|עובר|עוברת|קופץ|קופצת|מתעורר|מתעוררת|"
    r"נשאב|נשאבת|ממריא|ממריאה|נוסע|נוסעת|יוצא|יוצאת|רץ|רצה|שוחה|נבלע|נבלעת|נסחף|נסחפת|"
    r"לוקח אותך|לוקחת אותך|לוקחים אותך|נושא אותך|נושאת אותך|מביא אותך|מביאה אותך|"
    r"בולע אותך|בולעת אותך)(?=\s|$)")

# The player ends the dream: waking up in bed, or saying it is over.
ENDING = re.compile(
    r"\b(wake|wakes|woke) up (in|at) (your|my|his|her|the|a) (bed|room|home)\b|\bthe dream (ends|is over)\b"
    r"|(מתעורר|מתעוררת|מתעוררים|התעוררת|התעוררתי)\S* (במיטה|בבית|בחדר)"
    r"|החלום (נגמר|הסתיים|מסתיים)|סוף החלום")


def _names(label: str, kind: str) -> bool:
    return any(re.search(rf"\b{re.escape(w)}", label) for w in _EN_WORDS.get(kind, set()) | {kind})


def refine_kinds(out: TurnOutput) -> None:
    """The model sometimes files a thing under a kind its own label contradicts: a cheese
    under "food", "a small golden key" under "gem", a castle under "other". When the label
    names exactly one other kind and not its own, the renderer's builder for that kind fits."""
    for t in out.add:
        label = t.label.lower()
        if not label or (t.kind not in ("other", "food") and _names(label, t.kind)):
            continue
        hits = {k for k in _EN_WORDS if k != t.kind and _names(label, k)}
        if len(hits) == 1:
            t.kind = hits.pop()
    # The dreamer itself as "other" ("an old wise owl") gets the shape its label names.
    if out.self_form == "other" and out.self_label:
        hits = {k for k in _EN_WORDS if k != "other" and _names(out.self_label.lower(), k)}
        if len(hits) == 1:
            out.self_form = hits.pop()


def nudge(out: TurnOutput, text: str, current_setting: str, current_form: str,
          current_voice: str, present_kinds: set[str], present_labels: list[str] = (),
          max_add: int = 2, current_label: str = "") -> list[str]:
    """Fill what the model left out. Returns a short note of each change, for the debug log."""
    for t in out.add:                    # "other: erupting volcano" is a volcano the dream can draw
        if t.kind == "other" and (k := kind_of(t.label)):
            t.kind = k
    if out.self_form == "other" and (k := kind_of(out.self_label)):
        out.self_form = k
    heard = read(text)
    notes: list[str] = []
    refine_kinds(out)

    # The model may only move the dream when the player sends it somewhere.
    if (out.setting_kind not in ("same", current_setting) and current_setting != "void"
            and not heard.setting and not TRAVEL.search(" " + (text or "").lower() + " ")):
        notes.append(f"kept setting (model wanted {out.setting_kind})")
        out.setting_kind = "same"
    target_now = out.setting_kind if out.setting_kind != "same" else current_setting
    # A thing that only repeats the place ("a small dark cave" in a cave, "a tall mountain" in
    # the mountains) would float in front of the place it describes.
    place = target_now[:-1] if target_now.endswith("s") else target_now
    before = len(out.add)
    out.add = [t for t in out.add
               if not (target_now != "void" and re.search(rf"\b{place}", t.label.lower())
                       and (t.kind == "other" or not _names(t.label.lower(), t.kind)))]
    if len(out.add) < before:
        notes.append("dropped a thing that repeated the place")

    target = out.setting_kind if out.setting_kind != "same" else current_setting
    # The player named a new place but the model stayed (or wrote the current place again):
    # move when the words really send the dreamer there, not for a mountain seen far away.
    travels = bool(TRAVEL.search(" " + (text or "").lower() + " "))
    if (heard.setting and out.setting_kind in ("same", current_setting)
            and heard.setting != current_setting
            and (current_setting == "void" or heard.setting_phrase or travels)):
        out.setting_kind = target = heard.setting
        out.facts.append(f"I am in a {heard.setting}" if heard.setting not in ("space", "underwater", "clouds")
                         else {"space": "I am in space", "underwater": "I am under the water",
                               "clouds": "I am above the clouds"}[heard.setting])
        if out.effect in ("none", "ripple"):
            out.effect = "warp"
        notes.append(f"setting={heard.setting}")

    if not out.ending and ENDING.search((text or "").lower()):
        out.ending = True
        notes.append("ending")

    if heard.effect and out.effect in ("none", "ripple"):
        out.effect = heard.effect
        notes.append(f"effect={heard.effect}")
    if heard.weather and out.setting_weather == "same":
        out.setting_weather = heard.weather
        notes.append(f"weather={heard.weather}")
    if heard.time and out.setting_time == "same":
        out.setting_time = heard.time
        notes.append(f"time={heard.time}")

    if heard.form and out.self_form in ("same", "other") and heard.form != current_form:
        out.self_form = heard.form
        out.facts.append(f"I am a {heard.form}")
        if out.effect in ("none", "ripple"):
            out.effect = "sparkle"
        notes.append(f"self={heard.form}")
    # Keep the player's own word when it is narrower than the kind ("fox" drawn as a wolf).
    gloss = heard.form_word if heard.form_word.isascii() else HE_GLOSS.get(heard.form_word, "")
    if (heard.form and gloss and gloss != heard.form and out.self_form == heard.form
            and gloss not in out.self_label.lower()):
        out.self_label = f"{gloss}, {out.self_label}" if out.self_label else gloss
        if not any(gloss in f.lower() for f in out.facts):
            out.facts.append(f"I am a {gloss}")
        notes.append(f"label={gloss}")
    if heard.voice and out.voice == "same" and current_voice == "neutral":
        out.voice = heard.voice
        notes.append(f"voice={heard.voice}")

    # The dreamer is not a thing in its own dream: "you are a fox" sometimes comes back as an
    # added fox (or a "dog" labelled fox) next to the dreamer's own new form.
    form = out.self_form if out.self_form not in ("same", current_form) else None
    if form:
        words = {w for w in _EN_WORDS.get(form, set())} | ({heard.form_word} if heard.form_word.isascii() else set())
        before = len(out.add)
        out.add = [t for t in out.add if t.kind != form and not any(
            re.search(rf"\b{re.escape(w)}", t.label.lower()) for w in words if w)]
        if len(out.add) < before:
            notes.append("dropped a thing that was the dreamer itself")

    have = set(present_kinds) | {t.kind for t in out.add} | SETTING_HAS.get(target, set())
    if target == "moon":
        have.add("moon")
    labels = " ".join(t.label.lower() for t in out.add) + " " + " ".join(present_labels)
    added = 0
    me_kind = out.self_form if out.self_form != "same" else current_form
    me_label = (out.self_label or current_label or "").lower()
    for kind, count, word in heard.things:
        if added >= max_add or len(out.add) >= 4:
            break
        if kind in have:
            continue
        if kind == me_kind:
            # One of the dreamer's own kind is most likely the dreamer itself ("the fox runs
            # to the lake"); several, or a different word for it, are others ("three wolves"
            # around a fox).
            en_word = word if word.isascii() else HE_GLOSS.get(word, kind)
            if count <= 1 and (not me_label or kind in me_label or en_word.rstrip("s") in me_label):
                continue
        if (word.isascii() and word in labels) or _names(labels, kind):
            continue                              # the model drew it already, under another kind
        cap = vocab.MAX_COUNT.get(kind, vocab.DEFAULT_MAX_COUNT)
        out.add.append(ThingSpec(id=f"{kind}-{len(present_kinds) + added + 1}", kind=kind,
                                 label=word if word.isascii() else HE_GLOSS.get(word, kind),
                                 count=max(1, min(cap, count)),
                                 where=vocab.DEFAULT_WHERE.get(kind, "around"),
                                 motion=vocab.DEFAULT_MOTION.get(kind, "still")))
        have.add(kind)
        added += 1
        notes.append(f"add={kind}x{count}")
    return notes
