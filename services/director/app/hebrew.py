"""Who is a he and who is a she, from how Hebrew is written.

The voice reads unvocalized Hebrew, where the same letters can be a man's or a woman's word:
"אני רוצה" is rotze or rotza, "שלך" is shelcha or shelach. It reads them right when it knows
who speaks and who is spoken to, so the director works both out from the words themselves:
the dreamer's own present-tense words about itself ("אני מרגישה"), the player's ("אני הולך"),
and the way the dreamer addresses the player ("אתה", or "את" as "you").
"""
from __future__ import annotations

import re

HEBREW = re.compile("[א-ת]")
WORD = re.compile("[א-ת]+")

# Words that may sit between "אני" and the word that tells: "אני לא יודעת", "אני כבר מרגיש".
BETWEEN = {"לא", "כבר", "עדיין", "עוד", "פשוט", "ממש", "קצת", "כל", "כך", "גם", "רק", "באמת", "כנראה",
           "בטח", "מאוד", "די", "שוב", "עכשיו", "פה", "כאן", "תמיד"}
# Common words about oneself whose spelling is the same for both (ל"ה verbs and the like).
SHARED = {"רוצה", "רואה", "עושה", "קונה", "שותה", "בונה", "עולה", "שוחה", "בוכה", "צופה", "פונה",
          "עונה", "נהנה", "נראה", "מנסה", "מחכה", "מצפה", "מקווה", "יפה", "שונה", "קשה", "מתגלה",
          "מתנסה", "שם", "חי", "פה", "כאן", "בסדר", "לבד"}
# Adjectives and forms the rules below would miss.
MALE_WORDS = {"יכול", "צריך", "שמח", "עייף", "רעב", "בטוח", "גדול", "קטן", "חזק", "חלש", "עצוב",
              "סקרן", "מבולבל", "אבוד", "ער", "מוכן", "ילד", "בן", "גבר", "בחור", "נער", "מלך", "נסיך",
              "רץ", "עף", "טס", "שט", "צף", "שר", "בא", "קם"}
FEMALE_WORDS = {"ילדה", "בת", "אישה", "בחורה", "נערה", "מלכה", "נסיכה", "פיה", "ערה",
                "רצה", "עפה", "טסה", "שטה", "צפה", "שרה", "באה", "קמה", "יכולה", "צריכה", "שמחה",
                "עייפה", "רעבה", "בטוחה", "גדולה", "קטנה", "חזקה", "חלשה", "עצובה", "סקרנית",
                "מבולבלת", "אבודה", "מוכנה"}
I = {"אני", "ואני", "שאני", "כשאני"}


def _says(word: str) -> str:
    """'female', 'male' or '' for one word a speaker uses about themselves."""
    if word in SHARED:
        return ""
    if word in FEMALE_WORDS:
        return "female"
    if word in MALE_WORDS:
        return "male"
    if len(word) >= 4 and word.endswith("ת") and not word.endswith("ות"):
        return "female"                                  # הולכת, יודעת, נמצאת, מדברת
    if len(word) >= 4 and word.endswith("ה"):
        if re.fullmatch(".ו.ה|[מנ]..ה|מ.ו.ה|מת..ה", word):
            return ""                                    # רוצה, מנסה, מקווה: the same for both
        return "female"                                  # מרגישה, יכולה, שמחה, ממשיכה
    if len(word) >= 4 and (re.fullmatch(".ו..", word) or word[0] in "מנ"):
        return "male"                                    # הולך, יודע, מרגיש, נמצא...
    return ""


def self_gender(text: str) -> str:
    """How the speaker of `text` speaks of themselves: 'female', 'male' or '' (unknown,
    mixed, or no Hebrew)."""
    if not HEBREW.search(text or ""):
        return ""
    words = WORD.findall(text)
    votes = {"female": 0, "male": 0}
    for i, w in enumerate(words):
        if w not in I:
            continue
        rest = [w2 for w2 in words[i + 1:i + 5] if w2 not in BETWEEN]
        g = _says(rest[0]) if rest else ""
        if not g and len(rest) > 1 and (rest[1] in MALE_WORDS or rest[1] in FEMALE_WORDS):
            g = _says(rest[1])               # "אני דרקון גדול": the adjective tells
        if g:
            votes[g] += 1
    if votes["female"] > votes["male"]:
        return "female"
    if votes["male"] > votes["female"]:
        return "male"
    return ""


def _object(word: str) -> bool:
    """A word that "את" marks as an object ("את הדלת", "את זה"), so "את" is not "you"."""
    return word.startswith(("ה", "עצמ")) or word in {"זה", "זו", "זאת", "כל", "מה", "מי", "שלי", "שלך"}


ASKING = {"מה", "איפה", "למה", "איך", "מי", "האם", "אם", "גם", "כי", "לאן", "מאיפה", "כמה", "מתי"}
YOU_MALE = {"אתה", "ואתה", "שאתה", "כשאתה"}
YOU_FEMALE = {"ואת", "שאת", "כשאת"}


def addressed(text: str) -> str:
    """How `text` speaks to its listener: 'male' for "אתה", 'female' for "את" used as "you"
    (at the start of a sentence, after a question word, or as "ואת"/"שאת"), else ''."""
    if not HEBREW.search(text or ""):
        return ""
    male = female = 0
    for sentence in re.split(r"[.!?…,;:\n]+", text):
        words = WORD.findall(sentence)
        for i, w in enumerate(words):
            if w in YOU_MALE:
                male += 1
            elif w in YOU_FEMALE or (w == "את" and (i == 0 or words[i - 1] in ASKING)
                                     and not (i + 1 < len(words) and _object(words[i + 1]))):
                female += 1
    if male > female:
        return "male"
    if female > male:
        return "female"
    return ""
