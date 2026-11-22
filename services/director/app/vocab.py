"""The dream's visual vocabulary.

The LLM never writes graphics code. It picks from these words, and the browser
renderer (services/web/static/js/dream/) has a builder for every one of them.
Anything outside the list becomes kind "other" with a free-text label.
"""

SETTINGS = [
    "void",        # black mist: the start of every dream
    "forest",
    "meadow",
    "desert",
    "ocean",
    "mountains",
    "cave",
    "city",
    "space",
    "moon",        # standing on the moon's surface
    "clouds",      # above the clouds
    "snow",
    "underwater",
]

TIMES = ["night", "dawn", "day", "dusk"]
WEATHER = ["clear", "mist", "rain", "snow", "storm"]

# kind -> one-line hint for the prompt
THINGS = {
    "fairy": "small glowing winged fairy",
    "butterfly": "butterflies",
    "fireflies": "swarm of tiny lights",
    "bird": "birds",
    "fish": "fish",
    "cat": "cat",
    "dog": "dog",
    "wolf": "wolf or fox",
    "horse": "horse, deer or unicorn",
    "dragon": "dragon or big serpent",
    "person": "a human figure",
    "ghost": "spirit or ghost",
    "tree": "trees",
    "flower": "flowers",
    "mushroom": "mushrooms",
    "rock": "rocks or stones",
    "crystal": "crystals",
    "cloud": "clouds",
    "fire": "fire or campfire",
    "water": "pond, lake or river",
    "moon": "a moon in the sky",
    "sun": "a sun",
    "planet": "a planet (Earth, Saturn...)",
    "star": "stars or a shooting star",
    "rainbow": "rainbow",
    "house": "house or hut",
    "castle": "castle or tower",
    "portal": "door, gate or portal",
    "spaceship": "spaceship or rocket",
    "boat": "boat or ship",
    "lantern": "lantern or lamp",
    "cheese": "a block of cheese",
    "food": "fruit, cake or other food",
    "gem": "gem, treasure or coin",
    "crown": "crown",
    "key": "key",
    "book": "book or scroll",
    "sword": "sword or wand",
    "orb": "magic orb or ball of light",
    "other": "anything else (describe it in label)",
}

THING_KINDS = list(THINGS.keys())

# What the dreamer itself looks like. "wisp" = a nameless spark of light.
SELF_FORMS = ["wisp"] + [k for k in THING_KINDS if k not in ("fireflies", "rain")]

SIZES = {"tiny": 0.35, "small": 0.65, "medium": 1.0, "large": 1.9, "huge": 4.0}
WHERE = ["center", "near", "far", "left", "right", "above", "ground", "around", "sky"]
MOTIONS = ["still", "float", "drift", "orbit", "buzz", "fly", "swim", "walk", "spin", "pulse"]
EFFECTS = ["none", "sparkle", "flash", "ripple", "warp", "ascend", "darken", "bloom"]
VOICES = ["neutral", "female", "male"]

# Sensible defaults when the model leaves something vague.
DEFAULT_MOTION = {
    "fairy": "buzz", "butterfly": "buzz", "fireflies": "drift", "bird": "fly", "fish": "swim",
    "cat": "walk", "dog": "walk", "wolf": "walk", "horse": "walk", "dragon": "fly",
    "person": "still", "ghost": "float", "cloud": "drift", "spaceship": "float",
    "boat": "float", "star": "pulse", "orb": "float", "lantern": "float", "crystal": "pulse",
    "gem": "spin", "crown": "spin", "key": "spin", "book": "float", "sword": "float",
    "cheese": "float", "food": "float", "planet": "spin", "portal": "pulse",
}
DEFAULT_WHERE = {
    "moon": "sky", "sun": "sky", "planet": "sky", "star": "sky", "rainbow": "sky",
    "cloud": "sky", "bird": "sky", "dragon": "sky", "tree": "around", "flower": "ground",
    "mushroom": "ground", "rock": "ground", "water": "ground", "house": "far", "castle": "far",
    "fireflies": "around", "fire": "near", "portal": "far",
}
MAX_COUNT = {"fireflies": 60, "tree": 40, "flower": 40, "star": 40, "mushroom": 24, "rock": 16,
             "butterfly": 12, "bird": 12, "fish": 16, "crystal": 12, "cloud": 10}
DEFAULT_MAX_COUNT = 6

MAX_THINGS = 16          # the dream "forgets" its oldest things beyond this
MAX_FACTS = 30
