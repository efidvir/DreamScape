"""Stand-ins for Ollama and the voice service, for trying the dream without a GPU.

    python dev/demo.py          # runs fakes + director + web, open http://localhost:8080

The fake Ollama plays a scripted story keyed on words the player uses (English or Hebrew):
fairy/forest, spaceship/moon, cheese, eat/god, dragon, castle, sea, cat, rain, snow, desert,
city, cave, underwater, house, portal, unicorn, flowers, mushrooms, crystals, ghost.
"""
from __future__ import annotations

import asyncio
import io
import json
import math
import re
import struct
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import Response, StreamingResponse

NONE_SELF = {"form": "same", "label": "", "color": ""}
SAME_SET = {"kind": "same", "time": "same", "weather": "same"}


def T(id, kind, label="", color="", size="medium", count=1, where="around", motion="still"):
    return {"id": id, "kind": kind, "label": label, "color": color, "size": size, "count": count,
            "where": where, "motion": motion}


def beat(facts, he, en, *, voice="same", self=None, setting=None, add=(), effect="none",
         clarity=0.4, backdrop="", ending=False, remove=()):
    return {"facts": list(facts), "reply": {"he": he, "en": en}, "voice": voice, "self": self or NONE_SELF,
            "setting": setting or SAME_SET, "add": list(add), "change": [], "remove": list(remove),
            "effect": effect, "clarity": clarity, "backdrop": backdrop, "ending": ending}


STORY = [  # most specific first: the first matching beat wins
    (r"\beat\b|eats|god|אוכלת|אוכל|אלה|אלוהים", beat(["I ate the cheese", "I became the god of the world"],
        "אני טועמת... ואור ממלא אותי. אני רואה הכול, כל עולם, כל כוכב. אני האלה של העולם.",
        "I taste it... and light fills me. I can see everything, every world, every star. I am the god of the world.",
        self={"form": "person", "label": "radiant goddess of the world", "color": "gold"},
        add=[T("halo-crown", "crown", "crown of starlight", "gold", "small", 1, "above", "spin")],
        remove=["moon-cheese"], effect="ascend", clarity=0.85, ending=True)),
    (r"cheese|גבינה", beat(["There is a block of cheese on the moon"],
        "רגע... מה זה הדבר הזהוב הזה? גוש של גבינה, זוהר כמו שמש קטנה. אולי לטעום?",
        "Wait... what is that golden thing? A block of cheese, glowing like a little sun. Should I taste it?",
        add=[T("moon-cheese", "cheese", "glowing block of cheese", "gold", "large", 1, "center", "float")],
        effect="sparkle", clarity=0.6)),
    (r"fairy|פיה", beat(["I am a fairy", "I am in a dark forest"],
        "פיה... כן. אני מרגישה את הכנפיים שלי רועדות. העצים כאן כל כך גבוהים. מה אני מחפשת ביער הזה?",
        "A fairy... yes. I can feel my wings tremble. The trees here are so tall. What am I looking for in this forest?",
        voice="female", self={"form": "fairy", "label": "small fairy with silver wings", "color": "silver"},
        setting={"kind": "forest", "time": "night", "weather": "mist"},
        add=[T("tall-trees", "tree", "tall dark trees", "darkgreen", "large", 24),
             T("fireflies", "fireflies", "fireflies", "gold", "tiny", 40, "around", "drift"),
             T("glow-shrooms", "mushroom", "glowing mushrooms", "violet", "small", 14, "ground")],
        effect="sparkle", clarity=0.35, backdrop="dark misty forest at night")),
    (r"spaceship|rocket|חללית|moon|ירח", beat(["I built a spaceship", "I flew to the moon"],
        "בניתי אותה מעלים ומאור ירח... והיער נעלם מתחתיי. הירח שקט וכסוף. מישהו מחכה לי כאן?",
        "I wove it from leaves and moonlight... and the forest falls away. The moon is silent and silver. Is someone waiting for me here?",
        setting={"kind": "moon", "time": "night", "weather": "clear"},
        add=[T("leaf-ship", "spaceship", "spaceship made of leaves", "palegreen", "medium", 1, "near", "float"),
             T("earth", "planet", "the blue Earth", "blue", "large", 1, "sky", "spin")],
        effect="warp", clarity=0.5, backdrop="grey lunar surface under a black starry sky")),
    (r"dragon|דרקון", beat(["A dragon appeared"], "דרקון! הכנפיים שלו מכסות את השמיים...", "A dragon! Its wings cover the sky...",
        add=[T("dragon", "dragon", "violet dragon", "purple", "large", 1, "sky", "fly")], effect="bloom")),
    (r"castle|טירה", beat(["There is a castle"], "טירה עולה מתוך הערפל...", "A castle rises out of the mist...",
        add=[T("castle", "castle", "old castle", "slategray", "large", 1, "far")])),
    (r"\bsea\b|ocean|ים|אוקיינוס", beat(["I am by the sea"], "הגלים לוחשים...", "The waves whisper...",
        setting={"kind": "ocean", "time": "dusk", "weather": "clear"}, add=[T("boat", "boat", "small boat", "brown", "medium", 1, "near", "float")])),
    (r"\bcat\b|חתול", beat(["A cat walks beside me"], "חתול שחור מתחכך בי.", "A black cat brushes against me.",
        add=[T("cat", "cat", "black cat", "#2a2a33", "medium", 1, "ground", "walk")])),
    (r"rain|storm|גשם|סערה", beat(["It is raining"], "טיפות גשם... ברק רחוק.", "Raindrops... distant lightning.",
        setting={"kind": "same", "time": "same", "weather": "storm"}, effect="darken")),
    (r"snow|שלג", beat(["Snow falls"], "שלג נופל בשקט.", "Snow falls quietly.", setting={"kind": "snow", "time": "dawn", "weather": "snow"})),
    (r"desert|מדבר", beat(["I am in a desert"], "חול חם...", "Warm sand...", setting={"kind": "desert", "time": "day", "weather": "clear"})),
    (r"city|עיר", beat(["I am in a city"], "אורות של עיר...", "City lights...", setting={"kind": "city", "time": "night", "weather": "mist"})),
    (r"cave|מערה", beat(["I am in a cave"], "מערה של גבישים...", "A cave of crystals...", setting={"kind": "cave", "time": "night", "weather": "mist"})),
    (r"underwater|מתחת למים", beat(["I am under water"], "בועות עולות סביבי...", "Bubbles rise around me...",
        setting={"kind": "underwater", "time": "day", "weather": "clear"}, add=[T("fish", "fish", "bright fish", "", "small", 9, "around", "swim")])),
    (r"house|בית", beat(["There is a house"], "בית קטן עם אור בחלון.", "A little house with a light in the window.",
        add=[T("house", "house", "little cottage", "", "medium", 1, "left")])),
    (r"portal|door|שער|דלת", beat(["A portal opened"], "שער של אור נפתח.", "A gate of light opens.",
        add=[T("portal", "portal", "swirling portal", "violet", "medium", 1, "far", "pulse")], effect="ripple")),
    (r"unicorn|חד.קרן", beat(["A unicorn is here"], "חד-קרן לבן!", "A white unicorn!",
        add=[T("unicorn", "horse", "white unicorn", "white", "medium", 1, "right", "walk")], effect="sparkle")),
    (r"flower|פרח", beat(["Flowers grow"], "פרחים פורחים.", "Flowers bloom.", add=[T("flowers", "flower", "glowing flowers", "pink", "small", 30, "ground")])),
    (r"crystal|גביש", beat(["Crystals grow"], "גבישים.", "Crystals.", add=[T("crystals", "crystal", "blue crystals", "cyan", "medium", 6, "ground", "pulse")])),
    (r"ghost|רוח", beat(["A ghost floats"], "רוח רפאים...", "A ghost...", add=[T("ghost", "ghost", "friendly ghost", "", "medium", 1, "left", "float")])),
]


def _story_reply(messages: list[dict]) -> dict:
    last = messages[-1]["content"] if messages else ""
    lang = "he" if "in Hebrew]" in last else "en"
    said = last.split("]", 2)[-1] if "THE PLAYER SAYS" in last else last
    said = said.split("THE PLAYER SAYS")[-1].lower()
    for pattern, b in STORY:
        if re.search(pattern, said):
            out = json.loads(json.dumps(b))
            out["reply"] = b["reply"][lang]
            return out
    out = json.loads(json.dumps(beat([], "אני שומעת אותך מבעד לערפל... ספרו לי עוד.",
                                     "I hear you through the mist... tell me more.", effect="ripple", clarity=0.2)))
    out["reply"] = out["reply"][lang]
    return out


ollama = FastAPI(title="fake ollama")


@ollama.get("/api/version")
async def version():
    return {"version": "fake"}


@ollama.post("/api/show")
async def show(req: Request):
    return {"model": (await req.json()).get("model")}


LOADED: set[str] = set()


@ollama.post("/api/generate")
async def generate(req: Request):
    body = await req.json()
    if body.get("keep_alive") == 0:
        LOADED.discard(body.get("model", ""))
    elif body.get("model"):
        LOADED.add(body["model"])
    return {"done": True}


@ollama.get("/api/ps")
async def ps():
    return {"models": [{"name": m, "model": m} for m in sorted(LOADED)]}


@ollama.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    out = _story_reply(body.get("messages", []))
    order = ["facts", "voice", "reply"]            # the real model writes the words early
    text = json.dumps({**{k: out[k] for k in order if k in out}, **out}, ensure_ascii=False)
    if not body.get("stream"):
        return {"message": {"role": "assistant", "content": text},
                "done": True, "eval_count": 180, "eval_duration": 2_000_000_000}

    async def pieces():
        for i in range(0, len(text), 24):
            yield json.dumps({"message": {"role": "assistant", "content": text[i:i + 24]},
                              "done": False}) + "\n"
            await asyncio.sleep(0.01)
        yield json.dumps({"message": {"role": "assistant", "content": ""}, "done": True,
                          "eval_count": 180, "eval_duration": 2_000_000_000}) + "\n"

    return StreamingResponse(pieces(), media_type="application/x-ndjson")


# --------------------------------------------------------------------------- fake voice
voice = FastAPI(title="fake voice")


@voice.get("/healthz")
async def vhealth():
    return {"service": "voice", "status": "ok", "stt": {"state": "ready", "detail": "fake"},
            "tts": {"state": "ready", "detail": "fake tone"}}


@voice.post("/stt")
async def stt(audio: UploadFile = File(...), lang: str = Form("auto"), prompt: str = Form("")):
    await audio.read()
    return {"text": "you are a fairy in a dark forest", "language": "en", "probability": 1.0}


@voice.post("/tts")
async def tts(req: Request):
    body = await req.json()
    secs = min(6.0, 0.9 + 0.045 * len(body.get("text", "")))
    sr, n = 22050, int(22050 * secs)
    frames = io.BytesIO()
    for i in range(n):
        t = i / sr
        env = (0.5 + 0.5 * math.sin(2 * math.pi * 3.3 * t)) * min(1, t * 8, (secs - t) * 8)
        v = 0.22 * env * (math.sin(2 * math.pi * 196 * t) + 0.4 * math.sin(2 * math.pi * 294 * t))
        frames.write(struct.pack("<h", int(max(-1, min(1, v)) * 32767)))
    data = frames.getvalue()
    head = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16) + b"data" + struct.pack("<I", len(data))
    return Response(content=head + data, media_type="audio/wav")


# --------------------------------------------------------------------------- fake imagery
# Paints simple glowing pictures (no AI) so the picture -> particle path can be tried offline.
imagery = FastAPI(title="fake imagery")


@imagery.get("/healthz")
async def ihealth():
    return {"service": "imagery", "status": "ok", "detail": "fake painter", "workers": 1}


def _picture(prompt: str, kind: str):
    """A simple glowing picture of the prompt (a PIL image), no AI."""
    import colorsys
    import hashlib
    from PIL import Image, ImageDraw, ImageFilter
    h = int(hashlib.md5(prompt.encode()).hexdigest()[:6], 16)
    hue = (h % 360) / 360
    named = {"gold": 0.13, "cheese": 0.14, "silver": 0.58, "fairy": 0.55, "green": 0.33, "purple": 0.78, "blue": 0.6}
    for word, value in named.items():
        if word in prompt.lower():
            hue = value
    rgb = lambda l, s=0.75: tuple(int(255 * c) for c in colorsys.hls_to_rgb(hue, l, s))
    W = 512
    if kind == "backdrop":
        im = Image.new("RGB", (W, W)); d = ImageDraw.Draw(im)
        for y in range(W):
            d.line([(0, y), (W, y)], fill=tuple(int(c * (0.25 + 0.6 * y / W)) for c in rgb(0.45, 0.5)))
        return im.filter(ImageFilter.GaussianBlur(6))
    im = Image.new("RGB", (W, W), "black"); d = ImageDraw.Draw(im)
    p = prompt.lower()
    if "cheese" in p:
        d.polygon([(110, 360), (420, 300), (150, 170)], fill=rgb(0.55, 0.9))
        for x, y, r in [(200, 300, 22), (260, 260, 14), (330, 310, 18)]:
            d.ellipse([x - r, y - r, x + r, y + r], fill=rgb(0.35, 0.9))
    elif "fairy" in p or "goddess" in p or "person" in p:
        d.ellipse([226, 120, 286, 180], fill=rgb(0.85, 0.4))                    # head
        d.polygon([(256, 170), (196, 400), (316, 400)], fill=rgb(0.6, 0.7))     # dress
        for sx in (-1, 1):                                                       # wings
            d.ellipse([256 + sx * 20 - (130 if sx < 0 else 0), 150, 256 + sx * 20 + (130 if sx > 0 else 0), 300], fill=rgb(0.78, 0.5))
    else:
        for k in range(7):
            r = 160 - k * 20
            d.ellipse([256 - r, 256 - r, 256 + r, 256 + r], fill=rgb(0.25 + k * 0.08))
    return im.filter(ImageFilter.GaussianBlur(3))


@imagery.post("/imagine")
async def imagine(req: Request):
    body = await req.json()
    return Response(content=_png(_picture(body.get("prompt", ""), body.get("kind", "sprite"))), media_type="image/png")


@imagery.post("/paint")
async def paint(req: Request):
    """The painter's packed paintings (services/imagery/app/relief.py), from the simple
    pictures above: depth is guessed from brightness, a place is sky, hills and water."""
    import sys
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "imagery"))
    from app import relief
    body = await req.json()
    prompt, kind = body.get("prompt", ""), body.get("kind", "thing")
    if kind == "scene":
        W, H = 1024, 512
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        u, v = x / W, y / H
        top = np.array(_picture(prompt, "sprite").resize((1, 1)).getpixel((0, 0)), np.float32) / 255
        ridge = 0.5 - 0.08 * np.sin(u * 9.3) - 0.04 * np.sin(u * 23.1 + 1)
        sky = (0.15 + 0.5 * v[..., None]) * (0.5 + top)
        hills = np.array([0.18, 0.2, 0.32]) + 0.15 * top
        water = np.array([0.1, 0.22, 0.38]) * (1 + 0.25 * np.sin(u * 120 + v * 260))[..., None]
        rgb = np.where((v < ridge)[..., None], sky, np.where((v < 0.62)[..., None], hills, water))
        near = np.where(v < ridge, 0.0, np.where(v < 0.62, 0.3, 0.35 + 0.65 * (v - 0.62) / 0.38))
        alpha = np.ones((H, W), np.float32)
    else:
        rgb = np.asarray(_picture(prompt, "sprite"), np.float32) / 255
        lum = rgb @ relief.LUMA
        alpha = relief.smoothstep(0.04, 0.16, relief.blur(lum, 2)) * relief.frame(*lum.shape)
        near = relief.normalize_depth(relief.blur(lum, 14), alpha)
    angle, detail = relief.orientation(rgb.astype(np.float32))
    png = relief.pack(np.clip(rgb, 0, 1), alpha, near.astype(np.float32), angle, detail)
    return Response(content=png, media_type="image/png")


def _png(im) -> bytes:
    buf = io.BytesIO(); im.save(buf, format="PNG"); return buf.getvalue()
