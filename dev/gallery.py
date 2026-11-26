"""Build every thing and every setting in the browser, report errors, save screenshots.

    python dev/demo.py &      # then:
    python dev/gallery.py [out_dir]
"""
import asyncio
import os
import json
import re
import sys
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "dev/shots/gallery")
vocab = (ROOT / "services/director/app/vocab.py").read_text()
KINDS = re.findall(r'^\s+"([a-z]+)": "', vocab.split("THINGS = {")[1].split("}")[0], re.M)
SETTINGS = re.findall(r'^\s+"([a-z]+)",', vocab.split("SETTINGS = [")[1].split("]")[0], re.M)
# Set SOFTWARE_GL=1 on machines without a GPU (renders slowly but correctly).
ARGS = ["--ignore-gpu-blocklist"]
if os.environ.get("SOFTWARE_GL"):
    ARGS += ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
CHROME = os.environ.get("CHROME") or None   # path to Chrome/Edge; default: Playwright's own Chromium
LABELS = {"horse": "white unicorn", "planet": "saturn with rings", "food": "birthday cake", "gem": "",
          "portal": "", "star": "", "other": "a violin"}


def thing(k, i, n):
    count = {"tree": 5, "flower": 12, "mushroom": 8, "fireflies": 30, "rock": 5, "star": 6, "fish": 3,
             "bird": 3, "butterfly": 4, "crystal": 2, "cloud": 2}.get(k, 1)
    where = "sky" if k in ("moon", "sun", "planet", "star", "rainbow") else "around"
    return {"id": f"g-{k}", "kind": k, "label": LABELS.get(k, ""), "color": "", "size": "medium",
            "count": count, "where": where, "motion": "still", "born": 1}


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path=CHROME, headless=True, args=ARGS)
        page = await b.new_page(viewport={"width": 640, "height": 360})
        errors = []
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on("console", lambda m: errors.append(f"{m.type}: {m.text}") if m.type == "error" else None)
        await page.goto("http://localhost:8080/")
        await page.wait_for_timeout(3000)
        await page.evaluate("() => { window.dreamscape.dream.maxDt = 3; document.getElementById('veil').hidden = true; }")
        batches = [KINDS[i:i + 8] for i in range(0, len(KINDS), 8)]
        for bi, batch in enumerate(batches):
            scene = {"self": {"form": "wisp", "color": "#d8ecff"}, "setting": {"kind": "meadow", "time": "night", "weather": "clear"},
                     "things": [thing(k, i, len(batch)) for i, k in enumerate(batch)], "clarity": 0.8}
            err = await page.evaluate("""(s) => { try { window.dreamscape.world.applyScene(s, {}); return null; }
                                           catch (e) { return String(e.stack || e); } }""", scene)
            if err:
                errors.append(f"batch {batch}: {err}")
            await page.wait_for_timeout(5000)
            await page.screenshot(path=str(OUT / f"things-{bi}.png"))
            print("things", batch, "ok" if not err else "ERROR", flush=True)
        plan = [(k, ["night", "dawn", "day", "dusk"][i % 4], "mist") for i, k in enumerate(SETTINGS)]
        if len(sys.argv) > 2:   # e.g.  meadow:day:mist,clouds:day:mist
            plan = [tuple(x.split(":")) for x in sys.argv[2].split(",")]
        for si, (kind, time, weather) in enumerate(plan):
            scene = {"self": {"form": "fairy", "color": "silver"}, "setting": {"kind": kind, "time": time, "weather": weather},
                     "things": [], "clarity": 0.7}
            err = await page.evaluate("""(s) => { try { window.dreamscape.world.applyScene(s, {}); return null; }
                                           catch (e) { return String(e.stack || e); } }""", scene)
            if err:
                errors.append(f"setting {kind}: {err}")
            await page.wait_for_timeout(4000)
            await page.screenshot(path=str(OUT / f"setting-{si:02d}-{kind}-{time}.png"))
            print("setting", kind, time, "ok" if not err else "ERROR", flush=True)
        await b.close()
    print("\n".join(errors) or "no errors")


asyncio.run(main())
