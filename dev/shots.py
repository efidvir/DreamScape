"""Drive the demo in headless Chrome and save screenshots of a sample story.

    python dev/demo.py &            # then:
    python dev/shots.py [out_dir]
"""
import asyncio
import os
import sys
from pathlib import Path

from playwright.async_api import async_playwright

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "dev/shots")
URL = "http://localhost:8080/"
STORY = [
    ("he", "את פיה, ואת נמצאת ביער חשוך", 9),
    ("en", "You build a spaceship and fly to the moon", 10),
    ("en", "On the moon there is a block of cheese", 8),
    ("en", "You eat the cheese and become the god of the world", 9),
]
# Set SOFTWARE_GL=1 on machines without a GPU (renders slowly but correctly).
ARGS = ["--ignore-gpu-blocklist", "--autoplay-policy=no-user-gesture-required", "--use-fake-ui-for-media-stream"]
if os.environ.get("SOFTWARE_GL"):
    ARGS += ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
CHROME = os.environ.get("CHROME") or None   # path to Chrome/Edge; default: Playwright's own Chromium


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=CHROME, headless=True, args=ARGS)
        page = await browser.new_page(viewport={"width": 1280, "height": 720})
        errors = []
        page.on("console", lambda m: errors.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        await page.goto(URL)
        await page.wait_for_timeout(2500)
        # Software GL draws ~1 frame/s here; let each frame advance real time.
        await page.evaluate("() => { window.dreamscape.dream.maxDt = 3; }")
        await page.screenshot(path=str(OUT / "00-veil.png"))
        await page.click("#enter")
        await page.wait_for_timeout(7000)
        await page.screenshot(path=str(OUT / "01-void.png"))
        for i, (lang, text, wait) in enumerate(STORY, start=2):
            before = await page.text_content("#dreamer-line")
            await page.fill("#say", text)
            await page.press("#say", "Enter")
            # wait for the dreamer's answer, then for the scene to fade in
            for _ in range(120):
                await page.wait_for_timeout(500)
                if (await page.text_content("#dreamer-line")) != before:
                    break
            await page.wait_for_timeout(wait * 1000)
            await page.screenshot(path=str(OUT / f"{i:02d}-{lang}.png"))
        fps = await page.evaluate("() => window.dreamscape.dream.renderer.info.render.calls")
        print("draw calls last frame:", fps)
        await browser.close()
    print("\n".join(errors[:40]) or "no console errors")


asyncio.run(main())
