"""Text to speech with BlueTTS (Hebrew-first, also English), on the CPU through ONNX Runtime.

The model bundle is downloaded from Hugging Face on first start into /models/bluev3.
The dreamer starts with a neutral voice (two voices averaged) and switches to a
female or male voice once the story gives it a gender.
"""
from __future__ import annotations

import io
import logging
import os
import subprocess
import sys
import threading
import time

import numpy as np

log = logging.getLogger("voice.tts")


class NotReady(RuntimeError):
    """The voice is still downloading or loading."""


def folder_mb(path: str) -> int:
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total // (1024 * 1024)


class download_meter:
    """While a model downloads, keep owner.detail showing how many MB have arrived."""

    def __init__(self, owner, folder: str, label: str):
        self.owner, self.folder, self.label = owner, folder, label
        self.done = threading.Event()

    def _tick(self):
        start = folder_mb(self.folder)
        while not self.done.wait(2.0):
            mb = folder_mb(self.folder) - start
            if mb > 0:
                self.owner.detail = f"{self.label}: {mb} MB so far"

    def __enter__(self):
        self.owner.detail = self.label
        threading.Thread(target=self._tick, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.done.set()
        return False


def _voice_list(env: str, default: str) -> list[str]:
    return [v.strip() for v in os.environ.get(env, default).split(",") if v.strip()]


class TextToSpeech:
    def __init__(self):
        self.repo = os.environ.get("TTS_BUNDLE", "notmax123/BlueV3-onnx")
        self.root = os.environ.get("TTS_DIR", "/models/bluev3")
        self.speed = float(os.environ.get("TTS_SPEED", "0.95"))
        self.voices = {
            "neutral": _voice_list("VOICE_NEUTRAL", "noa,ariel"),
            "female": _voice_list("VOICE_FEMALE", "maya"),
            "male": _voice_list("VOICE_MALE", "daniel"),
        }
        self.engine = None
        self.styles: dict[str, object] = {}
        self.sample_rate = 44100
        self.state = "waiting"
        self.detail = ""
        self._lock = threading.Lock()

    def _download(self) -> None:
        from huggingface_hub import snapshot_download
        with download_meter(self, self.root, f"downloading the BlueTTS voice ({self.repo})"):
            snapshot_download(repo_id=self.repo, local_dir=self.root,
                              allow_patterns=["onnx/*", "voices/*", "frontend/*"])

    def _ensure_polyg2p(self) -> None:
        """English G2P ships as a wheel inside the model bundle, not on PyPI."""
        try:
            import polyg2p  # noqa: F401
            return
        except ImportError:
            pass
        frontend = os.path.join(self.root, "frontend")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links",
                            frontend, "polyg2p"], capture_output=True, text=True)
        if r.returncode != 0:
            log.warning("polyg2p install failed, English speech may be unavailable: %s", r.stderr[-400:])

    def _path(self, name: str) -> str:
        return os.path.join(self.root, "voices", f"{name}.json")

    def load(self) -> None:
        try:
            self.state = "loading"
            if not os.path.isdir(os.path.join(self.root, "onnx")):
                self._download()
            self._ensure_polyg2p()
            self.detail = "loading BlueTTS"
            from blue_onnx import BlueTTS
            self.engine = BlueTTS(onnx_dir=os.path.join(self.root, "onnx"),
                                  style_json=[self._path(v) for v in self.voices["neutral"]])
            self.sample_rate = self.engine.sample_rate
            self.styles["neutral"] = self.engine.style
            for key in ("female", "male"):
                try:
                    self.styles[key] = self._blend([self._path(v) for v in self.voices[key]])
                except Exception as e:  # noqa: BLE001
                    log.warning("voice %s unavailable: %s", key, e)
            self.engine.synthesize("שלום", lang="he")     # warm-up
            self.state, self.detail = "ready", f"BlueTTS, voices: {', '.join(self.styles)}"
        except Exception as e:  # noqa: BLE001
            self.state, self.detail = "error", f"{type(e).__name__}: {e}"[:300]
            log.exception("TTS failed to load")

    def _blend(self, paths: list[str]):
        """One speaker from one or more voice files (several are averaged, as BlueTTS does)."""
        from blue_onnx import Style, load_voice_style, mean_speed_scale
        style = load_voice_style(paths)
        self.engine.tts.check_style(style)
        if style.ttl.shape[0] > 1:
            style = Style(style.ttl.mean(axis=0, keepdims=True), style.dp.mean(axis=0, keepdims=True),
                          model=style.model, speed_scale=mean_speed_scale(style.speed_scale))
        return style

    def synthesize(self, text: str, lang: str, voice: str = "neutral") -> tuple[bytes, float]:
        if self.state != "ready":
            raise NotReady(f"text-to-speech not ready ({self.state})")
        import soundfile as sf
        lang = lang if lang in ("he", "en") else "he"
        style = self.styles.get(voice) or self.styles["neutral"]
        t0 = time.perf_counter()
        with self._lock:
            samples, sr = self.engine.synthesize(text, lang=lang, speed=self.speed, style=style)
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        buf = io.BytesIO()
        sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue(), round(time.perf_counter() - t0, 3)
