"""Text to speech with BlueTTS (Hebrew-first, also English) through ONNX Runtime, on the GPU
when TTS_DEVICE=cuda and the graphics card has room, otherwise on the CPU.

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


def gpu_providers(mem_mb: int) -> list | None:
    """ONNX Runtime providers that run the voice on the GPU, or None when this ONNX Runtime
    cannot (a CPU-only build, or CUDA libraries it cannot load). The memory cap keeps the
    voice from crowding the storyteller's model on an 8 GB card."""
    try:
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(cuda=True, cudnn=True, msvc=False)
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return None
    except Exception as e:  # noqa: BLE001
        log.warning("the GPU voice is unavailable (%s: %s)", type(e).__name__, e)
        return None
    opts = {"device_id": "0", "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "HEURISTIC", "cudnn_conv_use_max_workspace": "0",
            "gpu_mem_limit": str(int(mem_mb) * 1024 * 1024)}
    return [("CUDAExecutionProvider", opts), "CPUExecutionProvider"]


GENDER_HINT = {"male": 1, "female": 2}


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
        self.device = os.environ.get("TTS_DEVICE", "cpu").lower()     # cuda or cpu
        self.gpu_mb = int(os.environ.get("TTS_GPU_MB", "900"))
        self.active_device = ""
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

    def _engine(self, device: str):
        from blue_onnx import BlueTTS
        providers = gpu_providers(self.gpu_mb) if device in ("cuda", "gpu") else None
        if device in ("cuda", "gpu") and not providers:
            log.warning("TTS_DEVICE=%s, but ONNX Runtime cannot use the GPU here; the voice runs on the CPU",
                        device)
        engine = BlueTTS(onnx_dir=os.path.join(self.root, "onnx"),
                         style_json=[self._path(v) for v in self.voices["neutral"]],
                         providers=providers or ["CPUExecutionProvider"])
        used = engine.tts.vector_est_ort.get_providers()[0] if hasattr(engine, "tts") else ""
        return engine, ("cuda" if used == "CUDAExecutionProvider" else "cpu")

    def load(self, device: str | None = None) -> None:
        try:
            self.state = "loading"
            if not os.path.isdir(os.path.join(self.root, "onnx")):
                self._download()
            self._ensure_polyg2p()
            self.detail = "loading BlueTTS"
            with self._lock:     # a voice switch waits for the clip being spoken
                self.engine, self.active_device = self._engine((device or self.device).lower())
            self.sample_rate = self.engine.sample_rate
            self.styles["neutral"] = self.engine.style
            for key in ("female", "male"):
                try:
                    self.styles[key] = self._blend([self._path(v) for v in self.voices[key]])
                except Exception as e:  # noqa: BLE001
                    log.warning("voice %s unavailable: %s", key, e)
            self.engine.synthesize("שלום", lang="he")     # warm-up
            self.state = "ready"
            self.detail = f"BlueTTS on {self.active_device}, voices: {', '.join(self.styles)}"
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

    def _genders(self, speaker: int, listener: int) -> None:
        """Hebrew is read by RenikudPlus, which takes who speaks and who is spoken to
        (0 unknown, 1 male, 2 female): "אני רוצה" is rotze or rotza, "לך" lecha or lach."""
        g2p = getattr(getattr(self.engine, "tts", None), "g2p", None)
        if g2p is not None:
            g2p.speaker, g2p.target_speaker = int(speaker), int(listener)

    def synthesize(self, text: str, lang: str, voice: str = "neutral", speaker: int | None = None,
                   listener: int = 0) -> tuple[bytes, float]:
        if self.state != "ready":
            raise NotReady(f"text-to-speech not ready ({self.state})")
        import soundfile as sf
        lang = lang if lang in ("he", "en") else "he"
        style = self.styles.get(voice) or self.styles["neutral"]
        if speaker is None:
            speaker = GENDER_HINT.get(voice, 0)
        t0 = time.perf_counter()
        with self._lock:
            self._genders(speaker, listener)
            samples, sr = self.engine.synthesize(text, lang=lang, speed=self.speed, style=style)
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        buf = io.BytesIO()
        sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue(), round(time.perf_counter() - t0, 3)

    def phonemes(self, text: str, speaker: int = 0, listener: int = 0) -> dict:
        """How the voice will read Hebrew text (IPA), to check a reading without listening."""
        if self.state != "ready":
            raise NotReady(f"text-to-speech not ready ({self.state})")
        g2p = self.engine.tts.g2p
        with self._lock:
            self._genders(speaker, listener)
            ipa = g2p.phonemize(text, "he")
            model = g2p._load_renikud()
        return {"ipa": ipa, "gender_aware": bool(getattr(model, "supports_gender", False))}
