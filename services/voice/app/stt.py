"""Speech to text with faster-whisper.

Default model: ivrit.ai's Hebrew fine-tune of Whisper large-v3-turbo (CTranslate2).
English goes to the same model unless STT_MODEL_EN names a second one.
"""
from __future__ import annotations

import io
import logging
import os
import threading
import time

log = logging.getLogger("voice.stt")


class NotReady(RuntimeError):
    """The model is still downloading or loading."""


def decode_audio(data: bytes, sampling_rate: int = 16000):
    """Any browser recording (WebM/Opus, Ogg, MP4, WAV) to 16 kHz mono float32.

    faster-whisper's own decoder passes an argument that newer PyAV releases removed,
    so this one follows the same steps with the plain PyAV API."""
    import av
    import numpy as np

    resampler = av.AudioResampler(format="s16", layout="mono", rate=sampling_rate)
    chunks = []
    with av.open(io.BytesIO(data), mode="r") as container:
        frames = container.decode(audio=0)
        while True:
            try:
                frame = next(frames)
            except StopIteration:
                break
            except av.error.InvalidDataError:
                continue
            frame.pts = None
            chunks += [f.to_ndarray() for f in resampler.resample(frame)]
        chunks += [f.to_ndarray() for f in resampler.resample(None)]
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([c.reshape(-1) for c in chunks]).astype(np.float32) / 32768.0


class SpeechToText:
    def __init__(self):
        self.model_he = os.environ.get("STT_MODEL", "ivrit-ai/whisper-large-v3-turbo-ct2")
        self.model_en = os.environ.get("STT_MODEL_EN", "").strip() or self.model_he
        self.device = os.environ.get("STT_DEVICE", "cuda")
        self.compute_type = os.environ.get("STT_COMPUTE_TYPE", "int8_float16")
        self.beam_size = int(os.environ.get("STT_BEAM", "2"))
        self.root = os.environ.get("WHISPER_DIR", "/models/whisper")
        self.models: dict[str, object] = {}
        self.state = "waiting"
        self.detail = ""
        self.active_device = None
        self._lock = threading.Lock()

    def _load_one(self, name: str):
        """Load a model and run it once. CUDA problems (missing GPU, missing library) can
        surface on the first run instead of at load time, so the warm-up decides the device."""
        from faster_whisper import WhisperModel
        from .tts import download_meter
        try:
            with download_meter(self, self.root, f"loading {name} (downloads once)"):
                m = WhisperModel(name, device=self.device, compute_type=self.compute_type,
                                 download_root=self.root)
            self._warm_up(m)
            self.active_device = self.device
        except Exception as e:  # noqa: BLE001 - no usable GPU: keep working on the CPU
            if self.device == "cpu":
                raise
            log.warning("running %s on %s failed (%s); falling back to CPU", name, self.device, e)
            m = WhisperModel(name, device="cpu", compute_type="int8", download_root=self.root)
            self._warm_up(m)
            self.active_device = "cpu"
        return m

    @staticmethod
    def _warm_up(model) -> None:
        import numpy as np
        list(model.transcribe(np.zeros(16000, dtype=np.float32), language="he", beam_size=1)[0])

    def load(self) -> None:
        try:
            self.state, self.detail = "loading", f"loading {self.model_he}"
            self.models[self.model_he] = self._load_one(self.model_he)
            self.state, self.detail = "ready", f"{self.model_he} on {self.active_device}"
        except Exception as e:  # noqa: BLE001
            self.state, self.detail = "error", f"{type(e).__name__}: {e}"[:300]
            log.exception("STT failed to load")

    def _model_for(self, lang: str | None):
        name = self.model_en if lang == "en" else self.model_he
        if name not in self.models:
            with self._lock:
                if name not in self.models:
                    self.models[name] = self._load_one(name)
        return self.models[name]

    def transcribe(self, audio: bytes, lang: str | None, prompt: str | None = None) -> dict:
        if self.state != "ready":
            raise NotReady(f"speech-to-text not ready ({self.state})")
        t0 = time.perf_counter()
        language = lang if lang in ("he", "en") else None
        samples = decode_audio(audio)
        if samples.size < 1600:          # under 0.1 s: nothing to hear
            return {"text": "", "language": language or "", "probability": 0.0,
                    "duration_s": round(samples.size / 16000, 2), "took_s": 0.0}
        opts = dict(beam_size=self.beam_size, vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 400},
                    condition_on_previous_text=False, without_timestamps=True,
                    initial_prompt=prompt or None)
        segments, info = self._model_for(language).transcribe(samples, language=language, **opts)
        if language is None and info.language not in ("he", "en"):
            # The game speaks Hebrew or English: take the likelier of the two and listen again.
            probs = dict(info.all_language_probs or [])
            language = "en" if probs.get("en", 0) > probs.get("he", 0) else "he"
            segments, info = self._model_for(language).transcribe(samples, language=language, **opts)
        text = " ".join(s.text.strip() for s in segments).strip()
        if _hallucinated(text, prompt):
            text = ""
        return {"text": text, "language": info.language, "probability": round(info.language_probability, 3),
                "duration_s": round(info.duration, 2), "took_s": round(time.perf_counter() - t0, 3)}


# Whisper sometimes "hears" subtitle credits in noise, or repeats its prompt back.
_CREDITS = ("כתוביות", "תורגם", "subtitles", "amara.org", "thanks for watching", "thank you for watching")


def _hallucinated(text: str, prompt: str | None) -> bool:
    t = " ".join(text.lower().split())
    if not t:
        return False
    if any(c in t for c in _CREDITS) and len(t) < 80:
        return True
    p = " ".join((prompt or "").lower().split())
    return bool(p) and len(t) > 12 and (t in p)
