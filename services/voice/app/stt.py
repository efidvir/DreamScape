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
        from faster_whisper import WhisperModel
        try:
            m = WhisperModel(name, device=self.device, compute_type=self.compute_type,
                             download_root=self.root)
            self.active_device = self.device
        except Exception as e:  # noqa: BLE001 - no GPU: keep working on CPU
            log.warning("loading %s on %s failed (%s); falling back to CPU", name, self.device, e)
            m = WhisperModel(name, device="cpu", compute_type="int8", download_root=self.root)
            self.active_device = "cpu"
        return m

    def load(self) -> None:
        try:
            self.state, self.detail = "loading", f"loading {self.model_he}"
            self.models[self.model_he] = self._load_one(self.model_he)
            # Warm-up pass so the first real utterance is fast.
            import numpy as np
            list(self.models[self.model_he].transcribe(np.zeros(16000, dtype=np.float32),
                                                       language="he", beam_size=1)[0])
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
            raise RuntimeError(f"speech-to-text not ready ({self.state})")
        t0 = time.perf_counter()
        language = lang if lang in ("he", "en") else None
        segments, info = self._model_for(language).transcribe(
            io.BytesIO(audio),
            language=language,
            beam_size=self.beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
            condition_on_previous_text=False,
            without_timestamps=True,
            initial_prompt=prompt or None,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return {"text": text, "language": info.language, "probability": round(info.language_probability, 3),
                "duration_s": round(info.duration, 2), "took_s": round(time.perf_counter() - t0, 3)}
