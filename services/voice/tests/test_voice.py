import io
import types

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app import main


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "_load_all", lambda: None)   # no model downloads in tests
    with TestClient(main.app) as c:
        yield c


def test_not_ready_answers_503(client):
    assert client.get("/healthz").json()["status"] == "loading"
    assert client.post("/tts", json={"text": "שלום", "lang": "he"}).status_code == 503
    assert client.post("/stt", files={"audio": ("a.webm", b"xx", "audio/webm")}).status_code == 503


def test_tts_writes_a_wav_with_the_requested_voice(client):
    calls = {}

    def synth(text, lang, speed, style):
        calls.update(text=text, lang=lang, style=style)
        return np.sin(np.linspace(0, 200, 4410)).astype(np.float32), 44100

    main.tts.engine = types.SimpleNamespace(synthesize=synth)
    main.tts.styles = {"neutral": "N", "female": "F"}
    main.tts.state = "ready"
    r = client.post("/tts", json={"text": "hello", "lang": "en", "voice": "female"})
    assert r.status_code == 200 and calls == {"text": "hello", "lang": "en", "style": "F"}
    data, sr = sf.read(io.BytesIO(r.content))
    assert sr == 44100 and len(data) == 4410
    main.tts.state = "waiting"


def test_stt_passes_language_and_joins_segments(client):
    seen = {}

    class Model:
        def transcribe(self, audio, **kw):
            seen.update(kw)
            seg = [types.SimpleNamespace(text=" את פיה"), types.SimpleNamespace(text=" ביער ")]
            return iter(seg), types.SimpleNamespace(language="he", language_probability=0.98, duration=2.0)

    main.stt.models = {main.stt.model_he: Model()}
    main.stt.state = "ready"
    wav = io.BytesIO()
    sf.write(wav, np.sin(np.linspace(0, 900, 16000)).astype(np.float32) * 0.3, 16000, format="WAV")
    r = client.post("/stt", files={"audio": ("a.wav", wav.getvalue(), "audio/wav")}, data={"lang": "he", "prompt": "פיה"})
    assert r.json()["text"] == "את פיה ביער"
    assert seen["language"] == "he" and seen["vad_filter"] and seen["initial_prompt"] == "פיה"
    main.stt.state = "waiting"


def test_stt_falls_back_to_cpu_when_cuda_fails_on_first_run(monkeypatch):
    import faster_whisper
    from app.stt import SpeechToText
    made = []

    class FakeModel:
        def __init__(self, name, device, compute_type, download_root):
            self.device = device
            made.append((device, compute_type))

        def transcribe(self, audio, **kw):
            if self.device == "cuda":
                raise RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")
            return iter([]), None

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
    stt = SpeechToText()
    stt.device = "cuda"
    stt.load()
    assert stt.state == "ready" and stt.active_device == "cpu"
    assert made == [("cuda", stt.compute_type), ("cpu", "int8")]


def test_undecodable_audio_is_a_clean_422(client):
    main.stt.models = {main.stt.model_he: object()}
    main.stt.state = "ready"
    r = client.post("/stt", files={"audio": ("a.webm", b"\x1aE\xdf\xa3not-really-webm", "audio/webm")})
    assert r.status_code == 422
    main.stt.state = "waiting"


def test_a_caller_who_went_away_is_not_spoken_for(client, monkeypatch):
    calls = []
    main.tts.engine = types.SimpleNamespace(synthesize=lambda *a, **k: calls.append(a) or (np.zeros(10, np.float32), 16000))
    main.tts.styles = {"neutral": "N"}
    main.tts.state = "ready"

    async def gone(request):
        return True

    monkeypatch.setattr(main, "_gone", gone)
    r = client.post("/tts", json={"text": "hello", "lang": "en"})
    assert r.status_code == 499 and calls == []
    main.tts.state = "waiting"
