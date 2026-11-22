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
    r = client.post("/stt", files={"audio": ("a.webm", b"xx", "audio/webm")}, data={"lang": "he", "prompt": "פיה"})
    assert r.json()["text"] == "את פיה ביער"
    assert seen["language"] == "he" and seen["vad_filter"] and seen["initial_prompt"] == "פיה"
    main.stt.state = "waiting"
