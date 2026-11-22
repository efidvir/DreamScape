"""Run the whole dream on any machine, no GPU, no models: fake LLM + fake voice,
real director and real web page. Then open http://localhost:8080
(set NO_PICTURES=1 to see the dream without the stand-in picture painter)

    pip install -r services/web/requirements.txt -r services/director/requirements.txt python-multipart pillow
    python dev/demo.py
"""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(module, port, cwd, env=None):
    e = {**os.environ, **(env or {}), "PYTHONPATH": str(cwd)}
    return subprocess.Popen([PY, "-m", "uvicorn", module, "--host", "127.0.0.1", "--port", str(port),
                             "--log-level", "warning"], cwd=cwd, env=e)


def main():
    data = tempfile.mkdtemp(prefix="dreams-")
    procs = [
        run("fakes:ollama", 11500, ROOT / "dev"),
        run("fakes:voice", 9020, ROOT / "dev"),
        run("fakes:imagery", 9030, ROOT / "dev"),
        run("app.main:app", 9010, ROOT / "services" / "director",
            {"OLLAMA_URL": "http://127.0.0.1:11500", "DATA_DIR": data, "LLM_MODEL": "fake"}),
        run("app.main:app", int(os.environ.get("PORT", "8080")), ROOT / "services" / "web",
            {"DIRECTOR_URL": "http://127.0.0.1:9010", "VOICE_URL": "http://127.0.0.1:9020", "HEARTBEAT_S": "1",
             "IMAGERY_URL": "" if os.environ.get("NO_PICTURES") else "http://127.0.0.1:9030"}),
    ]
    print(f"DreamScape demo on http://localhost:{os.environ.get('PORT', '8080')}  (Ctrl+C to stop)")
    try:
        while all(p.poll() is None for p in procs):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait(timeout=10)


if __name__ == "__main__":
    main()
