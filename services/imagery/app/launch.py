"""Start the Dask scheduler and the API together in one container.

Painting on the processor runs at a lower priority than everything else on the machine,
so the voices and the storyteller never wait for a picture."""
import os
import subprocess
import sys

if os.environ.get("IMAGERY_DEVICE", "auto").lower() != "cuda":
    try:
        os.nice(int(os.environ.get("IMAGERY_NICE", "10")))
    except (AttributeError, OSError, ValueError):
        pass

sched = subprocess.Popen(["dask", "scheduler", "--host", "0.0.0.0", "--port", "8786",
                          "--dashboard-address", ":8787"])
serve = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0",
         "--port", os.environ.get("PORT", "9030")]
if os.environ.get("RELOAD") == "1":      # live code (docker-compose.override.yml)
    serve += ["--reload", "--reload-dir", "/srv/app"]
try:
    code = subprocess.call(serve)
finally:
    sched.terminate()
sys.exit(code)
