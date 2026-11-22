"""Start the Dask scheduler and the API together in one container."""
import os
import subprocess
import sys

sched = subprocess.Popen(["dask", "scheduler", "--host", "0.0.0.0", "--port", "8786",
                          "--dashboard-address", ":8787"])
try:
    code = subprocess.call([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0",
                            "--port", os.environ.get("PORT", "9030")])
finally:
    sched.terminate()
sys.exit(code)
