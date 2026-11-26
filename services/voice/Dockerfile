FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt requirements-gpu.txt ./
RUN pip install -r requirements.txt
# The voice can also run on the GPU (TTS_DEVICE=cuda): ONNX Runtime's CUDA build takes the
# place of the CPU one BlueTTS installs, with the CUDA 13 libraries it loads. Whisper keeps
# its CUDA 12 cuBLAS; the two sets live in separate folders.
RUN pip uninstall -y onnxruntime && pip install -r requirements-gpu.txt

# Let CTranslate2 find the pip-installed cuBLAS (ONNX Runtime finds its own libraries).
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib \
    HF_HOME=/models/hf \
    WHISPER_DIR=/models/whisper \
    TTS_DIR=/models/bluev3

COPY app ./app
EXPOSE 9020
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9020"]
