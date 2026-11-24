# One Dockerfile for both roles: the imagery API + Dask scheduler (on the laptop) and the
# Dask GPU worker (on the second machine). The same file means matching Dask versions.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/hf MODELS_DIR=/models
WORKDIR /srv
# The laptop paints on its processor, so its image takes the small processor-only PyTorch
# (about 5 GB less than the CUDA one); the GPU worker builds with TORCH=cu126.
ARG TORCH=cpu
RUN pip install torch --index-url https://download.pytorch.org/whl/${TORCH}
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app ./app
ENV PYTHONPATH=/srv
EXPOSE 9030 8786 8787
CMD ["python", "-m", "app.launch"]
