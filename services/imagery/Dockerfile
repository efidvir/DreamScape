# One image for both roles: the imagery API + Dask scheduler (on the laptop) and the
# Dask GPU worker (on the second machine). Same image means matching Dask versions.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/hf
WORKDIR /srv
RUN pip install torch --index-url https://download.pytorch.org/whl/cu126
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app ./app
ENV PYTHONPATH=/srv
EXPOSE 9030 8786 8787
CMD ["python", "-m", "app.launch"]
