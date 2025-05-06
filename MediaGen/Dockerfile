# MediaGen/Dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Set environment variable to avoid timezone prompt during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    espeak-ng \
    netcat \
    iproute2 \
    net-tools \
    procps \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories with proper permissions
RUN mkdir -p /app/model_cache /app/generated_media /app/static/media /app/dask_logs && \
    chmod -R 777 /app/generated_media /app/static/media /app/dask_logs

# Set environment variables
ENV HF_HOME=/app/model_cache
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV DIFFUSERS_CACHE=/app/model_cache
ENV OUTPUT_DIR=/app/generated_media
ENV PIP_PREFER_BINARY=1
ENV DASK_SCHEDULER_PORT=8786
ENV DASK_DASHBOARD_PORT=8787

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch packages from the PyTorch index
RUN pip install --no-cache-dir \
    torch==2.0.0 \
    torchaudio==2.0.1 \
    --index-url https://download.pytorch.org/whl/cu117

# Install key dependencies in the right order with specific versions
# This prevents incompatible package versions
RUN pip install --no-cache-dir numpy==1.23.5 && \
    pip install --no-cache-dir scipy==1.10.1 && \
    pip install --no-cache-dir scikit-learn==1.2.2 && \
    pip install --no-cache-dir numba==0.56.4 && \
    pip install --no-cache-dir librosa==0.10.0 && \
    pip install --no-cache-dir pydantic==1.10.8 && \
    pip install --no-cache-dir diffusers==0.12.0 && \
    pip install --no-cache-dir transformers==4.27.0

# Install Dask and required dependencies
RUN pip install --no-cache-dir \
    "dask==2024.1.1" \
    "distributed==2024.1.1" \
    "bokeh>=2.4.3" \
    "msgpack>=1.0.4"

# Install FastAPI, Uvicorn and related dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    python-multipart \
    aiofiles

# Install Google Cloud dependencies
RUN pip install --no-cache-dir \
    google-api-core \
    google-auth \
    google-cloud \
    google-cloud-texttospeech \
    google-cloud-speech

# Copy requirements.txt for additional dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt || echo "Some requirements could not be installed"

# Copy application code
COPY . /app/

# Expose the ports for the application and Dask
EXPOSE 9001 8786 8787

# Use JSON array format for CMD to properly handle signals
CMD ["sh", "-c", "dask-scheduler --host 0.0.0.0 --port 8786 --dashboard-address 0.0.0.0:8787 & sleep 3 && uvicorn main:app --host 0.0.0.0 --port 9001"]