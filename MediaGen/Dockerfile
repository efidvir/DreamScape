# MediaGen/Dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Set environment variable to avoid timezone prompt during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    python3-opencv \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories with proper permissions
RUN mkdir -p /app/model_cache /app/generated_media /app/static/media && \
    chmod -R 777 /app/generated_media /app/static/media

# Set environment variables
ENV HF_HOME=/app/model_cache
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV DIFFUSERS_CACHE=/app/model_cache
ENV OUTPUT_DIR=/app/generated_media
ENV PIP_PREFER_BINARY=1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# ✅ Install compatible torch and torchaudio explicitly from PyTorch index
RUN pip install --no-cache-dir \
    torch==2.0.0 \
    torchaudio==2.0.1 \
    --index-url https://download.pytorch.org/whl/cu117

# Install problematic packages separately first
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    scipy==1.10.1 \
    librosa==0.10.1 \
    numba==0.57.1

# Copy application code
COPY . /app/

# Install lightweight dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    requests \
    python-multipart \
    pillow \
    httpx \
    cryptography \
    pydantic

# Install remaining packages
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 9001

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9001"]
