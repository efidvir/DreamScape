# container_LLM/Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set environment variables to be non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, pip, Git and utilities
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install specific versions to ensure compatibility
RUN pip3 install --no-cache-dir \
    transformers==4.37.2 \
    torch==2.1.2 \
    accelerate==0.26.1 \
    safetensors==0.4.1 \
    sentencepiece==0.1.99 \
    protobuf==3.20.3

# Create directories for models
RUN mkdir -p /model_cache /downloaded_models

# Create an empty game_progress.json
RUN echo '{"stages": {"0": {"instructions": "You are in darkness with no memory of your identity. Your goal is to discover your name and origin.", "missing": ["name", "origin"]}}}' > /app/game_progress.json

# Copy application code
COPY app.py /app/

# Expose port 9000
EXPOSE 9000

# Run with appropriate settings
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:9000", "--workers", "1", "--timeout", "300", "--graceful-timeout", "60", "app:app"]