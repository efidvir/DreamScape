# mediagen/Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set environment variables to be non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, pip, and dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    gcc \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Set up directories
RUN mkdir -p /app/generated_media
RUN mkdir -p /app/model_cache

# Copy requirements first to utilize Docker cache
COPY requirements.txt .

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy the preload script and run it
COPY preload_models.py /app/preload_models.py
RUN python3 /app/preload_models.py

# Copy application code
COPY . .

# Set environment variables
ENV OUTPUT_DIR=/app/generated_media
ENV HF_HOME=/app/model_cache
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV PORT=9001
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE 9001

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9001"]