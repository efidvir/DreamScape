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

# Install PyTorch packages from the PyTorch index
RUN pip install --no-cache-dir \
    torch==2.0.0 \
    torchaudio==2.0.1 \
    --index-url https://download.pytorch.org/whl/cu117

# Install key dependencies in the right order
RUN pip install --no-cache-dir numpy==1.23.5 && \
    pip install --no-cache-dir numba==0.56.4 && \
    pip install --no-cache-dir scipy==1.10.1 && \
    pip install --no-cache-dir librosa==0.10.0 && \
    pip install --no-cache-dir pydantic==1.10.8 && \
    pip install --no-cache-dir diffusers==0.12.0 && \
    pip install --no-cache-dir transformers==4.27.0

# Copy requirements and install remaining packages
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Expose the port
EXPOSE 9001

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9001"]