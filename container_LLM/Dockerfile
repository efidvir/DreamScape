# Use an NVIDIA CUDA base image with cuDNN and runtime libraries.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Install Python, pip, and Git.
RUN apt-get update && apt-get install -y python3 python3-pip git

# Copy requirements file and install dependencies.
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Pre-download the model at build time to cache heavy downloads.
RUN python3 -c "from transformers import pipeline; pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')"

# Copy the FastAPI application code.
COPY app.py /app.py

# Expose port 9000.
EXPOSE 9000

# Run the FastAPI app using Gunicorn with Uvicorn workers.
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:9000", "app:app"]
