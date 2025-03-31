FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 git ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Increase Hugging Face Hub download timeout to 300 seconds.
ENV HF_HUB_DOWNLOAD_TIMEOUT=300

# Pre-download models at build time
RUN python -c "import torch; from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained('stabilityai/sd-turbo', local_files_only=False, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to('cpu')"
RUN python -c "import torch; model, utils = torch.hub.load('snakers4/silero-models', 'silero_tts', language='en', speaker='v3_en', trust_repo=True); model.apply_tts(text='test', speaker='en_0')"

# Copy the rest of the application code
COPY . .

RUN mkdir -p generated_media && chmod -R 777 generated_media

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9001"]
