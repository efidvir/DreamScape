import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
import cv2
import numpy as np
import os
import soundfile as sf

# Load Stable Diffusion Turbo model for image generation
device = "cuda" if torch.cuda.is_available() else "cpu"
image_model_id = "stabilityai/sd-turbo"
image_pipe = StableDiffusionPipeline.from_pretrained(
    image_model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
image_pipe = image_pipe.to(device)

# Lightweight TTS using Silero
tts_model, _ = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language='en',
    speaker='v3_en',
    trust_repo=True
)
tts_model.to(device)

# Function to generate audio
async def generate_audio(instruction: str, output_path: str):
    try:
        audio = tts_model.apply_tts(
            text=instruction,
            speaker='en_0',
            sample_rate=48000
        )
        sf.write(output_path, audio, 48000)
        print(f"✅ Lightweight AI Audio generated: {output_path}")
    except Exception as e:
        print(f"❌ Audio generation failed: {e}")
        raise

# Function to generate looping visuals using AI generative model
async def generate_visual(instruction: str, output_path: str):
    try:
        frames = []
        for step in np.linspace(0.5, 1.0, 24):  # 24 frames for 1 second
            result = image_pipe(instruction, guidance_scale=step)
            image = result.images[0]
            frame = np.array(image.convert('RGB'))[:, :, ::-1]  # PIL to OpenCV BGR
            frames.append(frame)

        height, width, _ = frames[0].shape
        video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (width, height))

        for _ in range(5):  # Repeat the loop for ~5 seconds
            for frame in frames:
                video.write(frame)

        video.release()
        print(f"✅ AI-generated looping video saved: {output_path}")
    except Exception as e:
        print(f"❌ Visual generation failed: {e}")
        raise

# Example test
if __name__ == "__main__":
    import asyncio
    instruction_text = "A colorful butterfly in a sunny garden."
    asyncio.run(generate_audio(instruction_text, "test_audio.wav"))
    asyncio.run(generate_visual(instruction_text, "test_video.mp4"))
