#!/usr/bin/env python3
"""
Pre-download and cache the text-to-video diffusion model during container build.
This eliminates the first-request delay in production.
"""
import os
import torch
from diffusers import DiffusionPipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_preloader")

# Set cache directory
os.environ["HF_HOME"] = "/app/model_cache"
os.makedirs("/app/model_cache", exist_ok=True)

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Preloading models using device: {device}")

def preload_text_to_video_model():
    """Preload the text-to-video model during build time"""
    logger.info("Pre-downloading text-to-video diffusion model...")
    
    # Use the same model ID and configuration as in your application
    model_id = "damo-vilab/text-to-video-ms-1.7b"
    
    # Just download the model files without fully loading to memory
    # This will cache the files in HF_HOME
    DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
        cache_dir=os.environ.get("HF_HOME")
    )
    
    logger.info("Text-to-video model pre-downloaded successfully!")

if __name__ == "__main__":
    preload_text_to_video_model()
    logger.info("Model preloading complete!")