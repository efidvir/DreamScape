# MediaGen/media_generation.py
import os
import time
import asyncio
import logging
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
from typing import Optional, Dict, Any
from tts_service import tts_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the cache directory
os.environ["HF_HOME"] = "/app/model_cache"
os.environ["TRANSFORMERS_CACHE"] = "/app/model_cache"
os.environ["DIFFUSERS_CACHE"] = "/app/model_cache"

# Ensure output directory exists
os.makedirs("./generated_media", exist_ok=True)

# Check for GPU availability, but default to CPU for stability
force_cpu = True  # Set to True to force CPU mode for stability
if torch.cuda.is_available() and not force_cpu:
    device = "cuda"
    try:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        logger.info(f"Using GPU: {gpu_name} with {gpu_mem:.2f} GB memory")
    except Exception as e:
        logger.warning(f"Error getting GPU info: {e}")
else:
    device = "cpu"
    logger.info("Using CPU for generation (more stable, but slower)")

# Video generation settings
DEFAULT_DURATION = 8  # seconds
DEFAULT_FPS = 8  # frames per second
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512

# Color palettes for different themes
DARK_PALETTE = {
    'background': (10, 10, 15),     # Very dark blue-gray
    'highlight': (25, 25, 40),      # Slightly lighter blue-gray
    'text': (200, 200, 220),        # Light blue-gray
    'accent': (70, 70, 120)         # Dark purple-blue
}

LIGHT_PALETTE = {
    'background': (240, 240, 245),  # Very light blue-gray
    'highlight': (220, 220, 230),   # Slightly darker blue-gray
    'text': (30, 30, 50),           # Dark blue-gray
    'accent': (100, 100, 180)       # Medium purple-blue
}

async def generate_visual(scene_description: str, output_path: str, duration: int = DEFAULT_DURATION):
    """
    Generate a video visualization based on the scene description.
    Instead of using ML models that are causing issues, creates an artistic 
    visualization with text and graphics.
    
    Args:
        scene_description: Text description from LLM's video_scene field
        output_path: Where to save the generated video
        duration: Length of the video in seconds
    
    Returns:
        str: Path to the generated video file
    """
    try:
        # Determine if it's a dark scene based on keywords
        is_dark_scene = any(word in scene_description.lower() 
                           for word in ['dark', 'night', 'shadow', 'mysterious', 
                                        'noir', 'gloom', 'evil', 'horror'])
        
        # Select appropriate color palette
        palette = DARK_PALETTE if is_dark_scene else LIGHT_PALETTE
        
        # Create an artistic visualization video
        await create_artistic_video(scene_description, output_path, duration, DEFAULT_FPS, palette)
        
        return output_path
    except Exception as e:
        logger.error(f"Error in generate_visual: {e}")
        # Fall back to a simple video if anything fails
        return await generate_fallback_video(scene_description, output_path, duration)

async def create_artistic_video(scene_description: str, output_path: str, duration: int, fps: int, palette: dict):
    """
    Create an artistic video with animated elements based on the scene description.
    
    Args:
        scene_description: Description of the scene
        output_path: Where to save the video
        duration: Video duration in seconds
        fps: Frames per second
        palette: Color palette to use
    """
    width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT
    num_frames = duration * fps
    
    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process the scene description to extract key elements
    keywords = extract_keywords(scene_description)
    
    # Generate frames
    for frame_num in range(num_frames):
        # Create a new frame with gradient background
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create gradient background
        progress = frame_num / num_frames
        create_gradient_background(img, palette, progress)
        
        # Add animated elements based on keywords
        add_animated_elements(img, keywords, frame_num, num_frames, palette)
        
        # Add text overlay with the scene description
        add_text_overlay(img, scene_description, frame_num, num_frames, palette)
        
        # Write frame to video
        video.write(img)
    
    video.release()
    logger.info(f"Created artistic video at {output_path}")

def extract_keywords(text: str):
    """Extract important keywords from the scene description."""
    # List of important visual keywords to look for
    visual_keywords = [
        'figure', 'shadow', 'light', 'dark', 'bright', 'night', 'day',
        'rain', 'fog', 'mist', 'smoke', 'fire', 'water', 'storm',
        'moon', 'sun', 'star', 'cloud', 'sky', 'mountain', 'forest',
        'city', 'street', 'room', 'door', 'window', 'wall', 'floor',
        'eyes', 'face', 'hand', 'body', 'blood', 'weapon', 'gun',
        'knife', 'sword', 'magical', 'mystic', 'ancient', 'modern',
        'futuristic', 'technological', 'natural', 'organic', 'mechanical'
    ]
    
    # Find keywords in the text
    found_keywords = []
    for keyword in visual_keywords:
        if keyword in text.lower():
            found_keywords.append(keyword)
    
    # Add some default keywords if none found
    if not found_keywords:
        found_keywords = ['figure', 'shadow']
    
    return found_keywords

def create_gradient_background(img, palette, progress):
    """Create a gradient background with subtle animation."""
    height, width = img.shape[:2]
    bg_color = palette['background']
    highlight = palette['highlight']
    
    # Base gradient - top to bottom
    for y in range(height):
        # Calculate gradient intensity with some oscillation
        gradient_factor = y / height
        oscillation = 0.05 * np.sin(progress * 2 * np.pi + gradient_factor * np.pi)
        gradient_factor = max(0, min(1, gradient_factor + oscillation))
        
        # Interpolate between background and highlight
        r = int(bg_color[0] + (highlight[0] - bg_color[0]) * gradient_factor)
        g = int(bg_color[1] + (highlight[1] - bg_color[1]) * gradient_factor)
        b = int(bg_color[2] + (highlight[2] - bg_color[2]) * gradient_factor)
        
        # Apply to row
        img[y, :] = (b, g, r)  # OpenCV uses BGR
    
    # Add some soft circular highlight areas that move slowly
    for i in range(3):
        # Calculate position with slow movement
        angle = progress * np.pi * 2 * (0.2 + i * 0.1)
        x_center = int(width * (0.3 + 0.4 * np.sin(angle)))
        y_center = int(height * (0.3 + 0.4 * np.cos(angle)))
        
        # Calculate radius with pulsation
        radius = int(width * 0.2 * (0.8 + 0.2 * np.sin(progress * np.pi * 4 + i)))
        
        # Draw soft radial gradient
        for y in range(max(0, y_center - radius), min(height, y_center + radius)):
            for x in range(max(0, x_center - radius), min(width, x_center + radius)):
                # Calculate distance from center
                dx, dy = x - x_center, y - y_center
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance < radius:
                    # Calculate intensity (stronger near center)
                    intensity = 1 - (distance / radius) ** 2
                    intensity *= 0.15  # Reduce effect strength
                    
                    # Get current color and adjust
                    b, g, r = img[y, x]
                    r = min(255, r + int(intensity * (palette['accent'][0] - r)))
                    g = min(255, g + int(intensity * (palette['accent'][1] - g)))
                    b = min(255, b + int(intensity * (palette['accent'][2] - b)))
                    
                    img[y, x] = (b, g, r)

def add_animated_elements(img, keywords, frame_num, num_frames, palette):
    """Add animated visual elements based on keywords."""
    height, width = img.shape[:2]
    progress = frame_num / num_frames
    
    # Draw different elements based on keywords
    if 'figure' in keywords or 'shadow' in keywords:
        # Draw a mysterious figure silhouette
        figure_height = int(height * 0.6)
        figure_width = int(figure_height * 0.4)
        
        # Position at bottom center with slight movement
        x_pos = width // 2 - figure_width // 2 + int(width * 0.05 * np.sin(progress * np.pi * 2))
        y_pos = height - figure_height // 2
        
        # Draw a simple silhouette - just a dark oval for the head and a trapezoid for the body
        # Head
        head_radius = figure_width // 2
        cv2.circle(img, (x_pos + figure_width // 2, y_pos - figure_height // 2 + head_radius), 
                  head_radius, (0, 0, 0), -1)
        
        # Body - trapezoid
        body_points = np.array([
            [x_pos + figure_width // 3, y_pos - figure_height // 2 + head_radius * 2],  # Top left
            [x_pos + figure_width * 2 // 3, y_pos - figure_height // 2 + head_radius * 2],  # Top right
            [x_pos + figure_width, y_pos + figure_height // 2],  # Bottom right
            [x_pos, y_pos + figure_height // 2]  # Bottom left
        ])
        cv2.fillPoly(img, [body_points], (0, 0, 0))
        
        # Add slight mysterious glow around the figure
        glow_color = palette['accent']
        for i in range(15, 0, -5):
            # Draw glow with decreasing opacity
            alpha = (i / 15) * 0.3
            glow_img = img.copy()
            
            # Expanded head glow
            cv2.circle(glow_img, 
                     (x_pos + figure_width // 2, y_pos - figure_height // 2 + head_radius), 
                     head_radius + i, glow_color, 2)
            
            # Expanded body glow - just use lines for simplicity
            expanded_points = np.array([
                [x_pos + figure_width // 3 - i, y_pos - figure_height // 2 + head_radius * 2 - i],
                [x_pos + figure_width * 2 // 3 + i, y_pos - figure_height // 2 + head_radius * 2 - i],
                [x_pos + figure_width + i, y_pos + figure_height // 2 + i],
                [x_pos - i, y_pos + figure_height // 2 + i]
            ])
            cv2.polylines(glow_img, [expanded_points], True, glow_color, 2)
            
            # Blend with main image
            cv2.addWeighted(img, 1 - alpha, glow_img, alpha, 0, img)
    
    # Add more effects for other keywords as needed
    if 'light' in keywords or 'bright' in keywords:
        # Add some light rays
        num_rays = 5
        for i in range(num_rays):
            angle = (progress * np.pi / 2) + (i * 2 * np.pi / num_rays)
            length = width * 0.7
            start_x = int(width * 0.5)
            start_y = int(height * 0.3)
            end_x = int(start_x + length * np.cos(angle))
            end_y = int(start_y + length * np.sin(angle))
            
            # Draw with decreasing opacity
            for j in range(10, 0, -2):
                alpha = (j / 10) * 0.3
                ray_img = img.copy()
                cv2.line(ray_img, (start_x, start_y), (end_x, end_y), palette['accent'], j)
                cv2.addWeighted(img, 1 - alpha, ray_img, alpha, 0, img)
    
    if 'fog' in keywords or 'mist' in keywords or 'smoke' in keywords:
        # Add some fog/mist effect
        for _ in range(20):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            radius = np.random.randint(10, 50)
            opacity = np.random.uniform(0.05, 0.2)
            
            fog_img = img.copy()
            cv2.circle(fog_img, (x, y), radius, (255, 255, 255), -1)
            cv2.addWeighted(img, 1 - opacity, fog_img, opacity, 0, img)

def add_text_overlay(img, scene_description, frame_num, num_frames, palette):
    """Add text overlay with the scene description."""
    height, width = img.shape[:2]
    progress = frame_num / num_frames
    text_color = palette['text']
    
    # Split text into lines
    words = scene_description.split()
    lines = []
    current_line = ""
    
    for word in words:
        if len(current_line + word) < 40:  # Limit line length
            current_line += word + " "
        else:
            lines.append(current_line)
            current_line = word + " "
    
    if current_line:
        lines.append(current_line)
    
    # Animate opacity based on frame progress
    opacity = 1.0
    if progress < 0.2:  # Fade in
        opacity = progress / 0.2
    elif progress > 0.8:  # Fade out
        opacity = (1.0 - progress) / 0.2
    opacity = max(0, min(1, opacity))
    
    # Create transparent overlay for text
    text_overlay = np.zeros_like(img, dtype=np.uint8)
    
    # Position text at the bottom
    y_pos = height - 20 - len(lines) * 30
    
    # Add each line of text
    for line in lines:
        # Get text size with OpenCV (more compatible than PIL)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(line, font, 0.7, 2)[0]
        
        # Center the text horizontally
        x_pos = (width - text_size[0]) // 2
        
        # Draw text with a subtle shadow for better readability
        cv2.putText(text_overlay, line, (x_pos + 2, y_pos + 2), 
                   font, 0.7, (0, 0, 0), 2)
        cv2.putText(text_overlay, line, (x_pos, y_pos), 
                   font, 0.7, text_color, 2)
        
        y_pos += 30
    
    # Blend text overlay with main image
    cv2.addWeighted(img, 1, text_overlay, opacity, 0, img)

async def generate_fallback_video(scene_description: str, output_path: str, duration: int = DEFAULT_DURATION):
    """
    Generate a simple fallback video with text overlay.
    Creates a dark background with text of the scene description.
    """
    width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT
    fps = DEFAULT_FPS
    
    # Create a very simple video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create a dark background
    background = np.zeros((height, width, 3), dtype=np.uint8)
    background[:, :] = (15, 15, 25)  # Dark blue-ish background
    
    # Add text with the scene description
    text = scene_description
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (200, 200, 200)  # Light gray
    
    # Split text into multiple lines for better visibility
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        if len(current_line + word) < 40:  # Limit line length
            current_line += word + " "
        else:
            lines.append(current_line)
            current_line = word + " "
    
    if current_line:
        lines.append(current_line)
    
    # Position text in the center of the frame
    y_pos = height // 2 - (len(lines) * 30) // 2
    
    # Create a frame with the text
    frame = background.copy()
    for line in lines:
        text_size = cv2.getTextSize(line, font, 0.7, 1)[0]
        x_pos = (width - text_size[0]) // 2
        cv2.putText(frame, line, (x_pos, y_pos), font, 0.7, text_color, 1)
        y_pos += 30
    
    # Write the frame multiple times to create the video
    num_frames = duration * fps
    for _ in range(num_frames):
        video.write(frame)
    
    video.release()
    logger.info(f"Generated basic fallback video at {output_path}")
    return output_path

def enhance_prompt(scene_description: str) -> str:
    """
    Enhance the scene description to make it more suitable for image generation.
    
    Args:
        scene_description: Original scene description from LLM
        
    Returns:
        str: Enhanced prompt for better image generation
    """
    # Add qualifiers to make the output more visually appealing
    base_prompt = scene_description.strip()
    
    # Check if it's the dark character scenario
    if "darkness" in base_prompt.lower() or "shadow" in base_prompt.lower():
        enhancers = [
            "cinematic lighting, dark atmosphere, volumetric fog, dramatic shadows",
            "atmospheric, cinematic, high detail, moody lighting",
            "film noir style, mysterious, high contrast, shadowy"
        ]
        
        # Combine with a random enhancer
        import random
        enhancer = random.choice(enhancers)
        enhanced_prompt = f"{base_prompt}, {enhancer}, trending on artstation, masterpiece, detailed"
        
        return enhanced_prompt
    
    # For other scenes
    return f"{base_prompt}, cinematic quality, detailed, atmospheric lighting, trending on artstation, masterpiece"

async def generate_audio(prompt: str, output_path: str, **kwargs) -> str:
    """
    Generate audio based on the provided prompt.
    This is a fallback function that creates a simple audio file.
    
    Args:
        prompt: Text prompt to generate audio from
        output_path: Path where the audio should be saved
        **kwargs: Additional parameters for audio generation
        
    Returns:
        str: Path to the generated audio file
    """
    # This is a simple placeholder that creates a silence file
    import numpy as np
    from scipy.io import wavfile
    
    # Generate 3 seconds of silence at 44.1kHz
    sample_rate = 44100
    duration = 3  # seconds
    samples = np.zeros(int(sample_rate * duration))
    
    # Save as WAV file
    wavfile.write(output_path, sample_rate, samples.astype(np.float32))
    
    return output_path

async def generate_audio_local(
    text: str, 
    output_path: str,
    voice_model: Optional[str] = None,
    speaker: Optional[str] = None,
    language: Optional[str] = None,
    speed: float = 1.0
) -> str:
    """
    Generate audio using local TTS service with various options.
    
    Args:
        text: Text to synthesize
        output_path: Path where the audio should be saved
        voice_model: Specific TTS model to use
        speaker: Speaker ID for multi-speaker models
        language: Language code for multilingual models
        speed: Speech rate (1.0 is normal)
        
    Returns:
        str: Path to the generated audio file
    """
    try:
        logger.info(f"Generating audio with local TTS for: '{text[:50]}...'")
        
        # Synthesize speech using the local TTS service
        tts_service.synthesize(
            text=text,
            output_path=output_path,
            voice_model=voice_model,
            speaker=speaker,
            language=language,
            speed=speed
        )
        
        return output_path
    except Exception as e:
        logger.error(f"Local TTS generation failed: {e}")
        raise

async def generate_transcript_from_audio(audio_path: str) -> str:
    """
    Generate a transcript from an audio file.
    This is a placeholder that should be replaced with proper STT functionality.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        str: Transcribed text
    """
    try:
        from tts_stt import transcribe_audio
        
        # Use the Google Cloud speech-to-text service from tts_stt.py
        transcript = await transcribe_audio(audio_path)
        return transcript
        
    except ImportError:
        logger.warning("Could not import transcribe_audio from tts_stt.py")
        return "Placeholder transcript. Speech-to-text functionality not available."
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return f"Error transcribing audio: {str(e)}"