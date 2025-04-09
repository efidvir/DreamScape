# MediaGen/media_generation.py
import os
import logging
import torch
import numpy as np
from io import BytesIO
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the cache directory
os.environ["HF_HOME"] = "/app/model_cache"
os.environ["TRANSFORMERS_CACHE"] = "/app/model_cache"
os.environ["DIFFUSERS_CACHE"] = "/app/model_cache"

# Ensure output directory exists
os.makedirs("./generated_media", exist_ok=True)

# Determine device (prefer GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device for generation: {device}")

# Create a placeholder function for text-to-video that doesn't depend on diffusers
async def generate_visual(scene_description: str, output_path: str, duration: int = 8):
    """
    Generate a video placeholder.
    
    Args:
        scene_description: Text prompt for generation
        output_path: Where to save the generated video
        duration: Target duration in seconds (passed to model as guidance)

    Returns:
        str: Path to generated video file
    """
    try:
        logger.info(f"Generating placeholder video for: {scene_description}")
        
        # Generate a simple placeholder video
        import subprocess
        import tempfile
        
        # Create a text file with the scene description
        text_file = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
        text_file.write(scene_description.encode('utf-8'))
        text_file.close()
        
        # Generate a plain color video with the text as an overlay
        try:
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-f', 'lavfi', 
                '-i', f'color=c=gray:s=640x480:d={duration}',
                '-vf', f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:textfile='{text_file.name}':fontcolor=white:fontsize=24:x=(w-text_w)/2:y=(h-text_h)/2:line_spacing=10",
                '-c:v', 'libx264', 
                '-pix_fmt', 'yuv420p',
                output_path
            ]
            subprocess.run(ffmpeg_cmd, check=True)
            
            logger.info(f"Video placeholder generated at {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating video with ffmpeg: {e}")
            
            # Fallback: Generate an extremely simple black video
            ffmpeg_simple_cmd = [
                'ffmpeg', '-y',
                '-f', 'lavfi',
                '-i', f'color=c=black:s=640x480:d={duration}',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                output_path
            ]
            subprocess.run(ffmpeg_simple_cmd, check=True)
            logger.info(f"Simple fallback video generated at {output_path}")
            
        finally:
            # Clean up the temporary text file
            if os.path.exists(text_file.name):
                os.unlink(text_file.name)
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error generating video: {e}")
        raise

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
        
        # Use the TTS from the tts_stt module
        from tts_stt import synthesize_speech
        
        # Generate the audio file
        return await synthesize_speech(
            text=text,
            output_path=output_path,
            voice_model=voice_model,
            speaker=speaker,
            language=language,
            speed=speed
        )
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