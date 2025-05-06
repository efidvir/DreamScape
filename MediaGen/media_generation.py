# MediaGen/media_generation.py
import os
import logging
import torch
import numpy as np
from io import BytesIO
from typing import Optional, Dict, Any
import dask
from dask.distributed import Client
import time
import threading
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set the cache directory
os.environ["HF_HOME"] = "/app/model_cache"
os.environ["TRANSFORMERS_CACHE"] = "/app/model_cache"
os.environ["DIFFUSERS_CACHE"] = "/app/model_cache"

# Ensure output directory exists
os.makedirs("./generated_media", exist_ok=True)

# Initialize Dask client
dask_client = None
dask_client_lock = threading.Lock()
worker_monitor_thread = None

def monitor_dask_workers():
    """
    Periodically monitors and logs Dask worker status.
    This runs in a background thread.
    """
    global dask_client
    
    logger.info("Starting Dask worker monitoring thread")
    
    while True:
        try:
            if dask_client is not None:
                workers = dask_client.scheduler_info()["workers"]
                if workers:
                    logger.info(f"Dask cluster status: {len(workers)} workers connected")
                    for worker_address, worker_info in workers.items():
                        worker_name = worker_info.get("name", "unnamed")
                        worker_status = worker_info.get("status", "unknown")
                        worker_resources = worker_info.get("resources", {})
                        
                        # Log GPU information if available
                        gpu_info = ""
                        if "GPU" in worker_resources:
                            gpu_info = f", GPU: {worker_resources['GPU']}"
                        
                        logger.info(f"  Worker: {worker_name} ({worker_address}) - Status: {worker_status}{gpu_info}")
                else:
                    logger.warning("No Dask workers currently connected")
        except Exception as e:
            logger.error(f"Error monitoring Dask workers: {e}")
        
        # Sleep for 60 seconds before checking again
        time.sleep(60)

def initialize_dask_client():
    """
    Initialize the Dask client connecting to the local scheduler.
    The scheduler runs in this container and workers connect to it from remote machines.
    """
    global dask_client, worker_monitor_thread
    
    with dask_client_lock:
        if dask_client is not None:
            return dask_client
        
        try:
            # Get scheduler port from environment or use default
            scheduler_port = int(os.environ.get("DASK_SCHEDULER_PORT", 8786))
            
            # Connect to the scheduler running in the same container
            logger.info(f"Connecting to Dask scheduler at localhost:{scheduler_port}")
            client = Client(f"localhost:{scheduler_port}")
            logger.info(f"✅ Connected to Dask scheduler at {client.scheduler.address}")
            
            # Wait for workers to connect
            max_wait_time = 5  # seconds
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                workers = client.scheduler_info()['workers']
                if workers:
                    logger.info(f"✅ Connected to {len(workers)} Dask workers: {list(workers.keys())}")
                    # Log worker details
                    for addr, info in workers.items():
                        worker_name = info.get("name", "unnamed")
                        logger.info(f"  Worker: {worker_name} ({addr}) - Status: {info['status']}")
                    break
                logger.info("Waiting for workers to connect...")
                time.sleep(1)
            
            if not workers:
                logger.warning("⚠️ No workers connected yet. Tasks will be processed locally until workers connect.")
            
            # Start the monitoring thread if not already running
            if worker_monitor_thread is None or not worker_monitor_thread.is_alive():
                worker_monitor_thread = threading.Thread(target=monitor_dask_workers, daemon=True)
                worker_monitor_thread.start()
                logger.info("Started background worker monitoring thread")
            
            dask_client = client
            return dask_client
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Dask client: {e}")
            logger.info("Falling back to local execution")
            return None

# Determine device (prefer GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device for local generation: {device}")

# Create a placeholder function for text-to-video that doesn't depend on diffusers
async def generate_visual(scene_description: str, output_path: str, duration: int = 8):
    """
    Generate a video using a text-to-video diffusion model.
    Uses Dask for distributed computing if available.
    
    Args:
        scene_description: Text prompt for generation
        output_path: Where to save the generated video
        duration: Target duration in seconds (not directly used by the model)

    Returns:
        str: Path to generated video file
    """
    try:
        logger.info(f"Generating video for: {scene_description}")
        
        # Make sure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Try to get a Dask client
        client = initialize_dask_client()
        
        if client and len(client.scheduler_info()['workers']) > 0:
            # Use Dask distributed computing
            logger.info("🚀 Using Dask for distributed video generation")
            
            # Log worker information
            workers = client.scheduler_info()['workers']
            worker_addresses = list(workers.keys())
            logger.info(f"Available workers: {len(worker_addresses)}")
            for addr in worker_addresses:
                worker_name = workers[addr].get("name", "unnamed")
                worker_status = workers[addr].get("status", "unknown")
                logger.info(f"  - {worker_name} ({addr}): {worker_status}")
            
            # Submit the generation job to Dask
            logger.info(f"Submitting job to Dask cluster...")
            future = client.submit(
                _generate_video_frames,
                scene_description=scene_description,
                output_path=output_path,
                duration=duration
            )
            
            # Wait for the result
            logger.info(f"Waiting for job to complete... (this may take a while)")
            result = future.result()
            logger.info(f"✅ Dask video generation completed: {result}")
            return result
        else:
            # Fall back to local generation
            logger.warning("⚠️ No Dask workers available, falling back to local generation")
            return await _generate_video_local(scene_description, output_path, duration)
        
    except Exception as e:
        logger.error(f"❌ Error generating video: {e}")
        # Fall back to a simple video if the model fails
        try:
            import subprocess
            
            # Generate a simple colored video with text
            logger.info("Generating simple fallback video...")
            ffmpeg_simple_cmd = [
                'ffmpeg', '-y',
                '-f', 'lavfi',
                '-i', f'color=c=darkblue:s=640x480:d=10',
                '-vf', f"drawtext=text='{scene_description}':fontcolor=white:fontsize=24:x=(w-text_w)/2:y=(h-text_h)/2:line_spacing=10",
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                output_path
            ]
            subprocess.run(ffmpeg_simple_cmd, check=True)
            logger.info(f"✅ Fallback video generated at {output_path}")
            return output_path
        except:
            logger.error(f"❌ Even fallback video generation failed")
            raise

def _generate_video_frames(scene_description: str, output_path: str, duration: int = 8):
    """
    Worker function to generate video frames using diffusers model.
    This runs on the Dask worker.
    
    Args:
        scene_description: Text prompt for generation
        output_path: Where to save the generated video
        duration: Target duration in seconds
        
    Returns:
        str: Path to the generated video
    """
    import os
    import torch
    import logging
    from diffusers import DiffusionPipeline
    from diffusers.utils import export_to_video
    import subprocess
    
    # Configure worker-specific logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("dask_worker")
    
    try:
        # Make sure output directory exists on the worker
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get worker information for logging
        try:
            import socket
            hostname = socket.gethostname()
            logger.info(f"Running on worker: {hostname}")
        except:
            pass
        
        # Load the text-to-video pipeline (using a lightweight model)
        logger.info("Loading diffusion model on worker...")
        pipe = DiffusionPipeline.from_pretrained(
            "damo-vilab/text-to-video-ms-1.7b", 
            torch_dtype=torch.float16, 
            variant="fp16"
        )
        
        # Move to GPU if available on the worker
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Worker using device: {device}")
        pipe = pipe.to(device)
        
        # Add dreamlike quality to the prompt
        dreamlike_prompt = f"dreamlike, ethereal, soft focus: {scene_description}"
        
        # Generate the video frames - using minimal settings to save computation
        logger.info("Starting video generation with the model...")
        frames = pipe(
            dreamlike_prompt,
            num_inference_steps=20,  # Reduce for faster generation
            num_frames=24,           # About 2-3 seconds at normal speed
            height=256,              # Low resolution
            width=256,
            guidance_scale=7.0,      # Lower guidance for more creative/dream-like results
        ).frames
        
        logger.info(f"Generated {len(frames)} frames, now exporting to video...")
        
        # Create temp path for frames
        temp_video_path = os.path.join(os.path.dirname(output_path), "temp_video.mp4")
        
        # Export frames to video
        export_to_video(frames, temp_video_path)
        
        # Use ffmpeg to slow down the video to target duration and add some dreamy effects
        target_duration = 10  # 10 seconds as requested
        
        logger.info("Applying post-processing effects with ffmpeg...")
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', temp_video_path,
            '-vf', f"setpts=4*PTS,gblur=sigma=1.5",  # Slow down and add blur for dreamy effect
            '-c:v', 'libx264',
            '-preset', 'fast',  # Speed up encoding
            '-crf', '28',       # Lower quality for smaller file
            '-pix_fmt', 'yuv420p',
            output_path
        ]
        
        subprocess.run(ffmpeg_cmd, check=True)
        logger.info(f"✅ Dream-like video generated at {output_path}")
        
        # Clean up temporary file
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
            logger.info("Temporary files cleaned up")
        
        return output_path
        
    except Exception as e:
        logger.error(f"❌ Worker error generating video: {e}")
        raise

async def _generate_video_local(scene_description: str, output_path: str, duration: int = 8):
    """
    Local fallback for video generation when Dask is not available.
    
    Args:
        scene_description: Text prompt for generation
        output_path: Where to save the generated video
        duration: Target duration in seconds
        
    Returns:
        str: Path to the generated video
    """
    # Import diffusers for text-to-video generation
    from diffusers import DiffusionPipeline
    from diffusers.utils import export_to_video
    import torch
    
    logger.info("Starting local video generation process...")
    
    # Load the text-to-video pipeline (using a lightweight model)
    logger.info("Loading diffusion model locally...")
    pipe = DiffusionPipeline.from_pretrained(
        "damo-vilab/text-to-video-ms-1.7b", 
        torch_dtype=torch.float16, 
        variant="fp16"
    )
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    
    # Add dreamlike quality to the prompt
    dreamlike_prompt = f"dreamlike, ethereal, soft focus: {scene_description}"
    
    # Generate the video frames - using minimal settings to save computation
    logger.info("Starting video generation with the model...")
    frames = pipe(
        dreamlike_prompt,
        num_inference_steps=20,  # Reduce for faster generation
        num_frames=24,           # About 2-3 seconds at normal speed
        height=256,              # Low resolution
        width=256,
        guidance_scale=7.0,      # Lower guidance for more creative/dream-like results
    ).frames
    
    logger.info(f"Generated {len(frames)} frames, now exporting to video...")
    
    # Create temp path for frames
    temp_video_path = os.path.join(os.path.dirname(output_path), "temp_video.mp4")
    
    # Export frames to video
    export_to_video(frames, temp_video_path)
    
    # Use ffmpeg to slow down the video to target duration and add some dreamy effects
    import subprocess
    
    logger.info("Applying post-processing effects with ffmpeg...")
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-i', temp_video_path,
        '-vf', f"setpts=4*PTS,gblur=sigma=1.5",  # Slow down and add blur for dreamy effect
        '-c:v', 'libx264',
        '-preset', 'fast',  # Speed up encoding
        '-crf', '28',       # Lower quality for smaller file
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    
    subprocess.run(ffmpeg_cmd, check=True)
    logger.info(f"✅ Dream-like video generated at {output_path}")
    
    # Clean up temporary file
    if os.path.exists(temp_video_path):
        os.unlink(temp_video_path)
        logger.info("Temporary files cleaned up")
    
    return output_path

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
    
    logger.info(f"Generating fallback audio for: '{prompt[:50]}...'")
    
    # Generate 3 seconds of silence at 44.1kHz
    sample_rate = 44100
    duration = 3  # seconds
    samples = np.zeros(int(sample_rate * duration))
    
    # Save as WAV file
    wavfile.write(output_path, sample_rate, samples.astype(np.float32))
    logger.info(f"Generated silent audio at {output_path}")
    
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
    Uses Dask for distributed processing if available.
    
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
        logger.info(f"Generating audio for: '{text[:50]}...'")
        
        # Try to get a Dask client
        client = initialize_dask_client()
        
        if client and len(client.scheduler_info()['workers']) > 0:
            # Use Dask distributed computing
            logger.info("🚀 Using Dask for distributed audio generation")
            
            # Log worker information
            workers = client.scheduler_info()['workers']
            worker_addresses = list(workers.keys())
            logger.info(f"Available workers: {len(worker_addresses)}")
            for addr in worker_addresses:
                worker_name = workers[addr].get("name", "unnamed")
                worker_status = workers[addr].get("status", "unknown")
                logger.info(f"  - {worker_name} ({addr}): {worker_status}")
            
            # Submit the generation job to Dask
            logger.info(f"Submitting TTS job to Dask cluster...")
            future = client.submit(
                _generate_audio_worker,
                text=text,
                output_path=output_path,
                voice_model=voice_model,
                speaker=speaker,
                language=language,
                speed=speed
            )
            
            # Wait for the result
            logger.info(f"Waiting for TTS job to complete...")
            result = future.result()
            logger.info(f"✅ Dask audio generation completed: {result}")
            return result
        else:
            # Fall back to local generation
            logger.warning("⚠️ No Dask workers available, falling back to local audio generation")
            from tts_stt import synthesize_speech
            
            # Generate the audio file
            logger.info("Starting local TTS synthesis...")
            result = await synthesize_speech(
                text=text,
                output_path=output_path,
                voice_model=voice_model,
                speaker=speaker,
                language=language,
                speed=speed
            )
            logger.info(f"✅ Local audio generation completed: {result}")
            return result
    except Exception as e:
        logger.error(f"❌ Audio generation failed: {e}")
        raise

def _generate_audio_worker(
    text: str, 
    output_path: str,
    voice_model: Optional[str] = None,
    speaker: Optional[str] = None,
    language: Optional[str] = None,
    speed: float = 1.0
) -> str:
    """
    Worker function to generate audio using TTS.
    This runs on the Dask worker.
    
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
    import os
    import logging
    import asyncio
    
    # Configure worker-specific logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("dask_worker")
    
    try:
        # Get worker information for logging
        try:
            import socket
            hostname = socket.gethostname()
            logger.info(f"Running TTS on worker: {hostname}")
        except:
            pass
            
        # Make sure output directory exists on the worker
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Import the TTS function from the worker's environment
        logger.info(f"Synthesizing speech for text: '{text[:50]}...'")
        from tts_stt import synthesize_speech
        
        # Use asyncio to run the async function in the sync worker context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            synthesize_speech(
                text=text,
                output_path=output_path,
                voice_model=voice_model,
                speaker=speaker,
                language=language,
                speed=speed
            )
        )
        loop.close()
        
        logger.info(f"✅ Speech synthesis completed on worker: {result}")
        return result
    except Exception as e:
        logger.error(f"❌ Worker error generating audio: {e}")
        raise

async def generate_transcript_from_audio(audio_path: str) -> str:
    """
    Generate a transcript from an audio file.
    Uses Dask for distributed processing if available.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        str: Transcribed text
    """
    try:
        logger.info(f"Transcribing audio from: {audio_path}")
        
        # Try to get a Dask client
        client = initialize_dask_client()
        
        if client and len(client.scheduler_info()['workers']) > 0:
            # Use Dask distributed computing
            logger.info("🚀 Using Dask for distributed audio transcription")
            
            # Log worker information
            workers = client.scheduler_info()['workers']
            worker_addresses = list(workers.keys())
            logger.info(f"Available workers: {len(worker_addresses)}")
            for addr in worker_addresses:
                worker_name = workers[addr].get("name", "unnamed")
                worker_status = workers[addr].get("status", "unknown")
                logger.info(f"  - {worker_name} ({addr}): {worker_status}")
            
            # Submit the transcription job to Dask
            logger.info(f"Submitting transcription job to Dask cluster...")
            future = client.submit(_transcribe_audio_worker, audio_path=audio_path)
            
            # Wait for the result
            logger.info(f"Waiting for transcription job to complete...")
            result = future.result()
            logger.info(f"✅ Dask audio transcription completed")
            return result
        else:
            # Fall back to local transcription
            logger.warning("⚠️ No Dask workers available, falling back to local transcription")
            from tts_stt import transcribe_audio
            
            # Use the speech-to-text service from tts_stt.py
            logger.info("Starting local audio transcription...")
            transcript = await transcribe_audio(audio_path)
            logger.info(f"✅ Local transcription completed: {transcript[:50]}...")
            return transcript
        
    except ImportError:
        logger.warning("❌ Could not import transcribe_audio from tts_stt.py")
        return "Placeholder transcript. Speech-to-text functionality not available."
    except Exception as e:
        logger.error(f"❌ Error transcribing audio: {e}")
        return f"Error transcribing audio: {str(e)}"

def _transcribe_audio_worker(audio_path: str) -> str:
    """
    Worker function to transcribe audio.
    This runs on the Dask worker.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        str: Transcribed text
    """
    import logging
    import asyncio
    
    # Configure worker-specific logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("dask_worker")
    
    try:
        # Get worker information for logging
        try:
            import socket
            hostname = socket.gethostname()
            logger.info(f"Running transcription on worker: {hostname}")
        except:
            pass
        
        # Import the STT function from the worker's environment
        logger.info(f"Starting transcription for audio file: {audio_path}")
        from tts_stt import transcribe_audio
        
        # Use asyncio to run the async function in the sync worker context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(transcribe_audio(audio_path))
        loop.close()
        
        logger.info(f"✅ Transcription completed on worker: {result[:50]}...")
        return result
    except Exception as e:
        logger.error(f"❌ Worker error transcribing audio: {e}")
        return f"Error transcribing audio: {str(e)}"