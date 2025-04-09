# MediaGen/main.py
import os
import uuid
import logging
import asyncio
import subprocess
import httpx
from typing import Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import shutil

# Import generation functions - now import both TTS approaches
from media_generation import generate_visual, generate_audio_local

# Import Google Cloud TTS/STT functions
from tts_stt import generate_audio_google, transcribe_audio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# Set up output directory
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./generated_media")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ensure the static media directory exists (for shared volume)
STATIC_MEDIA_DIR = "/app/static/media"
os.makedirs(STATIC_MEDIA_DIR, exist_ok=True)

# Dictionary to track task status
TASKS_STATUS = {}

# Create FastAPI app
app = FastAPI(title="Media Generation API")

# Mount the output directory as a static files directory
app.mount("/static_media", StaticFiles(directory=OUTPUT_DIR), name="static_media")

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TTSRequest(BaseModel):
    text: str
    voice_model: Optional[str] = None
    speaker: Optional[str] = None
    language: Optional[str] = None
    speed: float = Field(default=1.0, ge=0.5, le=2.0)

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "api": "Media Generation Service",
        "version": "1.0",
        "endpoints": {
            "tts": "/generate/tts",
            "visual": "/generate/visual",
            "transcript": "/generate/transcript",
            "test_endpoints": {
                "test_tts": "/test/audio",
                "test_visual": "/test/video"
            }
        }
    }

@app.get("/health")
def health():
    """Health check endpoint"""
    return JSONResponse({"status": "OK"})

@app.get("/healthz")
def healthz():
    """Alternative health check endpoint"""
    return JSONResponse({"status": "OK"})

@app.get("/debug/files")
def debug_files():
    """Debug endpoint to list files in the output directory"""
    output_dir_files = []
    static_media_files = []
    
    if os.path.exists(OUTPUT_DIR):
        output_dir_files = [
            {
                "name": f,
                "size": os.path.getsize(os.path.join(OUTPUT_DIR, f)),
                "is_readable": os.access(os.path.join(OUTPUT_DIR, f), os.R_OK)
            }
            for f in os.listdir(OUTPUT_DIR)
        ]
    
    if os.path.exists(STATIC_MEDIA_DIR):
        static_media_files = [
            {
                "name": f,
                "size": os.path.getsize(os.path.join(STATIC_MEDIA_DIR, f)),
                "is_readable": os.access(os.path.join(STATIC_MEDIA_DIR, f), os.R_OK)
            }
            for f in os.listdir(STATIC_MEDIA_DIR)
        ]
    
    return {
        "output_dir": {
            "path": OUTPUT_DIR,
            "absolute_path": os.path.abspath(OUTPUT_DIR),
            "exists": os.path.exists(OUTPUT_DIR),
            "files": output_dir_files
        },
        "static_media_dir": {
            "path": STATIC_MEDIA_DIR,
            "absolute_path": os.path.abspath(STATIC_MEDIA_DIR),
            "exists": os.path.exists(STATIC_MEDIA_DIR),
            "files": static_media_files
        }
    }

def get_file_info(file_path):
    """Get information about a file if it exists"""
    if os.path.exists(file_path):
        return {
            "exists": True,
            "size_bytes": os.path.getsize(file_path),
            "absolute_path": os.path.abspath(file_path),
            "is_readable": os.access(file_path, os.R_OK)
        }
    return {"exists": False}

def copy_to_shared_dir(source_path: str) -> Optional[str]:
    """Copy a file to the shared directory for frontend access"""
    if not os.path.exists(source_path):
        logger.error(f"Source file not found: {source_path}")
        return None
    
    try:
        filename = os.path.basename(source_path)
        dest_path = os.path.join(STATIC_MEDIA_DIR, filename)
        
        # Check if source and destination are the same file or would be the same file
        src_dir = os.path.dirname(source_path)
        dst_dir = STATIC_MEDIA_DIR
        
        # Method 1: Check if they're the same directory
        try:
            if os.path.samefile(src_dir, dst_dir):
                logger.info(f"Source and destination directories are the same. No need to copy: {filename}")
                return dest_path  # Return the path, but don't actually copy
        except OSError:
            # If samefile fails (can happen if paths don't exist yet), try path comparison
            if os.path.abspath(src_dir) == os.path.abspath(dst_dir):
                logger.info(f"Source and destination directories are the same (by path). No need to copy: {filename}")
                return dest_path
        
        # If dest file already exists but isn't the same file, remove it first
        if os.path.exists(dest_path) and not os.path.samefile(source_path, dest_path):
            os.remove(dest_path)
        
        # If they're different, perform the copy
        shutil.copy2(source_path, dest_path)
        logger.info(f"Copied file to shared directory: {dest_path}")
        return dest_path
    except FileExistsError:
        # If we get a FileExistsError, it might be the same file
        logger.info(f"File already exists at destination, likely the same file: {dest_path}")
        return dest_path
    except Exception as e:
        logger.error(f"Failed to copy file to shared directory: {str(e)}")
        return None

# Define a dependency for extracting the audio file
async def get_audio_file(
    file: UploadFile = File(None),
    audio: UploadFile = File(None)
):
    """Get the audio file from various possible form field names."""
    audio_file = file or audio
    if not audio_file:
        raise HTTPException(
            status_code=422, 
            detail="No audio file provided. Please upload a file with field name 'file' or 'audio'."
        )
    return audio_file

@app.post("/generate/transcript")
async def generate_transcript(
    audio_file: UploadFile = Depends(get_audio_file),
    webrtc_id: str = Form(None, description="Optional WebRTC session ID"),
    voice_model: str = Form(None, description="Optional TTS voice model"),
    tts_speed: float = Form(1.0, description="TTS speech rate")
):
    """
    Generate a transcript from an audio file and process it.
    This implementation uses the old logic from tts_stt.py with Google Cloud services.
    """
    task_id = str(uuid.uuid4()) if not webrtc_id else webrtc_id
    background_tasks = BackgroundTasks()
    
    logger.info(f"Processing audio for task {task_id}, file: {audio_file.filename}")
    
    # Save the uploaded file
    try:
        original_ext = os.path.splitext(audio_file.filename)[1] or ".wav"
        temp_input_path = os.path.join(OUTPUT_DIR, f"{task_id}_input{original_ext}")
        file_content = await audio_file.read()
        
        if not file_content:
            return JSONResponse(
                status_code=400,
                content={"error": "Empty audio file received"}
            )
            
        with open(temp_input_path, "wb") as f:
            f.write(file_content)
            
        logger.info(f"Saved uploaded file to {temp_input_path}")
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing uploaded file: {str(e)}"}
        )
    
    # Convert the audio to a compatible WAV format
    converted_path = os.path.join(OUTPUT_DIR, f"{task_id}_converted.wav")
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-loglevel", "error",
            "-i", temp_input_path,
            "-ar", "48000",
            "-ac", "1",
            "-f", "wav",
            converted_path
        ], check=True)
        logger.info(f"Converted audio to {converted_path}")
    except Exception as e:
        logger.error(f"Error converting audio: {str(e)}")
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        return JSONResponse(
            status_code=500,
            content={"error": f"Error converting audio file: {str(e)}"}
        )
    
    # Transcribe the converted audio
    try:
        transcript = await transcribe_audio(converted_path, 48000)
        TASKS_STATUS[task_id] = "transcript_generated"
        logger.info(f"Generated transcript: {transcript}")
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error transcribing audio: {str(e)}"}
        )
    finally:
        # Clean up temporary files
        for path in [temp_input_path, converted_path]:
            if os.path.exists(path):
                os.remove(path)
    
    # Send transcript to the LLM container
    llm_url = "http://container_llm:9000/generate"
    llm_payload = {"prompt": transcript, "session_id": task_id}
    try:
        logger.info(f"Sending transcript to LLM: {llm_payload}")
        async with httpx.AsyncClient() as client:
            llm_resp = await client.post(llm_url, json=llm_payload, timeout=30.0)
            llm_resp.raise_for_status()
            llm_data = llm_resp.json()
            
            # Extract structured fields from LLM response:
            user_response = llm_data.get("user_response", "")
            voice_response = llm_data.get("voice_response", "")
            stage_decision = llm_data.get("stage_decision", "")
            video_scene = llm_data.get("video_scene", "")
            debug_info = llm_data.get("debug_info", "")
            
            logger.info(f"Received LLM response: {llm_data}")
    except Exception as e:
        logger.error(f"Error contacting LLM: {str(e)}")
        user_response = ""
        voice_response = f"Error contacting LLM: {e}"
        stage_decision = ""
        video_scene = "A figure stands in the darkness."  # Default scene
        debug_info = f"Error contacting LLM: {e}"
    
    # Generate TTS audio for the voice_response using Google TTS
    tts_path = os.path.join(OUTPUT_DIR, f"{task_id}_tts.wav")
    try:
        # Generate TTS with Google
        await generate_audio_google(str(voice_response), tts_path)
        logger.info(f"Generated TTS audio at {tts_path}")
        
        # Copy to shared directory
        copy_to_shared_dir(tts_path)
    except Exception as e:
        logger.error(f"TTS generation error: {str(e)}")
        tts_path = None
        voice_response += f" [TTS generation error: {e}]"
    
    # Start video generation in background to avoid blocking response
    video_path = os.path.join(OUTPUT_DIR, f"{task_id}_visual.mp4")
    
    # Set status for video generation
    TASKS_STATUS[f"{task_id}_video"] = "generating"
    
    # Add video generation task to background
    background_tasks.add_task(
        generate_video_and_update_status,
        task_id, 
        video_scene, 
        video_path
    )
    
    # Always provide multiple URL options to maximize compatibility
    tts_url_options = {
        "url1": f"/get/audio/{task_id}",
        "url2": f"/download/tts/{task_id}",
        "url3": f"/static_media/{task_id}_tts.wav",
        "direct": f"/media/{task_id}_tts.wav"
    }
    
    video_url_options = {
        "url1": f"/get/visual/{task_id}",
        "url2": f"/download/{task_id}/visual",
        "url3": f"/static_media/{task_id}_visual.mp4",
        "direct": f"/media/{task_id}_visual.mp4"
    }
    
    # Return the combined response with video status
    response_payload = {
        "task_id": task_id,
        "transcript": transcript,
        "llm_user_response": user_response,
        "llm_voice_response": voice_response,
        "llm_stage_decision": stage_decision,
        "llm_video_scene": video_scene,
        "debug_info": debug_info,
        "tts_audio_url": tts_url_options["url1"],  # Primary URL
        "tts_audio_urls": tts_url_options,         # All options
        "video_status": "generating", 
        "video_url": video_url_options["url1"],    # Primary URL
        "video_urls": video_url_options            # All options
    }
    
    logger.info(f"Returning response for task {task_id}")
    return JSONResponse(content=response_payload, background=background_tasks)

@app.get("/list/voices")
async def list_voices_endpoint():
    """Endpoint to list all available voices"""
    try:
        # Fix the syntax error: voices = []] -> voices = []
        voices = []
        return {"success": True, "voices": voices}
    except Exception as e:
        logger.error(f"Error listing voices: {e}")
        return {"success": False, "error": str(e)}

@app.get("/get/audio/{task_id}")
def get_audio(task_id: str):
    """Serve TTS audio files with proper headers"""
    # Check for the file in output directory
    tts_path = os.path.join(OUTPUT_DIR, f"{task_id}_tts.wav")
    
    # Log file info for debugging
    file_info = get_file_info(tts_path)
    logger.info(f"Serving audio for task {task_id}: {file_info}")
    
    if not file_info["exists"]:
        # Try alternative path
        alt_path = os.path.join(OUTPUT_DIR, f"{task_id}_audio.wav")
        alt_info = get_file_info(alt_path)
        
        if alt_info["exists"]:
            tts_path = alt_path
            file_info = alt_info
            logger.info(f"Found alternative audio path: {tts_path}")
        else:
            raise HTTPException(status_code=404, detail="Audio file not found")
    
    # Set appropriate headers for streaming
    headers = {
        "Cache-Control": "public, max-age=3600",
        "Access-Control-Allow-Origin": "*",
        "Accept-Ranges": "bytes"
    }
    
    # Try to copy to shared directory, but don't fail if it doesn't work
    try:
        copy_result = copy_to_shared_dir(tts_path)
        if copy_result:
            logger.info(f"Successfully copied audio to shared directory: {copy_result}")
        else:
            logger.warning("Could not copy to shared directory, serving directly from output dir")
    except Exception as e:
        logger.warning(f"Error copying to shared directory: {e}")
    
    # Return the file with appropriate headers
    return FileResponse(
        tts_path, 
        media_type="audio/wav", 
        filename=f"{task_id}_audio.wav",
        headers=headers
    )

@app.head("/get/audio/{task_id}")
def head_audio(task_id: str):
    """HEAD endpoint for audio files (same as GET but without content)"""
    # Check all possible paths
    paths_to_check = [
        os.path.join(OUTPUT_DIR, f"{task_id}_tts.wav"),
        os.path.join(OUTPUT_DIR, f"{task_id}_audio.wav")
    ]
    
    for tts_path in paths_to_check:
        if os.path.exists(tts_path):
            file_size = os.path.getsize(tts_path)
            headers = {
                "Content-Length": str(file_size),
                "Cache-Control": "public, max-age=3600",
                "Access-Control-Allow-Origin": "*",
                "Accept-Ranges": "bytes",
                "Content-Type": "audio/wav"
            }
            return Response(headers=headers)
    
    raise HTTPException(status_code=404, detail="Audio file not found")

@app.get("/get/visual/{task_id}")
def get_visual(task_id: str):
    """Serve visual media files with proper headers"""
    visual_path = os.path.join(OUTPUT_DIR, f"{task_id}_visual.mp4")
    
    # Log file info for debugging
    file_info = get_file_info(visual_path)
    logger.info(f"Serving visual for task {task_id}: {file_info}")
    
    if not file_info["exists"]:
        raise HTTPException(status_code=404, detail="Visual file not found")
    
    # Set appropriate headers for streaming video
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "public, max-age=3600",
        "Access-Control-Allow-Origin": "*"
    }
    
    # Try to copy to shared directory to ensure frontend access
    copy_to_shared_dir(visual_path)
    
    # Return the file with appropriate headers
    return FileResponse(
        visual_path, 
        media_type="video/mp4", 
        filename=f"{task_id}_visual.mp4",
        headers=headers
    )

@app.get("/download/{task_id}/visual")
def download_visual(task_id: str):
    """Alternative endpoint for downloading visual files"""
    visual_path = os.path.join(OUTPUT_DIR, f"{task_id}_visual.mp4")
    
    # Log file info
    file_info = get_file_info(visual_path)
    logger.info(f"Serving visual (download) for task {task_id}: {file_info}")
    
    if not file_info["exists"]:
        raise HTTPException(status_code=404, detail="Visual file not found")
    
    # Set appropriate headers
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "public, max-age=3600",
        "Access-Control-Allow-Origin": "*"
    }
    
    # Copy to shared directory
    copy_to_shared_dir(visual_path)
    
    # Return file
    return FileResponse(
        visual_path, 
        media_type="video/mp4", 
        filename=f"{task_id}_visual.mp4",
        headers=headers
    )


@app.get("/debug/dirs")
def debug_directories():
    """Debug endpoint to check directory permissions and structure"""
    import stat
    
    def get_dir_info(path):
        try:
            exists = os.path.exists(path)
            if not exists:
                return {"exists": False}
                
            st = os.stat(path)
            mode = stat.filemode(st.st_mode)
            writable = os.access(path, os.W_OK)
            readable = os.access(path, os.R_OK)
            
            try:
                files = os.listdir(path)
                files_info = [{
                    "name": f,
                    "size": os.path.getsize(os.path.join(path, f)),
                    "readable": os.access(os.path.join(path, f), os.R_OK),
                    "writable": os.access(os.path.join(path, f), os.W_OK)
                } for f in files[:10]]  # List up to 10 files
            except Exception as e:
                files_info = {"error": str(e)}
                
            return {
                "exists": True,
                "path": path,
                "mode": mode,
                "writable": writable,
                "readable": readable,
                "files": files_info
            }
        except Exception as e:
            return {"error": str(e)}
    
    # Check various directories
    return {
        "OUTPUT_DIR": get_dir_info(OUTPUT_DIR),
        "STATIC_MEDIA_DIR": get_dir_info(STATIC_MEDIA_DIR),
        "current_dir": get_dir_info("."),
        "absolute_paths": {
            "OUTPUT_DIR": os.path.abspath(OUTPUT_DIR),
            "STATIC_MEDIA_DIR": os.path.abspath(STATIC_MEDIA_DIR)
        }
    }

@app.get("/get/visual_alternative/{task_id}")
def get_visual_alternative(task_id: str):
    """Alternative endpoint for serving visual files"""
    visual_path = os.path.join(OUTPUT_DIR, f"{task_id}_visual.mp4")
    
    # Log file info for debugging
    file_info = get_file_info(visual_path)
    logger.info(f"Serving visual (alt) for task {task_id}: {file_info}")
    
    if not file_info["exists"]:
        raise HTTPException(status_code=404, detail="Visual file not found")
    
    # Set appropriate headers
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "public, max-age=3600",
        "Access-Control-Allow-Origin": "*"
    }
    
    # Try to copy to shared directory to ensure frontend access
    copy_to_shared_dir(visual_path)
    
    # Return the file
    return FileResponse(
        visual_path, 
        media_type="video/mp4", 
        filename=f"{task_id}_visual.mp4",
        headers=headers
    )

@app.get("/download/tts/{task_id}")
def download_tts(task_id: str):
    """Endpoint for downloading TTS audio"""
    tts_path = os.path.join(OUTPUT_DIR, f"{task_id}_tts.wav")
    
    # Log file info
    file_info = get_file_info(tts_path)
    logger.info(f"Serving TTS audio (download) for task {task_id}: {file_info}")
    
    if not file_info["exists"]:
        raise HTTPException(status_code=404, detail="TTS audio file not found")
    
    # Set headers
    headers = {
        "Cache-Control": "public, max-age=3600",
        "Access-Control-Allow-Origin": "*"
    }
    
    # Copy to shared directory
    copy_to_shared_dir(tts_path)
    
    # Return the file
    return FileResponse(
        tts_path, 
        media_type="audio/wav", 
        filename=f"{task_id}_tts.wav",
        headers=headers
    )

@app.get("/get/audio_alternative/{task_id}")
def get_audio_alternative(task_id: str):
    """Alternative endpoint for audio retrieval"""
    # Try multiple paths
    paths_to_check = [
        os.path.join(OUTPUT_DIR, f"{task_id}_tts.wav"),
        os.path.join(OUTPUT_DIR, f"{task_id}_audio.wav")
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            # Log file info for debugging
            file_info = get_file_info(path)
            logger.info(f"Serving audio for task {task_id}: {file_info}")
            
            # Add CORS and caching headers
            headers = {
                'Cache-Control': 'public, max-age=3600',
                'Access-Control-Allow-Origin': '*',
                'Accept-Ranges': 'bytes'
            }
            
            # Try to copy to shared directory to ensure frontend access
            copy_to_shared_dir(path)
            
            # Return the file
            return FileResponse(
                path, 
                media_type="audio/wav",
                filename=f"{task_id}_audio.wav",
                headers=headers
            )
    
    # Return 404 if no file found
    raise HTTPException(status_code=404, detail="Audio file not found")

@app.get("/media/{filename}")
def serve_media_file_direct(filename: str):
    """Serve media files directly with appropriate headers"""
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    # Log file info for debugging
    file_info = get_file_info(file_path)
    logger.info(f"Serving file {filename}: {file_info}")
    
    if not file_info["exists"]:
        raise HTTPException(status_code=404, detail=f"File {filename} not found")
    
    # Determine content type
    if filename.endswith(".mp4"):
        media_type = "video/mp4"
    elif filename.endswith(".wav"):
        media_type = "audio/wav"
    else:
        media_type = "application/octet-stream"
    
    # Enable byte range requests for streaming video
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "public, max-age=3600",
        "Access-Control-Allow-Origin": "*"
    }
    
    # Try to copy to shared directory to ensure frontend access
    copy_to_shared_dir(file_path)
    
    # Return the file
    return FileResponse(
        file_path, 
        media_type=media_type,
        headers=headers
    )

@app.get("/audio_status/{task_id}")
async def get_audio_status(task_id: str):
    """Check the status of audio generation for a given task and provide URLs."""
    status = TASKS_STATUS.get(task_id, "unknown")
    audio_path = os.path.join(OUTPUT_DIR, f"{task_id}_tts.wav")
    print(f"DEBUG: Generated TTS at absolute path: {os.path.abspath(audio_path)}")
    alt_path = os.path.join(OUTPUT_DIR, f"{task_id}_audio.wav")
    
    # Get file info
    file_info = get_file_info(audio_path)
    if not file_info["exists"]:
        file_info = get_file_info(alt_path)
        if file_info["exists"]:
            audio_path = alt_path
    
    # If the file exists but status doesn't show completion, update it
    if file_info["exists"] and status not in ["audio_completed", "completed"]:
        TASKS_STATUS[task_id] = "audio_completed"
        status = "audio_completed"
        
        # Copy to shared directory to ensure frontend access
        copy_to_shared_dir(audio_path)
    
    # Multiple URL options for maximum compatibility
    audio_urls = {
        "url1": f"/get/audio/{task_id}",
        "url2": f"/get/audio_alternative/{task_id}",
        "url3": f"/media/{os.path.basename(audio_path)}",
    }
    
    return JSONResponse({
        "task_id": task_id,
        "status": status,
        "file_info": file_info,
        "audio_url": audio_urls["url1"],  # Primary URL
        "audio_urls": audio_urls          # All options
    })

@app.get("/video_status/{task_id}")
async def get_video_status(task_id: str):
    """Check the status of video generation for a given task and provide URLs."""
    status = TASKS_STATUS.get(f"{task_id}_video", TASKS_STATUS.get(task_id, "unknown"))
    video_path = os.path.join(OUTPUT_DIR, f"{task_id}_visual.mp4")
    
    # Get detailed file info
    file_info = get_file_info(video_path)
    
    # If the file exists but status doesn't show completion, update it
    if file_info["exists"] and status not in ["visual_completed", "completed"]:
        TASKS_STATUS[f"{task_id}_video"] = "completed"
        status = "completed"
        
        # Copy to shared directory to ensure frontend access
        copy_to_shared_dir(video_path)
    
    # Multiple URL options for maximum compatibility
    video_urls = {
        "url1": f"/get/visual/{task_id}",
        "url2": f"/download/{task_id}/visual",
        "url3": f"/static_media/{task_id}_visual.mp4",
        "direct": f"/media/{os.path.basename(video_path)}",
    }
    
    return JSONResponse({
        "task_id": task_id,
        "status": status,
        "file_info": file_info,
        "video_url": video_urls["url1"],  # Primary URL
        "video_urls": video_urls          # All options
    })

@app.get("/ensure_file_access/{task_id}")
def ensure_file_access(task_id: str):
    """Ensure that generated files are copied to the shared volume"""
    # Define source and destination paths
    source_files = [
        f"{task_id}_tts.wav",
        f"{task_id}_visual.mp4",
        f"{task_id}_audio.wav",
        f"{task_id}_transcript.txt"
    ]
    
    results = {}
    
    # Copy each file if it exists
    for filename in source_files:
        source_path = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(source_path):
            dest_path = copy_to_shared_dir(source_path)
            if dest_path:
                results[filename] = {
                    "copied": True,
                    "source": source_path,
                    "destination": dest_path,
                    "size": os.path.getsize(dest_path)
                }
            else:
                               results[filename] = {
                    "copied": False,
                    "error": "Failed to copy file"
                }
        else:
            results[filename] = {
                "copied": False,
                "reason": "Source file not found"
            }
    
    return {
        "task_id": task_id,
        "file_results": results
    }

async def generate_visual_task(task_id: str, instruction: str):
    """Task to generate visual content and update its status"""
    try:
        TASKS_STATUS[task_id] = "visual_started"
        visual_path = os.path.join(OUTPUT_DIR, f"{task_id}_visual.mp4")
        
        # Generate the visual
        await generate_visual(instruction, visual_path)
        
        # Copy to shared directory to ensure frontend access
        copy_to_shared_dir(visual_path)
        
        # Update status
        TASKS_STATUS[task_id] = "visual_completed"
        logger.info(f"Visual generation completed for task {task_id}")
    except Exception as e:
        logger.error(f"Visual generation error: {str(e)}")
        TASKS_STATUS[task_id] = f"visual_error: {str(e)}"

async def generate_audio_task(task_id: str, instruction: str, tts_options: Optional[Dict] = None):
    """Task to generate audio content and update its status"""
    try:
        TASKS_STATUS[task_id] = "audio_started"
        
        # Use default file path format
        audio_path = os.path.join(OUTPUT_DIR, f"{task_id}_tts.wav")
        
        # Use local TTS with options if provided
        if tts_options is None:
            tts_options = {}
            
        # Generate the audio
        await generate_audio_local(
            text=instruction, 
            output_path=audio_path,
            **tts_options
        )
        
        # Copy to shared directory to ensure frontend access
        copy_to_shared_dir(audio_path)
        
        # Update status
        TASKS_STATUS[task_id] = "audio_completed"
        logger.info(f"Audio generation completed for task {task_id}")
        return audio_path
    except Exception as e:
        logger.error(f"Audio generation error: {str(e)}")
        TASKS_STATUS[task_id] = f"audio_error: {str(e)}"
        raise

async def generate_video_and_update_status(task_id: str, video_scene: str, video_path: str):
    """Background task to generate video and update status."""
    try:
        # Generate the video
        await generate_visual(video_scene, video_path)
        
        # Copy to shared directory to ensure frontend access
        copy_to_shared_dir(video_path)
        
        # Update status when done
        TASKS_STATUS[f"{task_id}_video"] = "completed"
        logger.info(f"Video generation completed for task {task_id}")
    except Exception as e:
        logger.error(f"Error generating video for task {task_id}: {e}")
        TASKS_STATUS[f"{task_id}_video"] = f"error: {str(e)}"

@app.post("/generate/tts")
async def generate_tts(request: TTSRequest):
    """Generate text-to-speech audio with advanced options"""
    task_id = f"tts_{uuid.uuid4().hex}"
    
    # Extract TTS options from request
    tts_options = {
        "voice_model": request.voice_model,
        "speaker": request.speaker,
        "language": request.language,
        "speed": request.speed
    }
    
    # Start generation task
    try:
        await generate_audio_task(task_id, request.text, tts_options)
        
        # Try to ensure the file is accessible
        await asyncio.sleep(0.5)  # Brief pause to let file system catch up
        await asyncio.to_thread(ensure_file_access, task_id)
        
        # Return URLs for accessing the generated audio
        return {
            "success": True,
            "task_id": task_id,
            "status": TASKS_STATUS.get(task_id, "unknown"),
            "audio_url": f"/get/audio/{task_id}",
            "audio_urls": {
                "url1": f"/get/audio/{task_id}",
                "url2": f"/get/audio_alternative/{task_id}",
                "url3": f"/media/{task_id}_tts.wav",
            }
        }
    except Exception as e:
        logger.error(f"Error in TTS generation: {e}")
        return {"success": False, "error": str(e)}

@app.post("/generate/visual")
async def generate_visual_endpoint(background_tasks: BackgroundTasks, video_scene: str = Form(...)):
    """Generate video visualization based on scene description"""
    task_id = f"vid_{uuid.uuid4().hex}"
    video_path = os.path.join(OUTPUT_DIR, f"{task_id}_visual.mp4")
    
    # Start generation as a background task
    background_tasks.add_task(generate_video_and_update_status, task_id, video_scene, video_path)
    
    return {
        "task_id": task_id,
        "status": "started",
        "message": "Video generation has started in the background",
        "status_url": f"/video_status/{task_id}",
        "video_urls": {
            "url1": f"/get/visual/{task_id}",
            "url2": f"/download/{task_id}/visual",
            "url3": f"/static_media/{task_id}_visual.mp4",
        }
    }

@app.get("/test/audio")
async def test_audio_generation():
    """Test endpoint to generate a sample TTS audio"""
    task_id = f"test_{uuid.uuid4().hex[:6]}"
    output_path = os.path.join(OUTPUT_DIR, f"{task_id}_tts.wav")
    
    try:
        # Try with an English voice model
        tts_options = {
            "voice_model": "tts_models/en/ljspeech/tacotron2-DDC",
            "speed": 1.0
        }
        
        # Generate test audio
        await generate_audio_local(
            "This is a test of the local text to speech system with multiple voices.", 
            output_path,
            **tts_options
        )
        
        # Copy to shared directory to ensure frontend access
        copy_to_shared_dir(output_path)
        
        # Multiple URL options for maximum compatibility
        return {
            "success": True,
            "task_id": task_id,
            "audio_path": output_path,
            "audio_urls": {
                "url1": f"/get/audio/{task_id}",
                "url2": f"/get/audio_alternative/{task_id}",
                "url3": f"/media/{task_id}_tts.wav",
            }
        }
    except Exception as e:
        logger.error(f"Error in test audio generation: {e}")
        return {"success": False, "error": str(e)}

@app.get("/test/google_audio")
async def test_google_audio_generation():
    """Test endpoint to generate a sample Google TTS audio"""
    task_id = f"test_g_{uuid.uuid4().hex[:6]}"
    output_path = os.path.join(OUTPUT_DIR, f"{task_id}_tts.wav")
    
    try:
        # Use Google TTS
        await generate_audio_google(
            "This is a test of the Google Cloud text to speech system.", 
            output_path
        )
        
        # Copy to shared directory
        copy_to_shared_dir(output_path)
        
        return {
            "success": True,
            "task_id": task_id,
            "audio_path": output_path,
            "audio_urls": {
                "url1": f"/get/audio/{task_id}",
                "url2": f"/get/audio_alternative/{task_id}",
                "url3": f"/media/{task_id}_tts.wav",
            }
        }
    except Exception as e:
        logger.error(f"Error in Google test audio generation: {e}")
        return {"success": False, "error": str(e)}

@app.get("/test/video")
async def test_video_generation():
    """Test endpoint to generate a sample video"""
    task_id = f"test_{uuid.uuid4().hex[:6]}"
    output_path = os.path.join(OUTPUT_DIR, f"{task_id}_visual.mp4")
    
    try:
        # Generate test video
        await generate_visual("A mysterious figure emerges from the shadows", output_path)
        
        # Copy to shared directory to ensure frontend access
        copy_to_shared_dir(output_path)
        
        # Multiple URL options for maximum compatibility
        return {
            "success": True,
            "task_id": task_id,
            "video_path": output_path,
            "video_urls": {
                "url1": f"/get/visual/{task_id}",
                "url2": f"/download/{task_id}/visual",
                "url3": f"/static_media/{task_id}_visual.mp4",
            }
        }
    except Exception as e:
        logger.error(f"Error in test video generation: {e}")
        return {"success": False, "error": str(e)}

@app.get("/list_files")
def list_generated_files():
    """List all files in the generated_media directory with details"""
    try:
        files = os.listdir(OUTPUT_DIR)
        detailed_files = []
        
        for file in files:
            file_path = os.path.join(OUTPUT_DIR, file)
            file_info = get_file_info(file_path)
            detailed_files.append({
                "filename": file,
                "info": file_info
            })
            
        return {"files": detailed_files}
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return {"error": str(e)}

# ----------------- Main -----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9001)