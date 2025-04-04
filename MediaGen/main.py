import os
import uuid
import subprocess
import httpx
import logging
import threading
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request, Form, Depends
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

# Import our media generation functions
from media_generation import generate_visual, generate_audio
from tts_stt import generate_audio_google, transcribe_audio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# Ensure output directory exists
OUTPUT_DIR = "./generated_media"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize FastAPI
app = FastAPI()

# Mount the output directory as a static files directory
app.mount("/static_media", StaticFiles(directory=OUTPUT_DIR), name="static_media")

# Add CORS middleware with comprehensive settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "Content-Type", "Content-Length"],
)

class GenerationRequest(BaseModel):
    text: str

TASKS_STATUS = {}

# ------------------ Utility Functions ------------------
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

# ------------------ Background Task Handlers ------------------
async def generate_audio_task(task_id: str, instruction: str):
    try:
        TASKS_STATUS[task_id] = "audio_started"
        audio_path = os.path.join(OUTPUT_DIR, f"{task_id}_audio.wav")
        await generate_audio_google(instruction, audio_path)
        TASKS_STATUS[task_id] = "audio_completed"
    except Exception as e:
        logger.error(f"Audio generation error: {str(e)}")
        TASKS_STATUS[task_id] = f"audio_error: {str(e)}"

async def generate_visual_task(task_id: str, instruction: str):
    try:
        TASKS_STATUS[task_id] = "visual_started"
        visual_path = os.path.join(OUTPUT_DIR, f"{task_id}_visual.mp4")
        await generate_visual(instruction, visual_path)
        TASKS_STATUS[task_id] = "visual_completed"
    except Exception as e:
        logger.error(f"Visual generation error: {str(e)}")
        TASKS_STATUS[task_id] = f"visual_error: {str(e)}"

async def generate_video_and_update_status(task_id: str, video_scene: str, video_path: str):
    """Background task to generate video and update status."""
    try:
        # Generate the video
        await generate_visual(video_scene, video_path)
        
        # Update status when done
        TASKS_STATUS[f"{task_id}_video"] = "completed"
        logger.info(f"Video generation completed for task {task_id}")
    except Exception as e:
        logger.error(f"Error generating video for task {task_id}: {e}")
        TASKS_STATUS[f"{task_id}_video"] = f"error: {str(e)}"

# --------------------- Endpoints ----------------------
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
    webrtc_id: str = Form(None, description="Optional WebRTC session ID")
):
    """
    Generate a transcript from an audio file and process it.
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
    
    # Convert the audio to a Google-compatible WAV
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
    
    # Generate TTS audio for the voice_response
    tts_path = os.path.join(OUTPUT_DIR, f"{task_id}_tts.wav")
    try:
        await generate_audio_google(str(voice_response), tts_path)
        logger.info(f"Generated TTS audio at {tts_path}")
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

# ----------------- File Access Endpoints -----------------
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
    
    return FileResponse(
        file_path, 
        media_type=media_type,
        headers=headers
    )

@app.get("/download/{task_id}/visual")
def get_visual(task_id: str):
    visual_path = os.path.join(OUTPUT_DIR, f"{task_id}_visual.mp4")
    # Log file info for debugging
    file_info = get_file_info(visual_path)
    logger.info(f"Serving visual for task {task_id}: {file_info}")
    
    if os.path.exists(visual_path):
        headers = {
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600",
            "Access-Control-Allow-Origin": "*"
        }
        return FileResponse(
            visual_path, 
            media_type="video/mp4", 
            filename=f"{task_id}_visual.mp4",
            headers=headers
        )
    raise HTTPException(status_code=404, detail="Visual file not found")

@app.get("/get/visual/{task_id}")
def get_visual_alternative(task_id: str):
    visual_path = os.path.join(OUTPUT_DIR, f"{task_id}_visual.mp4")
    # Log file info for debugging
    file_info = get_file_info(visual_path)
    logger.info(f"Serving visual (alt) for task {task_id}: {file_info}")
    
    if os.path.exists(visual_path):
        headers = {
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600",
            "Access-Control-Allow-Origin": "*"
        }
        return FileResponse(
            visual_path, 
            media_type="video/mp4", 
            filename=f"{task_id}_visual.mp4",
            headers=headers
        )
    raise HTTPException(status_code=404, detail="Visual file not found")

@app.get("/download/tts/{task_id}")
def get_tts_audio(task_id: str):
    tts_path = os.path.join(OUTPUT_DIR, f"{task_id}_tts.wav")
    # Log file info for debugging
    file_info = get_file_info(tts_path)
    logger.info(f"Serving TTS audio for task {task_id}: {file_info}")
    
    if os.path.exists(tts_path):
        headers = {
            "Cache-Control": "public, max-age=3600",
            "Access-Control-Allow-Origin": "*"
        }
        return FileResponse(
            tts_path, 
            media_type="audio/wav", 
            filename=f"{task_id}_tts.wav",
            headers=headers
        )
    raise HTTPException(status_code=404, detail="TTS audio file not found")

@app.get("/get/audio/{task_id}")
def get_audio_alternative(task_id: str):
    tts_path = os.path.join(OUTPUT_DIR, f"{task_id}_tts.wav")
    # Log file info for debugging
    file_info = get_file_info(tts_path)
    logger.info(f"Serving TTS audio (alt) for task {task_id}: {file_info}")
    
    if os.path.exists(tts_path):
        headers = {
            "Cache-Control": "public, max-age=3600",
            "Access-Control-Allow-Origin": "*"
        }
        return FileResponse(
            tts_path, 
            media_type="audio/wav", 
            filename=f"{task_id}_tts.wav",
            headers=headers
        )
    raise HTTPException(status_code=404, detail="TTS audio file not found")

@app.get("/video_status/{task_id}")
async def get_video_status(task_id: str):
    """Check the status of video generation for a given task and provide all URLs."""
    status = TASKS_STATUS.get(f"{task_id}_video", "not_found")
    video_path = os.path.join(OUTPUT_DIR, f"{task_id}_visual.mp4")
    
    # Get detailed file info
    file_info = get_file_info(video_path)
    
    # If the file exists but status doesn't show completion, update it
    if file_info["exists"] and status != "completed":
        TASKS_STATUS[f"{task_id}_video"] = "completed"
        status = "completed"
    
    # Multiple URL options for maximum compatibility
    video_urls = {
        "url1": f"/get/visual/{task_id}",
        "url2": f"/download/{task_id}/visual",
        "url3": f"/static_media/{task_id}_visual.mp4",
        "direct": f"/media/{task_id}_visual.mp4"
    }
    
    return JSONResponse({
        "task_id": task_id,
        "status": status,
        "file_info": file_info,
        "video_url": video_urls["url1"],  # Primary URL 
        "video_urls": video_urls          # All options
    })

# ----------------- Debug and Testing Endpoints -----------------
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

@app.get("/test_video")
async def test_video_generation():
    """Test endpoint to generate a sample video"""
    task_id = f"test_{uuid.uuid4().hex[:6]}"
    output_path = os.path.join(OUTPUT_DIR, f"{task_id}_visual.mp4")
    
    try:
        await generate_visual("A mysterious figure emerges from the shadows", output_path)
        return {
            "success": True,
            "task_id": task_id,
            "video_path": output_path,
            "video_urls": {
                "url1": f"/get/visual/{task_id}",
                "url2": f"/download/{task_id}/visual", 
                "url3": f"/static_media/{task_id}_visual.mp4",
                "direct": f"/media/{task_id}_visual.mp4"
            }
        }
    except Exception as e:
        logger.error(f"Error in test video generation: {e}")
        return {"success": False, "error": str(e)}

@app.get("/test_audio")
async def test_audio_generation():
    """Test endpoint to generate a sample TTS audio"""
    task_id = f"test_{uuid.uuid4().hex[:6]}"
    output_path = os.path.join(OUTPUT_DIR, f"{task_id}_tts.wav")
    
    try:
        await generate_audio_google("This is a test of the text to speech system", output_path)
        return {
            "success": True,
            "task_id": task_id,
            "audio_path": output_path,
            "audio_urls": {
                "url1": f"/get/audio/{task_id}",
                "url2": f"/download/tts/{task_id}",
                "url3": f"/static_media/{task_id}_tts.wav",
                "direct": f"/media/{task_id}_tts.wav"
            }
        }
    except Exception as e:
        logger.error(f"Error in test audio generation: {e}")
        return {"success": False, "error": str(e)}

@app.get("/healthz")
def health():
    return JSONResponse({"status": "OK"})

# ----------------- Main -----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9001)