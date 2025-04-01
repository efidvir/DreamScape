import os
import uuid
import subprocess
import httpx
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from media_generation import generate_visual, generate_audio  # if needed for other endpoints
from tts_stt import generate_audio_google, transcribe_audio  # your implementations

# Ensure output directory exists
OUTPUT_DIR = "./generated_media"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI()

class GenerationRequest(BaseModel):
    text: str

TASKS_STATUS = {}

# ------------------ Background Task Handlers ------------------
async def generate_audio_task(task_id: str, instruction: str):
    try:
        TASKS_STATUS[task_id] = "audio_started"
        audio_path = os.path.join(OUTPUT_DIR, f"{task_id}_audio.wav")
        await generate_audio_google(instruction, audio_path)
        TASKS_STATUS[task_id] = "audio_completed"
    except Exception as e:
        TASKS_STATUS[task_id] = f"audio_error: {str(e)}"

async def generate_visual_task(task_id: str, instruction: str):
    try:
        TASKS_STATUS[task_id] = "visual_started"
        visual_path = os.path.join(OUTPUT_DIR, f"{task_id}_visual.mp4")
        await generate_visual(instruction, visual_path)
        TASKS_STATUS[task_id] = "visual_completed"
    except Exception as e:
        TASKS_STATUS[task_id] = f"visual_error: {str(e)}"

# --------------------- Other Endpoints ----------------------
@app.post("/generate/audio")
def generate_audio_endpoint(request: GenerationRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    TASKS_STATUS[task_id] = "queued"
    background_tasks.add_task(generate_audio_task, task_id, request.text)
    return JSONResponse({"task_id": task_id})

@app.post("/generate/visual")
def generate_visual_endpoint(request: GenerationRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    TASKS_STATUS[task_id] = "queued"
    background_tasks.add_task(generate_visual_task, task_id, request.text)
    return JSONResponse({"task_id": task_id})

@app.get("/status/{task_id}")
def check_status(task_id: str):
    status = TASKS_STATUS.get(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task ID not found")
    return JSONResponse({"task_id": task_id, "status": status})

@app.get("/download/{task_id}/audio")
def get_audio(task_id: str):
    audio_path = os.path.join(OUTPUT_DIR, f"{task_id}_audio.wav")
    if os.path.exists(audio_path):
        return FileResponse(audio_path, media_type="audio/wav", filename=f"{task_id}_audio.wav")
    raise HTTPException(status_code=404, detail="Audio file not found")

@app.get("/download/{task_id}/visual")
def get_visual(task_id: str):
    visual_path = os.path.join(OUTPUT_DIR, f"{task_id}_visual.mp4")
    if os.path.exists(visual_path):
        return FileResponse(visual_path, media_type="video/mp4", filename=f"{task_id}_visual.mp4")
    raise HTTPException(status_code=404, detail="Visual file not found")

@app.delete("/cleanup/{task_id}")
def cleanup(task_id: str):
    audio_path = os.path.join(OUTPUT_DIR, f"{task_id}_audio.wav")
    visual_path = os.path.join(OUTPUT_DIR, f"{task_id}_visual.mp4")
    for path in [audio_path, visual_path]:
        if os.path.exists(path):
            os.remove(path)
    TASKS_STATUS.pop(task_id, None)
    return JSONResponse({"task_id": task_id, "status": "cleaned up"})

# ----------------- Transcription, LLM, and TTS Integration -----------------
@app.post("/generate/transcript")
async def generate_transcript(file: UploadFile = File(..., alias="audio")):
    task_id = str(uuid.uuid4())
    
    # Save the uploaded file
    original_ext = os.path.splitext(file.filename)[1] or ".wav"
    temp_input_path = os.path.join(OUTPUT_DIR, f"{task_id}_input{original_ext}")
    file_content = await file.read()
    with open(temp_input_path, "wb") as f:
        f.write(file_content)
    
    # Convert the audio to a Google-compatible WAV (suppress FFmpeg logs)
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
    except Exception as e:
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        return JSONResponse({"task_id": task_id, "transcript": f"Error converting file: {e}"})
    
    # Transcribe the converted audio
    try:
        transcript = await transcribe_audio(converted_path, 48000)
        TASKS_STATUS[task_id] = "transcript_generated"
    finally:
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(converted_path):
            os.remove(converted_path)
    
    # Send transcript to the LLM container
    llm_url = "http://container_llm:9000/generate"
    llm_payload = {"transcript": transcript, "session_id": task_id}
    try:
        async with httpx.AsyncClient() as client:
            llm_resp = await client.post(llm_url, json=llm_payload, timeout=30.0)
            llm_resp.raise_for_status()
            llm_data = llm_resp.json()
            # Expected structured fields:
            user_response = llm_data.get("user_response", "")
            voice_response = llm_data.get("voice_response", "")
            stage_decision = llm_data.get("stage_decision", "")
            video_scene = llm_data.get("video_scene", "")
            debug_info = llm_data.get("debug_info", "")
    except Exception as e:
        # Log raw response if available
        raw_llm_response = ""
        try:
            raw_llm_response = llm_resp.text
        except:
            pass
        user_response = ""
        voice_response = f"Error contacting LLM: {e}. Raw response: {raw_llm_response}"
        stage_decision = ""
        video_scene = ""
        debug_info = f"Error contacting LLM: {e}"
    
    # Log the raw LLM response for debugging.
    print("LLM Raw Response:", llm_data if 'llm_data' in locals() else voice_response)
    
    # Generate TTS audio using the voice_response.
    tts_path = os.path.join(OUTPUT_DIR, f"{task_id}_tts.wav")
    try:
        await generate_audio_google(str(voice_response), tts_path)
    except Exception as e:
        tts_path = None
        voice_response += f" [TTS generation error: {e}]"
    
    # Return a combined response with separated fields.
    response_payload = {
        "task_id": task_id,
        "transcript": transcript,
        "llm_user_response": user_response,
        "llm_voice_response": voice_response,
        "llm_stage_decision": stage_decision,
        "llm_video_scene": video_scene,
        "debug_info": debug_info,
        "tts_audio_url": f"/download/tts/{task_id}" if tts_path and os.path.exists(tts_path) else ""
    }
    return JSONResponse(response_payload)


@app.get("/download/tts/{task_id}")
def get_tts_audio(task_id: str):
    tts_path = os.path.join(OUTPUT_DIR, f"{task_id}_tts.wav")
    if os.path.exists(tts_path):
        return FileResponse(tts_path, media_type="audio/wav", filename=f"{task_id}_tts.wav")
    raise HTTPException(status_code=404, detail="TTS audio file not found")

# ----------------- Legacy STT and TTS Endpoints -----------------
@app.post("/stt")
async def speech_to_text(file: UploadFile = File(..., alias="audio"), webrtc_id: str = Form(None)):
    temp_path = os.path.join(OUTPUT_DIR, f"temp_{uuid.uuid4()}.wav")
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    try:
        transcript = await transcribe_audio(temp_path, 48000)
        return {"transcript": transcript}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/tts")
async def text_to_speech(data: dict):
    text = data.get("text")
    if not text:
        return JSONResponse(status_code=400, content={"error": "Missing 'text'"})
    output_path = os.path.join(OUTPUT_DIR, f"tts_{uuid.uuid4()}.wav")
    await generate_audio_google(text, output_path)
    return FileResponse(output_path, media_type="audio/wav", filename="tts_output.wav")

@app.post("/stt_ack")
async def stt_ack(file: UploadFile = File(..., alias="audio"), webrtc_id: str = Form(None)):
    return JSONResponse({"message": "STT received"}, status_code=200)

@app.post("/debug_response")
async def debug_response(file: UploadFile = File(..., alias="audio"), webrtc_id: str = Form(None)):
    temp_path = os.path.join(OUTPUT_DIR, f"debug_{uuid.uuid4()}.wav")
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    from fastapi import BackgroundTasks
    background_tasks = BackgroundTasks()
    background_tasks.add_task(os.remove, temp_path)
    return FileResponse(temp_path, media_type="audio/wav", filename=os.path.basename(temp_path), background=background_tasks)

# ----------------- Health Check Endpoint -----------------
@app.get("/healthz")
def health():
    return JSONResponse({"status": "OK"})

# ----------------- Main -----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9001)
