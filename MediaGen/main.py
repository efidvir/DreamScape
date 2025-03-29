import os
import uuid
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from media_generation import generate_visual, generate_audio
from tts_stt import generate_audio_google, transcribe_audio
import subprocess

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

# --------------------- Endpoints ----------------------
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

# ----------------- Speech-to-Text Endpoint -----------------
@app.post("/stt")
async def speech_to_text(
    file: UploadFile = File(..., alias="audio"),
    webrtc_id: str = Form(None)
):
    """
    Process the uploaded audio file for speech-to-text.
    Pass the sample rate as a positional argument (48000 Hz) to match the audio header.
    """
    temp_path = f"temp_{uuid.uuid4()}.wav"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    try:
        transcript = await transcribe_audio(temp_path, 48000)
        return {"transcript": transcript}
    finally:
        os.remove(temp_path)

@app.post("/tts")
async def text_to_speech(data: dict):
    text = data.get("text")
    if not text:
        return JSONResponse(status_code=400, content={"error": "Missing 'text'"})
    output_path = os.path.join(OUTPUT_DIR, f"tts_{uuid.uuid4()}.wav")
    await generate_audio_google(text, output_path)
    return FileResponse(output_path, media_type="audio/wav")

import os
import uuid
import subprocess
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

# Assume these are defined/imported as needed:
# OUTPUT_DIR, TASKS_STATUS, and the asynchronous transcribe_audio() function

app = FastAPI()

@app.post("/generate/transcript")
async def generate_transcript(file: UploadFile = File(..., alias="audio")):
    task_id = str(uuid.uuid4())
    
    # Save the incoming file with its original extension (if available)
    original_ext = os.path.splitext(file.filename)[1] or ".wav"
    temp_input_path = os.path.join(OUTPUT_DIR, f"{task_id}_input{original_ext}")
    with open(temp_input_path, "wb") as f:
        f.write(await file.read())
    
    # Convert the audio file to a Google-compatible WAV file:
    # Linear16 encoding, 48000 Hz sample rate, mono channel.
    converted_path = os.path.join(OUTPUT_DIR, f"{task_id}_converted.wav")
    try:
        subprocess.run([
            "ffmpeg", "-y",              # Overwrite output if exists
            "-i", temp_input_path,         # Input file
            "-ar", "48000",                # Set sample rate to 48000 Hz
            "-ac", "1",                    # Set to mono audio
            "-f", "wav",                   # Force WAV output format
            converted_path                 # Output file
        ], check=True)
    except Exception as e:
        # Clean up and return an error if conversion fails
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        return JSONResponse({"task_id": task_id, "transcript": f"Error converting file: {e}"})
    
    try:
        # Call your transcription function with the converted file.
        transcript = await transcribe_audio(converted_path, 48000)
        TASKS_STATUS[task_id] = "transcript_generated"
        return JSONResponse({"task_id": task_id, "transcript": transcript})
    finally:
        # Clean up temporary files.
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(converted_path):
            os.remove(converted_path)

# ----------------- Dummy STT Acknowledgment Endpoint -----------------
@app.post("/stt_ack")
async def stt_ack(
    file: UploadFile = File(..., alias="audio"),
    webrtc_id: str = Form(None)
):
    """
    Dummy endpoint to acknowledge reception of an STT file.
    It does not process the file; it only returns a confirmation message.
    """
    return JSONResponse({"message": "STT received"}, status_code=200)

# ----------------- Debug Echo Response Endpoint -----------------
@app.post("/debug_response")
async def debug_response(
    file: UploadFile = File(..., alias="audio"),
    webrtc_id: str = Form(None)
):
    """
    Debug endpoint to simulate a response from the LLM after transcript and TTS.
    For debugging, it simply echoes back the same audio file that was received.
    """
    temp_path = f"debug_{uuid.uuid4()}.wav"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    # Schedule removal of the temporary file after sending the response.
    background_tasks = BackgroundTasks()
    background_tasks.add_task(os.remove, temp_path)
    
    return FileResponse(
        temp_path,
        media_type=file.content_type,
        filename=file.filename,
        background=background_tasks
    )

# --- Health check route ---
@app.get("/healthz")
def health():
    return JSONResponse({"status": "OK"})

# --- Run the server on port 9001 ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9001)
