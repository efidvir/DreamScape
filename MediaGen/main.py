import os
import uuid
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from media_generation import generate_visual, generate_audio
from tts_stt import generate_audio_google, transcribe_audio

# === CONFIG ===
OUTPUT_DIR = "./generated_media"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI()

# Request model
class GenerationRequest(BaseModel):
    text: str

# Store tasks
TASKS_STATUS = {}

# === Media generation tasks ===
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

# === Endpoints ===
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

@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    temp_path = f"temp_{uuid.uuid4()}.wav"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    try:
        transcript = await transcribe_audio(temp_path)
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

@app.post("/generate/transcript")
async def generate_transcript(file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())
    temp_path = os.path.join(OUTPUT_DIR, f"{task_id}_input.wav")
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    try:
        transcript = await transcribe_audio(temp_path)
        TASKS_STATUS[task_id] = "transcript_generated"
        transcript_path = os.path.join(OUTPUT_DIR, f"{task_id}_transcript.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        return JSONResponse({"task_id": task_id, "transcript": transcript})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
