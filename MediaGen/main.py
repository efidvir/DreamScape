# main.py
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uuid
import os
import shutil
import asyncio

app = FastAPI()

# Directory setup
OUTPUT_DIR = "./generated_media"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Request model
class GenerationRequest(BaseModel):
    text: str

# Store tasks
TASKS_STATUS = {}

# Simulated media generation function
async def generate_media(task_id: str, instruction: str):
    try:
        TASKS_STATUS[task_id] = "started"
        # Simulate audio generation
        await asyncio.sleep(2)  # Simulate processing
        audio_path = os.path.join(OUTPUT_DIR, f"{task_id}_audio.mp3")
        with open(audio_path, "wb") as f:
            f.write(b"FAKE_AUDIO_DATA")
        TASKS_STATUS[task_id] = "audio_generated"

        # Simulate visual generation
        await asyncio.sleep(3)  # Simulate processing
        visual_path = os.path.join(OUTPUT_DIR, f"{task_id}_visual.mp4")
        with open(visual_path, "wb") as f:
            f.write(b"FAKE_VISUAL_DATA")
        TASKS_STATUS[task_id] = "completed"

    except Exception as e:
        TASKS_STATUS[task_id] = f"error: {str(e)}"

# Endpoint to initiate media generation
@app.post("/generate")
def generate(request: GenerationRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    TASKS_STATUS[task_id] = "queued"
    background_tasks.add_task(generate_media, task_id, request.text)
    return JSONResponse({"task_id": task_id})

# Endpoint to check generation status
@app.get("/status/{task_id}")
def check_status(task_id: str):
    status = TASKS_STATUS.get(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task ID not found")
    return JSONResponse({"task_id": task_id, "status": status})

# Endpoint to retrieve generated audio
@app.get("/download/{task_id}/audio")
def get_audio(task_id: str):
    audio_path = os.path.join(OUTPUT_DIR, f"{task_id}_audio.mp3")
    if os.path.exists(audio_path):
        return FileResponse(audio_path, media_type="audio/mpeg", filename=f"{task_id}_audio.mp3")
    raise HTTPException(status_code=404, detail="Audio file not found")

# Endpoint to retrieve generated visuals
@app.get("/download/{task_id}/visual")
def get_visual(task_id: str):
    visual_path = os.path.join(OUTPUT_DIR, f"{task_id}_visual.mp4")
    if os.path.exists(visual_path):
        return FileResponse(visual_path, media_type="video/mp4", filename=f"{task_id}_visual.mp4")
    raise HTTPException(status_code=404, detail="Visual file not found")

# Cleanup endpoint (optional)
@app.delete("/cleanup/{task_id}")
def cleanup(task_id: str):
    audio_path = os.path.join(OUTPUT_DIR, f"{task_id}_audio.mp3")
    visual_path = os.path.join(OUTPUT_DIR, f"{task_id}_visual.mp4")

    for path in [audio_path, visual_path]:
        if os.path.exists(path):
            os.remove(path)

    TASKS_STATUS.pop(task_id, None)
    return JSONResponse({"task_id": task_id, "status": "cleaned up"})