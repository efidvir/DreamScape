from flask import Flask, render_template, request, jsonify, Response, send_file
import requests
import time
import datetime
import threading
import json
import os
import uuid
from io import BytesIO

app = Flask(__name__)

# Ensure output directory exists
OUTPUT_DIR = "./generated_media"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# 1) GLOBAL HEALTH LOGIC
# ------------------------------------------------------------------------------
container_health = {
    "frontend": "unknown",
    "backend": "unknown",
    "mediagen": "unknown",
    "llm": "unknown"
}

HEALTH_ENDPOINTS = {
    "frontend":  "http://frontend:80/healthz",   # Using port 80 for frontend
    "backend":   "http://backend:5000/healthz",
    "mediagen":  "http://mediagen:9001/healthz",
    "llm":       "http://container_llm:9000/healthz"
}

def is_container_alive(url: str) -> bool:
    try:
        r = requests.get(url, timeout=2)
        return (r.status_code == 200)
    except requests.RequestException:
        return False

def health_check_loop(interval=5):
    while True:
        for name, endpoint in HEALTH_ENDPOINTS.items():
            alive = is_container_alive(endpoint)
            container_health[name] = "alive" if alive else "dead"
        time.sleep(interval)

threading.Thread(target=health_check_loop, daemon=True).start()

# ------------------------------------------------------------------------------
# 2) BASIC ENDPOINTS
# ------------------------------------------------------------------------------
@app.route("/healthz")
def healthz():
    return jsonify({"status": "OK"}), 200

@app.route("/")
def home():
    return render_template("index.html")

# ------------------------------------------------------------------------------
# 3) AUDIO PROXY & DEBUG ENDPOINTS
# ------------------------------------------------------------------------------
@app.route("/input_hook", methods=["POST"])
def input_hook():
    """
    Receives audio from the browser (FormData with 'audio' and 'webrtc_id')
    and forwards it to the MediaGen container for transcript processing.
    Note: Instead of posting to /stt, we now forward to /generate/transcript.
    """
    user_audio = request.files.get("audio")
    webrtc_id = request.form.get("webrtc_id", "no_id_provided")
    if not user_audio:
        return jsonify({"error": "No audio file provided"}), 400

    # Forward to the mediagen container's /generate/transcript endpoint.
    forward_url = "http://mediagen:9001/generate/transcript"
    files = {
        "audio": (user_audio.filename, user_audio.stream, user_audio.mimetype)
    }
    data = {"webrtc_id": webrtc_id}

    try:
        response = requests.post(forward_url, files=files, data=data, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        return jsonify({"error": f"MediaGen service error: {e}"}), 500

    return jsonify(response.json())

@app.route("/debug_response", methods=["POST"])
def debug_response():
    """
    Dummy endpoint to simulate a response from the LLM.
    It echoes back the same audio file received from the user.
    """
    user_audio = request.files.get("audio")
    if not user_audio:
        return jsonify({"error": "No audio file provided"}), 400

    file_data = user_audio.read()
    print("Received audio file of size:", len(file_data))
    user_audio.stream = BytesIO(file_data)  # Reset stream

    task_id = str(uuid.uuid4())
    temp_path = os.path.join(OUTPUT_DIR, f"debug_{task_id}.wav")
    user_audio.save(temp_path)
    return send_file(
        temp_path,
        mimetype=user_audio.mimetype,
        as_attachment=True,
        download_name=user_audio.filename
    )

@app.route("/outputs")
def outputs():
    """
    SSE endpoint that streams output events to the client.
    For debugging, it simulates:
      1. A "play" event with an audio URL and a message.
      2. After a delay, a "record" event instructing the UI to switch back to recording mode.
    """
    webrtc_id = request.args.get("webrtc_id", "no_id_provided")
    def event_stream():
        time.sleep(3)
        play_event = {
            "action": "play",
            "audio_url": "/download/debug_audio.wav",  # Adjust this URL as needed.
            "message": "Audio playback started. Please listen."
        }
        yield f"data: {json.dumps(play_event)}\n\n"
        time.sleep(5)
        record_event = {
            "action": "record",
            "status": "green",
            "message": "Audio playback finished. Ready for new input."
        }
        yield f"data: {json.dumps(record_event)}\n\n"
    return Response(event_stream(), mimetype="text/event-stream")

# ------------------------------------------------------------------------------
# 4) KEEPALIVE & HEALTH SCORE ENDPOINTS
# ------------------------------------------------------------------------------
@app.route("/keepalive")
def keepalive():
    def generate():
        while True:
            payload = {
                "frontend": container_health["frontend"],
                "backend": container_health["backend"],
                "mediagen": container_health["mediagen"],
                "llm": container_health["llm"],
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
            }
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(5)
    return Response(generate(), mimetype="text/event-stream")

@app.route("/are_all_healthy")
def are_all_healthy():
    all_ok = all(status == "alive" for status in container_health.values())
    return jsonify({
        "all_healthy": all_ok,
        "container_health": container_health
    })

# ------------------------------------------------------------------------------
# 5) OTHER AUDIO/TRANSCRIPT ENDPOINTS
# ------------------------------------------------------------------------------
@app.route("/generate/audio", methods=["POST"])
def generate_audio_endpoint():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing text"}), 400
    task_id = str(uuid.uuid4())
    return jsonify({"task_id": task_id})

@app.route("/generate/visual", methods=["POST"])
def generate_visual_endpoint():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing text"}), 400
    task_id = str(uuid.uuid4())
    return jsonify({"task_id": task_id})

@app.route("/status/<task_id>")
def check_status(task_id: str):
    return jsonify({"task_id": task_id, "status": "completed"})

@app.route("/download/<task_id>/audio")
def get_audio(task_id: str):
    audio_path = os.path.join(OUTPUT_DIR, f"{task_id}_audio.wav")
    if os.path.exists(audio_path):
        return send_file(audio_path, mimetype="audio/wav", as_attachment=True, download_name=f"{task_id}_audio.wav")
    raise Exception("Audio file not found")

@app.route("/download/<task_id>/visual")
def get_visual(task_id: str):
    visual_path = os.path.join(OUTPUT_DIR, f"{task_id}_visual.mp4")
    if os.path.exists(visual_path):
        return send_file(visual_path, mimetype="video/mp4", as_attachment=True, download_name=f"{task_id}_visual.mp4")
    raise Exception("Visual file not found")

@app.route("/cleanup/<task_id>", methods=["DELETE"])
def cleanup(task_id: str):
    audio_path = os.path.join(OUTPUT_DIR, f"{task_id}_audio.wav")
    visual_path = os.path.join(OUTPUT_DIR, f"{task_id}_visual.mp4")
    for path in [audio_path, visual_path]:
        if os.path.exists(path):
            os.remove(path)
    return jsonify({"task_id": task_id, "status": "cleaned up"})

@app.route("/stt", methods=["POST"])
def speech_to_text():
    """
    This endpoint is kept for backward compatibility.
    It returns a dummy transcript.
    """
    file = request.files.get("audio")
    webrtc_id = request.form.get("webrtc_id", "no_id_provided")
    if not file:
        return jsonify({"error": "No audio file provided"}), 400
    temp_path = os.path.join(OUTPUT_DIR, f"temp_{uuid.uuid4()}.wav")
    file.save(temp_path)
    transcript = "ggggg transcript"
    os.remove(temp_path)
    return jsonify({"transcript": transcript})

@app.route("/tts", methods=["POST"])
def text_to_speech():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text'"}), 400
    output_path = os.path.join(OUTPUT_DIR, f"tts_{uuid.uuid4()}.wav")
    with open(output_path, "wb") as f:
        f.write(b"dummy audio data")
    return send_file(output_path, mimetype="audio/wav", as_attachment=True, download_name="tts_output.wav")

@app.route("/generate/transcript", methods=["POST"])
def generate_transcript():
    file = request.files.get("audio")
    if not file:
        return jsonify({"error": "No audio file provided"}), 400

    # Generate a unique task ID and temporary file path.
    task_id = str(uuid.uuid4())
    temp_path = os.path.join(OUTPUT_DIR, f"{task_id}_input.wav")
    file.save(temp_path)

    try:
        # Forward the audio file to the mediagen container's /stt endpoint.
        # (This endpoint should perform the actual transcription.)
        with open(temp_path, "rb") as f:
            files = {"audio": (file.filename, f, file.mimetype)}
            data = {"webrtc_id": "dummy"}  # Adjust or pass a real ID if needed.
            response = requests.post("http://mediagen:9001/generate/transcript", files=files, data=data, timeout=30)
            response.raise_for_status()
            transcript_data = response.json()
            transcript = transcript_data.get("transcript", "No transcript received")
    except requests.exceptions.Timeout:
        transcript = "Transcript request timed out"
    except Exception as e:
        transcript = f"Error: {e}"
    finally:
        # Remove the temporary audio file.
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Optionally, save the transcript to a file (for logging or debugging).
    transcript_path = os.path.join(OUTPUT_DIR, f"{task_id}_transcript.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    return jsonify({"task_id": task_id, "transcript": transcript})

@app.route("/stt_ack", methods=["POST"])
def stt_ack():
    file = request.files.get("audio")
    webrtc_id = request.form.get("webrtc_id", "no_id_provided")
    if not file:
        return jsonify({"error": "No audio file provided"}), 400
    return jsonify({"message": "STT received"})

# ------------------------------------------------------------------------------
# 6) MAIN
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=80)
