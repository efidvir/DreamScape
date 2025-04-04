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
    "frontend":  "http://frontend:80/healthz",
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

# ------------------------------------------------------------------------------
# 3) HOME PAGE 
# ------------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# ------------------------------------------------------------------------------
# 4) AUDIO PROXY & DEBUG ENDPOINTS 
# ------------------------------------------------------------------------------
@app.route("/input_hook", methods=["POST"])
def input_hook():
    """
    Receives audio from the browser and forwards it to the MediaGen container for transcript processing.
    Expects the MediaGen response in a structured JSON format (with keys such as "transcript", "llm_user_response", etc.).
    If the expected fields are missing, returns an error asking the user to re-upload the files.
    """
    user_audio = request.files.get("audio")
    webrtc_id = request.form.get("webrtc_id", "no_id_provided")
    if not user_audio:
        return jsonify({"error": "No audio file provided"}), 400

    forward_url = "http://mediagen:9001/generate/transcript"
    files = {
        "audio": (user_audio.filename, user_audio.stream, user_audio.mimetype)
    }
    data = {"webrtc_id": webrtc_id}

    try:
        response = requests.post(forward_url, files=files, data=data, timeout=30)
        response.raise_for_status()
        resp_json = response.json()
    except requests.RequestException as e:
        return jsonify({"error": f"MediaGen service error: {e}"}), 500

    # Check if the response has the expected keys.
    if not resp_json or "transcript" not in resp_json or "llm_user_response" not in resp_json:
        return jsonify({
            "error": "Response from MediaGen is incomplete or expired. Please re-upload your audio file."
        }), 400

    # Return the JSON response as-is.
    return jsonify(resp_json)

@app.route("/outputs")
def outputs():
    webrtc_id = request.args.get("webrtc_id", "no_id_provided")
    def event_stream():
        time.sleep(3)
        play_event = {
            "action": "play",
            "audio_url": "/download/debug_audio.wav",
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
# 5) KEEPALIVE & HEALTH SCORE ENDPOINTS
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
# 6) OTHER AUDIO/TRANSCRIPT ENDPOINTS (unchanged)
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
            data = {"webrtc_id": "dummy"}
            response = requests.post("http://mediagen:9001/generate/transcript", files=files, data=data, timeout=30)
            response.raise_for_status()
            transcript_data = response.json()
            transcript = transcript_data.get("transcript", "No transcript received")
    except requests.exceptions.Timeout:
        transcript = "Transcript request timed out"
    except Exception as e:
        transcript = f"Error: {e}"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

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

@app.route("/debug_response", methods=["POST"])
def debug_response():
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

@app.route('/mediagen/video_status/<task_id>')
def proxy_video_status(task_id):
    """Proxy endpoint to check video generation status in MediaGen."""
    try:
        response = requests.get(f"http://mediagen:9001/video_status/{task_id}", timeout=5)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route('/get/visual/<task_id>')
def proxy_visual(task_id):
    """Proxy endpoint to get video from MediaGen."""
    try:
        # Forward the request to MediaGen
        response = requests.get(f"http://mediagen:9001/get/visual/{task_id}", stream=True)
        
        # If the video exists, stream it back to the client
        if response.status_code == 200:
            return Response(
                response.iter_content(chunk_size=1024),
                content_type=response.headers['Content-Type'],
                status=response.status_code
            )
        else:
            return jsonify({"error": "Video not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------------------
# 6) MAIN
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=80)
