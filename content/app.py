from flask import Flask, render_template, request, jsonify, Response
import requests
import time
import datetime
import threading

app = Flask(__name__)

# ------------------------------------------------------------------------------
# 1) GLOBAL HEALTH LOGIC
# ------------------------------------------------------------------------------
# We'll store the health status of each container here as "alive" or "dead"
# (or "unknown" if not yet checked).
container_health = {
    "frontend": "unknown",
    "backend": "unknown",
    "mediagen": "unknown",
    "llm": "unknown"
}

# Define which containers to check and their /healthz endpoints.
# Update hostnames/ports to reflect your Docker Compose or environment.
HEALTH_ENDPOINTS = {
    "frontend":  "http://frontend:80/healthz",   # CHANGED from 5000 to 80
    "backend":   "http://backend:5000/healthz",
    "mediagen":  "http://mediagen:9001/healthz",
    "llm":       "http://container_llm:9000/healthz"
}


def is_container_alive(url: str) -> bool:
    """Return True if GET /healthz at 'url' returns 200 OK."""
    try:
        r = requests.get(url, timeout=2)
        return (r.status_code == 200)
    except requests.RequestException:
        return False

def health_check_loop(interval=5):
    """Runs in the background, periodically checking each container's health."""
    while True:
        for name, endpoint in HEALTH_ENDPOINTS.items():
            alive = is_container_alive(endpoint)
            container_health[name] = "alive" if alive else "dead"
        time.sleep(interval)

# Start the background health-check thread on startup
threading.Thread(target=health_check_loop, daemon=True).start()

# ------------------------------------------------------------------------------
# 2) BASIC ENDPOINTS
# ------------------------------------------------------------------------------
@app.route("/healthz")
def healthz():
    """
    Minimal health endpoint for *this* frontend container.
    E.g., 'curl http://<frontend_host>/healthz' -> 200 OK if running
    """
    return jsonify({"status": "OK"}), 200

@app.route("/")
def home():
    """
    Renders your main index.html (with swirl background, microphone toggle, wave, etc.)
    Make sure index.html is located in the 'templates/' folder so render_template works.
    """
    return render_template("index.html")

## ------------------------------------------------------------------------------
# 3) AUDIO PROXY ENDPOINTS
# ------------------------------------------------------------------------------
@app.route("/input_hook", methods=["POST"])
def input_hook():
    """
    Receives audio from the browser (FormData with 'audio' and 'webrtc_id')
    and forwards it to the MediaGen container for speech-to-text (STT).
    """
    user_audio = request.files.get("audio")
    webrtc_id = request.form.get("webrtc_id", "no_id_provided")
    if not user_audio:
        return jsonify({"error": "No audio file provided"}), 400

    # Forward the audio to MediaGen for transcription.
    # Since mediagen is running internally on port 9001, we use that here.
    forward_url = "http://mediagen:9001/stt"

    files = {
        "audio": (user_audio.filename, user_audio.stream, user_audio.mimetype)
    }
    data = {"webrtc_id": webrtc_id}

    try:
        response = requests.post(forward_url, files=files, data=data)
        response.raise_for_status()
    except requests.RequestException as e:
        return jsonify({"error": f"MediaGen service error: {e}"}), 500

    # Return the JSON result from the mediagen container (e.g., transcript)
    return jsonify(response.json())

@app.route("/outputs")
@app.route("/outputs")
def outputs():
    """
    SSE endpoint that streams output events from the MediaGen container.
    
    For debugging, this endpoint simulates:
      1. A "play" event that provides a URL for the audio file to play.
      2. After the audio is expected to finish, a "record" event indicating that 
         the UI should switch to green (ready for user input).
    """
    webrtc_id = request.args.get("webrtc_id", "no_id_provided")
    
    def event_stream():
        # Simulate processing delay before audio is ready
        time.sleep(3)
        # Simulate a play event with a dummy audio URL.
        # (Make sure this URL corresponds to a valid file served by your app.)
        play_event = {"action": "play", "audio_url": f"/download/debug_audio.wav"}
        yield f"data: {json.dumps(play_event)}\n\n"
        
        # Wait for the audio to finish playing (simulate duration)
        time.sleep(5)
        # Send a record event to instruct the frontend to switch to recording mode (green audio wave)
        record_event = {"action": "record", "status": "green"}
        yield f"data: {json.dumps(record_event)}\n\n"
    
    return Response(event_stream(), mimetype="text/event-stream")

# ------------------------------------------------------------------------------
# 4) KEEPALIVE & HEALTH SCORE ENDPOINTS
# ------------------------------------------------------------------------------
@app.route("/keepalive")
def keepalive():
    """
    SSE endpoint that streams the container_health dict every few seconds,
    letting the frontend see if each container is 'alive' or 'dead'.
    """
    def generate():
        while True:
            payload = {
                "frontend": container_health["frontend"],
                "backend":  container_health["backend"],
                "mediagen": container_health["mediagen"],
                "llm":      container_health["llm"],
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
            }
            # SSE requires each message in the form: "data: ...\n\n"
            yield f"data: {payload}\n\n"
            time.sleep(5)
    return Response(generate(), mimetype="text/event-stream")

@app.route("/are_all_healthy")
def are_all_healthy():
    """
    Simple JSON telling if all containers are 'alive'. 
    Could be used by your game logic to decide if the user can continue.
    """
    all_ok = all(status == "alive" for status in container_health.values())
    return jsonify({
        "all_healthy": all_ok,
        "container_health": container_health
    })

# ------------------------------------------------------------------------------
# 5) MAIN
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # In production, set debug=False
    # CHANGED from port=5000 to port=80 so it matches "80:80" in your docker-compose.
    app.run(debug=True, host="0.0.0.0", port=80)
