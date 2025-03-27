# backend.py
from flask import Flask, request, jsonify, Response, render_template_string
import requests
import datetime
import time

app = Flask(__name__)

# -----------------------------------------------------------------------------
# Global Monitoring
# -----------------------------------------------------------------------------
monitor_data = {
    "requests_count": 0,
    "last_transcript": None,
    "last_llm_response": None,
    "last_request_time": None,
    "last_stage_completed": None,
    "last_end_stage": None,
}

game_state = {
    "last_stage_completed": None,
    "last_end_stage": None,
    "game_progress": []
}

# Real-time logs stored in memory
live_logs = []


def log_event(message):
    """Append a log message with a timestamp to the live logs and print it."""
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    log = f"{timestamp} - {message}"
    print(log)
    live_logs.append(log)


# -----------------------------------------------------------------------------
# Health Checks
# -----------------------------------------------------------------------------
@app.route("/healthz")
def healthz():
    """
    A simple health endpoint for the backend itself.
    Returns 200 if everything is fine.
    """
    return jsonify({"status": "OK"}), 200


def is_container_alive(url: str) -> bool:
    """
    Attempt to GET /healthz from the given URL, returning True if 200 OK.
    """
    try:
        r = requests.get(url, timeout=2)
        return (r.status_code == 200)
    except requests.RequestException:
        return False


@app.route("/keepalive")
def keepalive():
    """
    SSE endpoint: check health of frontend, backend, mediagen, and llm containers
    every few seconds and stream the results.
    Adjust the container hostnames/ports below to match your Docker Compose.
    """
    def generate():
        while True:
            # Example: update these hostnames and ports as needed
            status_frontend = "alive" if is_container_alive("http://frontend:80/healthz") else "dead"
            status_backend = "alive" if is_container_alive("http://backend:5000/healthz") else "dead"
            status_mediagen = "alive" if is_container_alive("http://mediagen:9001/healthz") else "dead"
            status_llm = "alive" if is_container_alive("http://container_llm:9000/healthz") else "dead"

            payload = {
                "frontend": status_frontend,
                "backend": status_backend,
                "mediagen": status_mediagen,
                "llm": status_llm,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
            }
            yield f"data: {payload}\n\n"
            time.sleep(5)
    return Response(generate(), mimetype="text/event-stream")


# -----------------------------------------------------------------------------
# Main Endpoints
# -----------------------------------------------------------------------------
@app.route("/")
def home():
    log_event("Home page requested.")
    return "Backend Orchestrator is running."


@app.route("/input_hook", methods=["POST"])
def input_hook():
    """
    Receives audio from the frontend, forwards it to STT, then LLM, then TTS, 
    and finally streams audio back to the user.
    """
    log_event(f"Received /input_hook request from {request.remote_addr}.")
    monitor_data["requests_count"] += 1
    monitor_data["last_request_time"] = datetime.datetime.utcnow().isoformat() + "Z"
    
    # 1) Receive audio
    user_audio = request.files.get("audio")
    if not user_audio:
        log_event("No audio provided in the request.")
        return jsonify({"error": "No audio provided"}), 400
    audio_data = user_audio.read()
    log_event(f"Audio received: filename={user_audio.filename}, mimetype={user_audio.mimetype}")

    # 2) Forward audio to the media generation container's STT endpoint
    stt_url = "http://mediagen:5000/stt"  # Adjust if needed
    log_event(f"Forwarding audio to STT endpoint: {stt_url}")
    try:
        stt_response = requests.post(
            stt_url,
            files={"audio": (user_audio.filename, audio_data, user_audio.mimetype)}
        )
        stt_response.raise_for_status()
    except requests.RequestException as e:
        log_event(f"STT service error: {e}")
        return jsonify({"error": f"STT service error: {e}"}), 500

    transcript = stt_response.json().get("transcript")
    monitor_data["last_transcript"] = transcript or "N/A"
    if not transcript:
        log_event("No transcript received from STT service.")
        return jsonify({"error": "No transcript received from STT service"}), 500
    log_event(f"Transcript received: {transcript}")

    # 3) Send transcript to LLM container
    llm_url = "http://container_llm:5000/generate"
    log_event(f"Sending transcript to LLM endpoint: {llm_url}")
    try:
        llm_resp = requests.post(llm_url, json={"transcript": transcript})
        llm_resp.raise_for_status()
    except requests.RequestException as e:
        log_event(f"LLM service error: {e}")
        return jsonify({"error": f"LLM service error: {e}"}), 500

    llm_json = llm_resp.json()
    response_text = llm_json.get("response")
    stage_completed = llm_json.get("stage_completed", "N/A")
    end_stage = llm_json.get("end_stage", False)

    monitor_data["last_llm_response"] = response_text or "N/A"
    monitor_data["last_stage_completed"] = stage_completed
    monitor_data["last_end_stage"] = end_stage

    log_event(f"LLM response received: {response_text}, "
              f"Stage completed: {stage_completed}, End stage: {end_stage}")

    # Update game state
    game_state["last_stage_completed"] = stage_completed
    game_state["last_end_stage"] = end_stage
    game_state["game_progress"].append({
        "transcript": transcript,
        "response": response_text,
        "stage_completed": stage_completed,
        "end_stage": end_stage,
        "time": datetime.datetime.utcnow().isoformat() + "Z"
    })

    if not response_text:
        log_event("No response received from LLM service.")
        return jsonify({"error": "No response received from LLM service"}), 500

    # 4) Send the LLM's text response to the TTS endpoint
    tts_url = "http://mediagen:5000/tts"
    log_event(f"Sending response text to TTS endpoint: {tts_url}")
    try:
        tts_resp = requests.post(tts_url, json={"text": response_text}, stream=True)
        tts_resp.raise_for_status()
    except requests.RequestException as e:
        log_event(f"TTS service error: {e}")
        return jsonify({"error": f"TTS service error: {e}"}), 500

    log_event("Starting to stream TTS audio back to the client.")

    # 5) Stream the TTS audio
    def generate():
        for chunk in tts_resp.iter_content(chunk_size=1024):
            if chunk:
                yield chunk
        log_event("Finished streaming TTS audio.")

    return Response(generate(), mimetype="audio/mpeg")


@app.route("/monitor", methods=["GET"])
def monitor():
    """
    A simple HTML page showing monitoring data, game state, and live logs.
    """
    html = render_template_string("""
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="UTF-8" />
        <title>Backend Monitor & Game State</title>
        <style>
          body { font-family: Arial, sans-serif; background: #f2f2f2; padding: 20px; }
          h1 { color: #333; }
          .data { background: #fff; padding: 10px; border: 1px solid #ccc; border-radius: 4px; margin-bottom: 20px; }
          p { margin: 5px 0; }
          pre { background: #eee; padding: 10px; border-radius: 4px; }
          #logs li { font-family: monospace; font-size: 12px; }
        </style>
      </head>
      <body>
        <h1>Backend Monitor</h1>
        <div class="data">
          <p><strong>Requests Processed:</strong> {{ monitor.requests_count }}</p>
          <p><strong>Last Request Time:</strong> {{ monitor.last_request_time }}</p>
          <p><strong>Last Transcript:</strong> {{ monitor.last_transcript }}</p>
          <p><strong>Last LLM Response:</strong> {{ monitor.last_llm_response }}</p>
          <p><strong>Last Stage Completed:</strong> {{ monitor.last_stage_completed }}</p>
          <p><strong>Last End Stage:</strong> {{ monitor.last_end_stage }}</p>
        </div>
        <h1>Game State</h1>
        <div class="data">
          <p><strong>Last Stage Completed:</strong> {{ game.last_stage_completed }}</p>
          <p><strong>Last End Stage:</strong> {{ game.last_end_stage }}</p>
          <h2>Game Progress:</h2>
          <pre>{{ game.game_progress | tojson(indent=2) }}</pre>
        </div>
        <h1>Live Logs</h1>
        <div class="data">
          <ul id="logs"></ul>
        </div>
        <script>
          const evtSource = new EventSource("/logs");
          evtSource.onmessage = function(event) {
            const logList = document.getElementById("logs");
            const li = document.createElement("li");
            li.textContent = event.data;
            logList.appendChild(li);
          };
        </script>
      </body>
    </html>
    """,
    monitor=monitor_data,
    game=game_state)
    return html


@app.route("/logs")
def logs():
    """
    Real-time SSE endpoint: streams live logs for debugging or monitoring.
    """
    def generate():
        last_index = 0
        while True:
            if last_index < len(live_logs):
                for log in live_logs[last_index:]:
                    yield f"data: {log}\n\n"
                last_index = len(live_logs)
            time.sleep(1)
    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    log_event("Starting backend orchestrator.")
    app.run(debug=True, host="0.0.0.0", port=5000)
