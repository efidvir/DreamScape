from flask import Flask, request, jsonify, Response, render_template_string
import requests
import datetime

app = Flask(__name__)

# Global monitor data for basic statistics.
monitor_data = {
    "requests_count": 0,
    "last_transcript": None,
    "last_llm_response": None,
    "last_request_time": None,
    "last_stage_completed": None,
    "last_end_stage": None,
}

# Global game state tracking.
game_state = {
    "last_stage_completed": None,
    "last_end_stage": None,
    "game_progress": []  # List of all interactions with timestamp
}

@app.route("/")
def home():
    return "Backend Orchestrator is running."

@app.route("/input_hook", methods=["POST"])
def input_hook():
    # Update monitor count and time.
    monitor_data["requests_count"] += 1
    monitor_data["last_request_time"] = datetime.datetime.utcnow().isoformat() + "Z"
    
    # 1. Receive audio from the client.
    user_audio = request.files.get("audio")
    if not user_audio:
        return jsonify({"error": "No audio provided"}), 400
    audio_data = user_audio.read()

    # 2. Forward audio to the media generation container's STT endpoint.
    stt_url = "http://mediagen_container:5000/stt"  # adjust this URL as needed
    try:
        stt_response = requests.post(
            stt_url,
            files={"audio": (user_audio.filename, audio_data, user_audio.mimetype)}
        )
        stt_response.raise_for_status()
    except requests.RequestException as e:
        return jsonify({"error": f"STT service error: {e}"}), 500

    transcript = stt_response.json().get("transcript")
    monitor_data["last_transcript"] = transcript or "N/A"
    if not transcript:
        return jsonify({"error": "No transcript received from STT service"}), 500

    # 3. Send transcript to the LLM container.
    llm_url = "http://llm_container:5000/generate"  # adjust this URL as needed
    try:
        llm_resp = requests.post(llm_url, json={"transcript": transcript})
        llm_resp.raise_for_status()
    except requests.RequestException as e:
        return jsonify({"error": f"LLM service error: {e}"}), 500

    # Expect the LLM response to include game state indicators.
    llm_json = llm_resp.json()
    response_text = llm_json.get("response")
    stage_completed = llm_json.get("stage_completed", "N/A")
    end_stage = llm_json.get("end_stage", False)

    monitor_data["last_llm_response"] = response_text or "N/A"
    monitor_data["last_stage_completed"] = stage_completed
    monitor_data["last_end_stage"] = end_stage

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
        return jsonify({"error": "No response received from LLM service"}), 500

    # 4. Send the response text to the media generation container's TTS endpoint.
    tts_url = "http://mediagen_container:5000/tts"  # adjust this URL as needed
    try:
        tts_resp = requests.post(tts_url, json={"text": response_text}, stream=True)
        tts_resp.raise_for_status()
    except requests.RequestException as e:
        return jsonify({"error": f"TTS service error: {e}"}), 500

    # 5. Stream the TTS audio back to the frontend.
    def generate():
        for chunk in tts_resp.iter_content(chunk_size=1024):
            if chunk:
                yield chunk

    return Response(generate(), mimetype="audio/mpeg")

@app.route("/monitor", methods=["GET"])
def monitor():
    # A simple HTML page showing monitoring and game state info.
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
      </body>
    </html>
    """,
    monitor=monitor_data,
    game=game_state)
    return html

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
