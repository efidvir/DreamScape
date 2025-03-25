from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

# Endpoint to receive audio from the client and forward it to the LLM container.
@app.route("/input_hook", methods=["POST"])
def input_hook():
    user_audio = request.files["audio"]
    webrtc_id = request.form.get("webrtc_id")
    # Update this URL to point to your LLM container's input endpoint.
    llm_url = "http://llm_container:5000/input_hook"
    
    files = {
        "audio": (user_audio.filename, user_audio.stream, user_audio.mimetype)
    }
    data = {"webrtc_id": webrtc_id}
    
    try:
        response = requests.post(llm_url, files=files, data=data)
        response.raise_for_status()
    except requests.RequestException as e:
        return jsonify({"error": f"LLM service error: {e}"}), 500
    
    return jsonify(response.json())

# SSE endpoint to proxy outputs from the LLM container.
@app.route("/outputs")
def outputs():
    webrtc_id = request.args.get("webrtc_id")
    # Update this URL to point to your LLM container's SSE endpoint.
    llm_sse_url = f"http://llm_container:5000/outputs?webrtc_id={webrtc_id}"
    
    # For a production system, you’d proxy the SSE stream from the LLM container.
    # Here, you could also simply redirect the client or implement an SSE proxy.
    # For now, we return a placeholder response.
    return jsonify({"message": "SSE proxy not implemented in this sample."})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
