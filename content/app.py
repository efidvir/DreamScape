from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/audio", methods=["POST"])
def audio():
    user_audio = request.files["audio"]
    # TODO: Process with your LLM or speech logic
    return jsonify({"response": "AI Response (placeholder)"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
