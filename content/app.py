from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# URLs for other containers (these should match Docker Compose service names)
AI_LOGIC_URL = "http://ai-logic:9000/api/interact"
DB_URL = "http://game-db:9002/api/save"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/interact', methods=['POST'])
def interact():
    user_input = request.json.get("input")
    stage = request.json.get("stage")

    try:
        ai_response = requests.post(AI_LOGIC_URL, json={
            "input": user_input,
            "stage": stage
        })
        ai_data = ai_response.json()

        # Save to DB
        requests.post(DB_URL, json={
            "input": user_input,
            "response": ai_data,
            "timestamp": request.json.get("timestamp")
        })

        return jsonify(ai_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
