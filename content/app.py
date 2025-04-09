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
OUTPUT_DIR = "/app/static/media"
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
        
        # Ensure task_id is in the response
        if "task_id" not in resp_json and webrtc_id != "no_id_provided":
            resp_json["task_id"] = webrtc_id
            
        # Add video_url if missing but task_id is present
        if "task_id" in resp_json and "video_url" not in resp_json:
            resp_json["video_url"] = f"/get/visual/{resp_json['task_id']}"
            
        # Also add audio_url if missing
        if "task_id" in resp_json and "tts_audio_url" not in resp_json:
            resp_json["tts_audio_url"] = f"/get/audio/{resp_json['task_id']}"
            
    except requests.RequestException as e:
        return jsonify({"error": f"MediaGen service error: {e}"}), 500

    # Check if the response has the expected keys.
    if not resp_json or "transcript" not in resp_json or "llm_user_response" not in resp_json:
        return jsonify({
            "error": "Response from MediaGen is incomplete or expired. Please re-upload your audio file."
        }), 400

    # Return the enhanced JSON response
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

@app.route("/check_audio_exists/<task_id>")
def check_audio_exists(task_id):
    """Proxy to MediaGen to check if audio file exists"""
    try:
        # Forward the request to MediaGen
        response = requests.get(f"http://mediagen:9001/get/audio/{task_id}", method="HEAD", timeout=2)
        if response.status_code == 200:
            return jsonify({
                "exists": True,
                "url": f"/get/audio/{task_id}",  # URL to proxy endpoint
                "size": int(response.headers.get("Content-Length", 0))
            })
        else:
            # Try alternative endpoint
            alt_response = requests.get(f"http://mediagen:9001/download/tts/{task_id}", method="HEAD", timeout=2)
            if alt_response.status_code == 200:
                return jsonify({
                    "exists": True,
                    "url": f"/download/tts/{task_id}",  # URL to proxy endpoint
                    "size": int(alt_response.headers.get("Content-Length", 0))
                })
    except Exception as e:
        print(f"Error checking audio in MediaGen: {e}")
    
    return jsonify({"exists": False})

# Add the new check_video_exists endpoint similar to check_audio_exists
@app.route("/check_video_exists/<task_id>")
def check_video_exists(task_id):
    """Proxy to MediaGen to check if video file exists"""
    try:
        # Forward the request to MediaGen
        response = requests.get(f"http://mediagen:9001/get/visual/{task_id}", method="HEAD", timeout=2)
        if response.status_code == 200:
            return jsonify({
                "exists": True,
                "url": f"/get/visual/{task_id}",  # URL to proxy endpoint
                "size": int(response.headers.get("Content-Length", 0))
            })
        else:
            # Try local file
            video_path = os.path.join(OUTPUT_DIR, f"{task_id}_visual.mp4")
            if os.path.exists(video_path):
                file_size = os.path.getsize(video_path)
                return jsonify({
                    "exists": True,
                    "url": f"/get/visual/{task_id}",
                    "size": file_size
                })
    except Exception as e:
        print(f"Error checking video in MediaGen: {e}")
    
    return jsonify({"exists": False})
    
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


# ------------------------------------------------------------------------------
# 7) ENHANCED VIDEO ENDPOINTS - Modified for better playback
# ------------------------------------------------------------------------------

@app.route("/get/visual/<task_id>")
def get_visual_alternative(task_id: str):
    visual_path = os.path.join(OUTPUT_DIR, f"{task_id}_visual.mp4")
    if os.path.exists(visual_path):
        # Create response without headers parameter
        response = send_file(
            visual_path, 
            mimetype="video/mp4", 
            as_attachment=False,
            download_name=f"{task_id}_visual.mp4",
            conditional=True,
            etag=True
        )
        
        # Add headers directly to the response object
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Cache-Control'] = 'public, max-age=3600'
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Content-Type'] = 'video/mp4'
        
        return response
        
    return jsonify({"error": "Visual file not found"}), 404

# Add this function to fetch files from MediaGen when needed
def fetch_file_from_mediagen(task_id, file_type="audio"):
    """
    Attempt to fetch a file from MediaGen and save it locally.
    
    Args:
        task_id: The task ID
        file_type: Either "audio" or "visual"
        
    Returns:
        Path to the saved file if successful, None otherwise
    """
    try:
        if file_type == "audio":
            url = f"http://mediagen:9001/get/audio/{task_id}"
            local_path = os.path.join(OUTPUT_DIR, f"{task_id}_tts.wav")
            mimetype = "audio/wav"
        else:  # visual
            url = f"http://mediagen:9001/get/visual/{task_id}"
            local_path = os.path.join(OUTPUT_DIR, f"{task_id}_visual.mp4")
            mimetype = "video/mp4"
        
        # Try to fetch the file
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Save the file locally
            with open(local_path, "wb") as f:
                f.write(response.content)
            
            print(f"Successfully fetched {file_type} from MediaGen: {local_path}")
            return local_path
    except Exception as e:
        print(f"Error fetching {file_type} from MediaGen: {e}")
    
    return None

# Update get_audio_alternative to fetch from MediaGen when needed
@app.route("/get/audio/<task_id>")
def get_audio_alternative(task_id: str):
    """Alternative endpoint for audio retrieval (for backward compatibility)"""
    # Try multiple paths
    paths_to_check = [
        os.path.join(OUTPUT_DIR, f"{task_id}_tts.wav"),
        os.path.join(OUTPUT_DIR, f"{task_id}_audio.wav")
    ]
    
    # Check if any of the local paths exist
    local_file = None
    for path in paths_to_check:
        if os.path.exists(path):
            local_file = path
            break
    
    # If not found locally, try to fetch from MediaGen
    if not local_file:
        local_file = fetch_file_from_mediagen(task_id, "audio")
    
    # If we have a file, serve it
    if local_file and os.path.exists(local_file):
        response = send_file(
            local_file, 
            mimetype="audio/wav", 
            as_attachment=False,
            download_name=f"{task_id}_audio.wav",
            conditional=True,
            etag=True
        )
        
        # Add headers directly to the response object
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Cache-Control'] = 'public, max-age=3600'
        response.headers['Access-Control-Allow-Origin'] = '*'
        
        return response
    
    # Return 404 if no file found
    return jsonify({"error": "Audio file not found"}), 404

@app.route("/download/<task_id>/audio")
def get_audio(task_id: str):
    """Serve audio files with proper headers for streaming"""
    # Check multiple possible paths (MediaGen uses different naming conventions)
    paths_to_check = [
        os.path.join(OUTPUT_DIR, f"{task_id}_tts.wav"),  # MediaGen's primary naming
        os.path.join(OUTPUT_DIR, f"{task_id}_audio.wav")  # Alternative naming
    ]
    
    for audio_path in paths_to_check:
        if os.path.exists(audio_path):
            # Create response without headers parameter
            response = send_file(
                audio_path, 
                mimetype="audio/wav", 
                as_attachment=False,
                download_name=f"{task_id}_audio.wav",
                conditional=True,
                etag=True
            )
            
            # Add headers directly to the response object
            response.headers['Accept-Ranges'] = 'bytes'
            response.headers['Cache-Control'] = 'public, max-age=3600'
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Content-Type'] = 'audio/wav'
            
            return response
            
    # Return 404 if no file found in any location
    return jsonify({"error": "Audio file not found"}), 404

@app.route("/download/<task_id>/visual")
def get_visual(task_id: str):
    visual_path = os.path.join(OUTPUT_DIR, f"{task_id}_visual.mp4")
    if os.path.exists(visual_path):
        # Create response without headers parameter
        response = send_file(
            visual_path, 
            mimetype="video/mp4", 
            as_attachment=False,
            download_name=f"{task_id}_visual.mp4",
            conditional=True,
            etag=True
        )
        
        # Add headers directly to the response object
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Cache-Control'] = 'public, max-age=3600'
        
        return response
        
    # Return 404 instead of raising exception
    return jsonify({"error": "Visual file not found"}), 404


# Add this endpoint to content/app.py 
@app.route('/audio_status/<task_id>')
def audio_status(task_id):
    """Check if an audio file is ready and provide its URL"""
    # Check multiple paths for local files
    local_paths = [
        os.path.join(OUTPUT_DIR, f"{task_id}_tts.wav"),
        os.path.join(OUTPUT_DIR, f"{task_id}_audio.wav")
    ]
    
    for path in local_paths:
        if os.path.exists(path):
            print(f"Found audio file for {task_id} at {path}")  # Add debug output
            return jsonify({
                "task_id": task_id,
                "status": "completed",
                "audio_url": f"/get/audio/{task_id}"
            })
    
    # If not found locally, try checking with MediaGen
    try:
        response = requests.get(f"http://mediagen:9001/audio_status/{task_id}", timeout=3)
        if response.status_code == 200:
            data = response.json()
            print(f"MediaGen returned audio status: {data}")  # Add debug output
            
            # If MediaGen says the file is ready, try to copy it to the local OUTPUT_DIR
            if data.get("status") in ["completed", "audio_completed"] and data.get("file_info", {}).get("exists", False):
                # Try to fetch the file from MediaGen and save locally
                try:
                    audio_response = requests.get(f"http://mediagen:9001/get/audio/{task_id}", timeout=10)
                    if audio_response.status_code == 200:
                        local_path = os.path.join(OUTPUT_DIR, f"{task_id}_tts.wav")
                        with open(local_path, "wb") as f:
                            f.write(audio_response.content)
                        print(f"Copied audio from MediaGen to {local_path}")
                except Exception as e:
                    print(f"Error copying audio from MediaGen: {e}")
            
            return jsonify(data)
        else:
            return jsonify({
                "task_id": task_id, 
                "status": "generating", 
                "message": "Audio still being generated"
            })
    except Exception as e:
        return jsonify({
            "task_id": task_id, 
            "status": "unknown",
            "error": str(e)
        })

@app.route('/debug/list_files')
def list_files():
    """Debug endpoint to list files in OUTPUT_DIR"""
    files = []
    if os.path.exists(OUTPUT_DIR):
        for filename in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, filename)
            try:
                stats = os.stat(file_path)
                files.append({
                    "filename": filename,
                    "size": stats.st_size,
                    "modified": datetime.datetime.fromtimestamp(stats.st_mtime).isoformat(),
                    "is_file": os.path.isfile(file_path)
                })
            except Exception as e:
                files.append({
                    "filename": filename,
                    "error": str(e)
                })
    
    return jsonify({
        "output_dir": OUTPUT_DIR,
        "file_count": len(files),
        "files": files
    })   

@app.route('/video_status/<task_id>')
def video_status(task_id):
    """Check if a video is ready and provide its URL"""
    # First check if the file exists locally
    video_path = os.path.join(OUTPUT_DIR, f"{task_id}_visual.mp4")
    file_exists = os.path.exists(video_path)
    
    if file_exists:
        return jsonify({
            "task_id": task_id,
            "status": "completed",
            "video_url": f"/get/visual/{task_id}"
        })
    
    # If not found locally, try checking with MediaGen
    try:
        response = requests.get(f"http://mediagen:9001/video_status/{task_id}", timeout=3)
        if response.status_code == 200:
            data = response.json()
            
            # If MediaGen says the file is ready, try to copy it locally
            if data.get("status") == "completed" and data.get("video_url"):
                # Try to fetch the video from MediaGen
                fetch_file_from_mediagen(task_id, "visual")
            
            return jsonify(data)
        else:
            return jsonify({
                "task_id": task_id, 
                "status": "generating", 
                "message": "Video still being generated"
            })
    except Exception as e:
        return jsonify({
            "task_id": task_id, 
            "status": "unknown",
            "error": str(e)
        })

@app.route('/mediagen/video_status/<task_id>')
def proxy_video_status(task_id):
    """Proxy endpoint to check video generation status in MediaGen."""
    try:
        response = requests.get(f"http://mediagen:9001/video_status/{task_id}", timeout=5)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route('/get/visual/<task_id>', methods=['HEAD'])
def head_visual(task_id: str):
    """HEAD request handler for video files"""
    # First check local file
    video_path = os.path.join(OUTPUT_DIR, f"{task_id}_visual.mp4")
    
    if os.path.exists(video_path):
        # Get file stats
        file_stats = os.stat(video_path)
        
        # Create a Response object
        response = Response("")
        
        # Add headers to the response
        response.headers['Content-Length'] = str(file_stats.st_size)
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Cache-Control'] = 'public, max-age=3600'
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Content-Type'] = 'video/mp4'
        
        return response
    
    # If not found locally, check with MediaGen
    try:
        response = requests.head(f"http://mediagen:9001/get/visual/{task_id}", timeout=2)
        if response.status_code == 200:
            # Forward headers
            resp = Response("")
            resp.headers['Content-Length'] = response.headers.get('Content-Length', '0')
            resp.headers['Accept-Ranges'] = 'bytes'
            resp.headers['Cache-Control'] = 'public, max-age=3600'
            resp.headers['Access-Control-Allow-Origin'] = '*'
            resp.headers['Content-Type'] = 'video/mp4'
            return resp
    except Exception as e:
        print(f"Error checking video HEAD in MediaGen: {e}")
        
    # Return 404 if no file found
    return "", 404

@app.route('/get/audio/<task_id>', methods=['HEAD'])
def head_audio(task_id: str):
    """HEAD request handler for audio files"""
    # Check multiple possible paths
    paths_to_check = [
        os.path.join(OUTPUT_DIR, f"{task_id}_tts.wav"),
        os.path.join(OUTPUT_DIR, f"{task_id}_audio.wav")
    ]
    
    for audio_path in paths_to_check:
        if os.path.exists(audio_path):
            # Get file stats
            file_stats = os.stat(audio_path)
            
            # Create a Response object
            response = Response("")
            
            # Add headers to the response
            response.headers['Content-Length'] = str(file_stats.st_size)
            response.headers['Accept-Ranges'] = 'bytes'
            response.headers['Cache-Control'] = 'public, max-age=3600'
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Content-Type'] = 'audio/wav'
            
            return response
            
    # Return 404 if no file found
    return "", 404

# ------------------------------------------------------------------------------
# 8) EXISTING ENDPOINTS - Unmodified
# ------------------------------------------------------------------------------
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

@app.route('/get/visual/<task_id>')
def proxy_visual(task_id):
    """Proxy endpoint to get video from MediaGen."""
    try:
        # First check if we have the file locally
        local_path = os.path.join(OUTPUT_DIR, f"{task_id}_visual.mp4")
        if os.path.exists(local_path):
            # Create response without headers parameter
            response = send_file(
                local_path, 
                mimetype="video/mp4", 
                as_attachment=False,
                download_name=f"{task_id}_visual.mp4",
                conditional=True,
                etag=True
            )
            
            # Add headers directly to the response object
            response.headers['Accept-Ranges'] = 'bytes'
            response.headers['Cache-Control'] = 'public, max-age=3600'
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Content-Type'] = 'video/mp4'
            
            return response
            
        # Forward the request to MediaGen
        response = requests.get(f"http://mediagen:9001/get/visual/{task_id}", stream=True)
        
        # If the video exists, stream it back to the client
        if response.status_code == 200:
            # Try to save to local cache
            try:
                video_data = response.content
                with open(local_path, "wb") as f:
                    f.write(video_data)
                print(f"Cached video file locally: {local_path}")
                
                # Serve the local file
                return send_file(
                    local_path,
                    mimetype="video/mp4",
                    as_attachment=False,
                    download_name=f"{task_id}_visual.mp4",
                    conditional=True,
                    etag=True
                )
            except Exception as e:
                print(f"Error caching video: {e}")
                
                # Fall back to proxying
                return Response(
                    response.iter_content(chunk_size=1024),
                    content_type=response.headers['Content-Type'],
                    status=response.status_code,
                    headers={
                        'Accept-Ranges': 'bytes',
                        'Cache-Control': 'public, max-age=3600'
                    }
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