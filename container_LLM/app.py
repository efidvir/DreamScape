import json
import os
from fastapi import FastAPI, Request, HTTPException
from transformers import pipeline, AutoTokenizer
import torch
from uuid import uuid4

app = FastAPI()

# Use GPU if available; else fallback to CPU.
# The pipeline expects 0 for GPU and -1 for CPU.
device = 0 if torch.cuda.is_available() else -1
model_name = "EleutherAI/gpt-neo-1.3B"
try:
    generator = pipeline("text-generation", model=model_name, device=device)
    print(f"Device set to use {'cuda' if device == 0 else 'cpu'}")
except Exception as e:
    print("Error loading model:", e)
    raise e

# Initialize the tokenizer for our model.
tokenizer = AutoTokenizer.from_pretrained(model_name)

# In-memory game sessions storage.
# Each session stores conversation history and the current game stage.
game_sessions = {}

# Path to the game progress file.
GAME_PROGRESS_FILE = "game_progress.json"

def load_game_progress():
    """Load the game progress (stages, goals, missing details) from file."""
    if not os.path.exists(GAME_PROGRESS_FILE):
        # If the file doesn't exist, create a default structure.
        default_progress = {
            "stages": {
                "0": {
                    "instructions": "You are in darkness with no memory of your identity. Your goal is to discover your name and origin.",
                    "missing": ["name", "origin"]
                }
            }
        }
        with open(GAME_PROGRESS_FILE, "w") as f:
            json.dump(default_progress, f, indent=2)
        return default_progress
    else:
        with open(GAME_PROGRESS_FILE, "r") as f:
            return json.load(f)

def update_game_progress(stage, filled_details):
    """
    Optionally update the game progress file with new details filled by the user.
    This function can be expanded as needed.
    """
    progress = load_game_progress()
    stage_str = str(stage)
    if stage_str in progress.get("stages", {}):
        # For example, we append filled details to a new key.
        progress["stages"][stage_str]["filled"] = filled_details
        with open(GAME_PROGRESS_FILE, "w") as f:
            json.dump(progress, f, indent=2)

# Fixed system instructions remain the base narrative.
SYSTEM_INSTRUCTIONS = (
    "You are a character in a game. You are in darkness, and you don't know who you are, "
    "where you are, or anything else whatsoever. Your purpose is to interact with the user to obtain information about everything. "
    "As the user answers, missing details are slowly filled in and memorized. These missing details are unique to each stage of the game. "
    "When the missing details for a stage are completely filled in from the user's input, that stage ends and a new stage with new missing details begins. "
    "All conversation history is kept in memory and used for context and logic in your responses."
)

@app.get("/healthz")
def healthz():
    """Health-check endpoint for the LLM container."""
    return {"status": "OK"}

@app.post("/generate")
async def generate(request: Request):
    """
    Receives a JSON payload with 'prompt' and an optional 'session_id'.
    Constructs a prompt with the fixed system instructions, stage-specific instructions,
    conversation history, and the user prompt. It then instructs the LLM to return a JSON
    object with the following keys (all values must be non-empty plain strings):
      - user_response: The text to display.
      - voice_response: A version optimized for text-to-speech.
      - stage_decision: "advance" if the stage should progress, or empty otherwise.
      - video_scene: A description for a visual scene.
      - debug_info: Any extra debugging information (empty if none).
    """
    try:
        data = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from e

    prompt = data.get("prompt", "") or "Hello, how are you?"
    print(f"Received prompt: {prompt}")
    session_id = data.get("session_id", str(uuid4()))

    if session_id not in game_sessions:
        game_sessions[session_id] = {"history": [], "stage": 0}
    session = game_sessions[session_id]

    progress = load_game_progress()
    current_stage = str(session["stage"])
    stage_config = progress.get("stages", {}).get(current_stage, {})
    stage_instructions = stage_config.get("instructions", "")
    stage_missing = stage_config.get("missing", [])

    # Revised prompt with explicit instructions and placeholders.
    full_prompt = (
        SYSTEM_INSTRUCTIONS +
        f"\n\n[Stage {session['stage']}]\n" +
        f"Stage Instructions: {stage_instructions}\n" +
        f"Missing Details: {', '.join(stage_missing)}\n" +
        (("Previous History:\n" + "\n".join(session['history']) + "\n") if session['history'] else "") +
        f"User: {prompt}\n" +
        "Character: Based on the above, generate a fresh, engaging, and creative response. "
        "Fill in the JSON template below by replacing the placeholder text with meaningful, non-empty, single-line responses. "
        "Do NOT output any additional text or commentary, and do not echo back the instructions. "
        "Your response must be a completed JSON object with exactly the following keys:\n"
        "  \"user_response\": The clear and engaging response to the user's input.\n"
        "  \"voice_response\": A version optimized for text-to-speech with natural phrasing.\n"
        "  \"stage_decision\": Enter \"advance\" if the stage should progress, or leave empty otherwise.\n"
        "  \"video_scene\": A description of a visual scene that fits the current conversation.\n"
        "  \"debug_info\": Any additional debugging details; leave empty if none.\n\n"
        "Return ONLY the JSON object between the markers below.\n\n"
        "<<<JSON_START>>>\n"
        "{\n"
        "  \"user_response\": \"<Your engaging response here>\",\n"
        "  \"voice_response\": \"<Your TTS optimized response here>\",\n"
        "  \"stage_decision\": \"<advance or leave empty>\",\n"
        "  \"video_scene\": \"<Description of a visual scene>\",\n"
        "  \"debug_info\": \"<Any debug info, leave empty if none>\"\n"
        "}\n"
        "<<<JSON_END>>>\n"
        "For example, a correct output might look like:\n"
        "<<<JSON_START>>>\n"
        "{\n"
        "  \"user_response\": \"I am here, ready to discover my identity.\",\n"
        "  \"voice_response\": \"I am here, ready to discover my identity.\",\n"
        "  \"stage_decision\": \"advance\",\n"
        "  \"video_scene\": \"A dark room gradually illuminated by a single beam of light.\",\n"
        "  \"debug_info\": \"\"\n"
        "}\n"
        "<<<JSON_END>>>\n"
    )

    # Use the tokenizer with explicit truncation.
    input_ids = tokenizer(full_prompt, return_tensors="pt", truncation=True).input_ids
    input_length = input_ids.shape[1]
    new_tokens = 100  # Adjust as needed
    max_length_total = input_length + new_tokens

    try:
        result = generator(
            full_prompt,
            max_length=max_length_total,
            temperature=0.8,
            do_sample=True
        )[0]["generated_text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {e}") from e

    # Extract the JSON object between the markers.
    try:
        start_marker = "<<<JSON_START>>>"
        end_marker = "<<<JSON_END>>>"
        start = result.find(start_marker)
        end = result.find(end_marker, start)
        if start != -1 and end != -1:
            json_text = result[start + len(start_marker):end].strip()
        else:
            json_text = result.strip()
        structured_response = json.loads(json_text)
    except Exception as e:
        structured_response = {
            "user_response": result.strip(),
            "voice_response": "",
            "stage_decision": "",
            "video_scene": "",
            "debug_info": f"Error parsing structured output: {e}"
        }

    session["history"].append(f"User: {prompt}")
    session["history"].append(f"Character: {json.dumps(structured_response)}")

    if structured_response.get("stage_decision", "").lower() == "advance":
        session["stage"] += 1
        filled_info = structured_response.get("debug_info", "")
        update_game_progress(session["stage"] - 1, filled_info)

    return {
        "session_id": session_id,
        "stage": session["stage"],
        "user_response": structured_response.get("user_response", ""),
        "voice_response": structured_response.get("voice_response", ""),
        "stage_decision": structured_response.get("stage_decision", ""),
        "video_scene": structured_response.get("video_scene", ""),
        "debug_info": structured_response.get("debug_info", ""),
        "history": session["history"]  # For debugging; remove in production if desired.
    }
