from fastapi import FastAPI, Request, HTTPException
from transformers import pipeline
import torch
from uuid import uuid4

app = FastAPI()

# Use GPU if available; else fallback to CPU.
# Note: The Transformers pipeline expects an integer: 0 for GPU, -1 for CPU.
device = 0 if torch.cuda.is_available() else -1
model_name = "EleutherAI/gpt-neo-1.3B"
try:
    generator = pipeline("text-generation", model=model_name, device=device)
    print(f"Device set to use {'cuda' if device == 0 else 'cpu'}")
except Exception as e:
    print("Error loading model:", e)
    raise e

# In-memory game sessions storage.
# Each session stores conversation history and the current game stage.
game_sessions = {}

# System instructions for the game.
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
    Receives a JSON payload with the user's prompt and an optional session_id.
    Constructs a full prompt that includes system instructions, the current stage, and previous conversation history,
    then generates a response using a smarter model.
    If the generated text contains the trigger "Stage Complete", the session's stage is incremented.
    """
    try:
        data = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from e

    prompt = data.get("prompt", "")
    session_id = data.get("session_id", str(uuid4()))

    # Initialize session if not present.
    if session_id not in game_sessions:
        game_sessions[session_id] = {"history": [], "stage": 0}
    session = game_sessions[session_id]

    # Build the full prompt with system instructions, current stage header, and previous history.
    full_prompt = (
        SYSTEM_INSTRUCTIONS +
        f"\n\n[Stage {session['stage']}]\n" +
        (("Previous History:\n" + "\n".join(session["history"]) + "\n") if session["history"] else "") +
        f"User: {prompt}\nCharacter:"
    )

    try:
        result = generator(full_prompt, max_length=300, temperature=0.8, do_sample=True)[0]["generated_text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {e}") from e

    # Append the exchange to session history.
    session["history"].append(f"User: {prompt}")
    session["history"].append(f"Character: {result}")

    # Example trigger: if the generated text contains "Stage Complete", increment the stage.
    if "Stage Complete" in result:
        session["stage"] += 1

    return {
        "session_id": session_id,
        "stage": session["stage"],
        "response": result,
        "history": session["history"]  # For debugging purposes; remove in production if needed.
    }

# To run this FastAPI app with Gunicorn using Uvicorn workers, use:
# gunicorn -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:9000 app:app
