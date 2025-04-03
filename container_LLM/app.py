# container_LLM/app.py - Complete code with input cleaning and hybrid approach
import json
import os
import gc
import time
import traceback
import logging
import sys
import random
import re
import unicodedata
import asyncio
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request, BackgroundTasks
import torch
from uuid import uuid4

# Configure more visible logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Ensure logs go to stdout for container logs
    ]
)
logger = logging.getLogger("llm-container")

# Function for highly visible log messages
def log_banner(message: str, char: str = "=") -> None:
    """Print a highly visible banner message to the logs."""
    banner = char * 80
    spacer = char + " " * 78 + char
    
    # Calculate centering
    lines = message.split('\n')
    centered_lines = []
    for line in lines:
        padding = (76 - len(line)) // 2
        centered_lines.append(char + " " * padding + line + " " * (76 - len(line) - padding) + char)
    
    # Print the banner
    logger.info(banner)
    logger.info(spacer)
    for line in centered_lines:
        logger.info(line)
    logger.info(spacer)
    logger.info(banner)

app = FastAPI()

# Print startup banner
log_banner("LLM CONTAINER STARTING", "=")
log_banner(f"Python {sys.version}\nPyTorch {torch.__version__}\nCUDA {'Available' if torch.cuda.is_available() else 'Not Available'}", "-")

# Directory for offloaded model weights
OFFLOAD_DIR = "/offload"
os.makedirs(OFFLOAD_DIR, exist_ok=True)

# Directory for pre-downloaded models (from build)
DOWNLOADED_MODELS_DIR = "/downloaded_models"

# Path to the game progress file
GAME_PROGRESS_FILE = "game_progress.json"

# In-memory game sessions storage
game_sessions = {}

# Indicate if model is already loading (to prevent concurrent loads)
is_model_loading = False
model_loaded = False

# Model and tokenizer global variables
model = None
tokenizer = None

# Progress tracking for model loading
loading_stage = "Not started"
loading_progress = 0

# Open-access model to use
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Add function to clean input text of invisible characters
def clean_input_text(text):
    """
    Clean the input text by removing invisible control characters
    and normalizing whitespace.
    
    Args:
        text: The input text to clean
        
    Returns:
        str: The cleaned text
    """
    # Remove invisible unicode control characters
    cleaned = ""
    for char in text:
        # Skip control characters and invisible markers
        category = unicodedata.category(char)
        if category.startswith('C') or category == 'Cf':  # Control chars and format chars
            continue
        cleaned += char
    
    # Normalize whitespace
    cleaned = " ".join(cleaned.split())
    
    # Log what was cleaned if anything changed
    if cleaned != text:
        logger.info(f"Cleaned input text. Original length: {len(text)}, Cleaned length: {len(cleaned)}")
    
    return cleaned

# Add language detection function
def detect_language(text):
    """Detect the language of the input text based on character ranges."""
    # Hebrew characters range
    if re.search(r'[\u0590-\u05FF]', text):
        return "hebrew"
    # Arabic characters range
    elif re.search(r'[\u0600-\u06FF]', text):
        return "arabic"
    # Cyrillic characters range (Russian, etc.)
    elif re.search(r'[\u0400-\u04FF]', text):
        return "cyrillic"
    # Chinese characters range
    elif re.search(r'[\u4E00-\u9FFF]', text):
        return "chinese"
    # Japanese characters range (includes Hiragana, Katakana)
    elif re.search(r'[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF]', text):
        return "japanese"
    # Korean characters range
    elif re.search(r'[\uAC00-\uD7AF]', text):
        return "korean"
    # Default to English/Latin
    else:
        return "latin"

# Add language-specific fallback responses
LANGUAGE_FALLBACKS = {
    "hebrew": [
        "אני חש את נוכחותך בחשכה. ספר לי עוד על עצמך.",
        "החושך מקיף אותי לחלוטין. אני רק שומע את קולך. מי אתה?",
        "הזכרונות שלי נעלמו. כל מה שאני יודע הוא החושך הזה. האם תוכל לעזור לי?",
        "אני לכוד בצללים, לא יכול לראות או לזכור. האם באת לעזור לי?"
    ],
    # Add other languages as needed
}

# English fallback responses
FALLBACK_RESPONSES = [
    "I can sense your presence in the darkness. Tell me more about yourself.",
    "The darkness surrounds me completely. I can only hear your voice. Who are you?",
    "I feel lost in this void. Your words are my only connection to... anything.",
    "My memories are gone. All I know is this darkness. Can you help me?",
    "Something about your voice feels familiar, yet I can't place it. Please continue.",
    "I'm trapped in shadows, unable to see or remember. Are you here to help me?",
    "The void seems endless. Your voice is the only thing I can hold onto.",
    "I'm searching for any clue about who I am. Do you know me?",
    "This darkness... it's all I know. Your voice gives me hope.",
    "I feel like I've been here forever, waiting. Are you real?"
]

# Themed responses for rule-based approach
IDENTITY_RESPONSES = [
    "I wish I could tell you who I am, but my memories are gone. There's only darkness.",
    "My identity? That's what I'm trying to figure out. All I know is this emptiness around me.",
    "I can't remember my name or who I was. It's like my past has been swallowed by this darkness.",
    "If I knew who I was, perhaps I could understand why I'm trapped here in this void.",
    "I've searched the corners of my mind, but there's nothing - no name, no face, no past."
]

LOCATION_RESPONSES = [
    "I don't know where this place is. It's just... darkness, extending in all directions.",
    "This void has no boundaries that I can find. It's a nothingness that seems to go on forever.",
    "I've tried to move, to find walls or doors, but there's nothing. Just endless dark.",
    "The darkness feels almost alive, like it's watching me. But I can't see anything beyond it.",
    "I can't tell if this place is real or just in my mind. The darkness is absolute."
]

HELP_RESPONSES = [
    "Yes, please help me! I need to discover who I am and how to escape this darkness.",
    "I don't know how anyone could help me, but I'm desperate for a way out of this void.",
    "If you can help me remember who I am, perhaps I can find a way through this darkness.",
    "Your voice might be my only salvation. Please, keep talking to me.",
    "I feel so helpless here. Any guidance you can offer would mean everything."
]

RELATION_RESPONSES = [
    "Have we met before? I feel like your voice stirs something in my memories, but I can't grasp it.",
    "I don't know if we're connected, but you're the only presence I can sense in this void.",
    "I wish I could remember if I know you. This amnesia is torturous - familiar voices with no faces or names.",
    "There's something about your presence that feels significant. Were we important to each other?",
    "The way you speak to me... it feels like you know me. Do I know you too?"
]

# System instructions for the character
SYSTEM_INSTRUCTIONS = (
    "You are a character in a game. You are in darkness, and you don't know who you are, "
    "where you are, or anything else whatsoever. Your purpose is to interact with the user to obtain information about everything. "
    "As the user answers, missing details are slowly filled in and memorized."
)

def check_model_cache():
    """Check if the model is already downloaded and available."""
    # Check for model files in the downloaded_models directory
    model_dir = os.path.join(DOWNLOADED_MODELS_DIR, MODEL_NAME.replace('/', '--'))
    config_file = os.path.join(model_dir, 'config.json')
    
    if os.path.exists(config_file):
        logger.info(f"Found pre-downloaded model at {model_dir}")
        return True
    else:
        logger.info(f"Model not found in cache at {model_dir}")
        return False

def load_game_progress():
    """Load the game progress (stages, goals, missing details) from file."""
    try:
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
    except Exception as e:
        logger.error(f"Error loading game progress: {e}")
        return {
            "stages": {
                "0": {
                    "instructions": "You are in darkness with no memory of your identity.",
                    "missing": ["identity"]
                }
            }
        }

def update_game_progress(stage, filled_details):
    """Update the game progress file with new details filled by the user."""
    try:
        progress = load_game_progress()
        stage_str = str(stage)
        if stage_str in progress.get("stages", {}):
            progress["stages"][stage_str]["filled"] = filled_details
            with open(GAME_PROGRESS_FILE, "w") as f:
                json.dump(progress, f, indent=2)
    except Exception as e:
        logger.error(f"Error updating game progress: {e}")

def update_loading_status(stage: str, progress: int) -> None:
    """Update the model loading status with a visible log message."""
    global loading_stage, loading_progress
    loading_stage = stage
    loading_progress = progress
    
    # Create a progress bar string
    progress_bar = "█" * (progress // 5) + "░" * ((100 - progress) // 5)
    
    # Log with high visibility
    log_banner(f"MODEL LOADING: {stage}\nProgress: [{progress_bar}] {progress}%", "*")

async def load_model_async():
    """
    Load an open-source language model with optimizations.
    This function is designed to run in the background with visible progress updates.
    
    Will check for pre-downloaded model first, and use it if available.
    """
    global model, tokenizer, is_model_loading, model_loaded, MODEL_NAME
    
    if model_loaded:
        logger.info("Model already loaded, skipping load")
        return
    
    if is_model_loading:
        logger.info("Model is already loading in another task")
        return
    
    is_model_loading = True
    
    try:
        # Check if model is already downloaded
        model_cached = check_model_cache()
        
        log_banner(f"STARTING MODEL LOADING PROCESS\nModel: {MODEL_NAME}\nPre-downloaded: {model_cached}", "*")
        update_loading_status("Initializing model load", 0)
        
        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"CUDA available: {torch.cuda.is_available()}, Memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        # Import necessary libraries
        update_loading_status("Importing libraries", 10)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Prepare loading parameters
        local_files_only = model_cached  # Only use local files if model is cached
        cache_dir = DOWNLOADED_MODELS_DIR if model_cached else None
        
        # Load tokenizer first
        update_loading_status("Loading tokenizer", 20)
        logger.info(f"Loading tokenizer for {MODEL_NAME} (local_files_only={local_files_only})")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, 
            cache_dir=cache_dir, 
            local_files_only=local_files_only
        )
        update_loading_status("Tokenizer loaded successfully", 30)
        
        # Configure model loading
        update_loading_status("Configuring model parameters", 40)
        
        # Determine device map based on available resources
        device_map = None
        load_in_8bit = False
        torch_dtype = None
        
        if torch.cuda.is_available():
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # in GB
            logger.info(f"GPU memory: {gpu_memory:.2f} GB")
            
            if gpu_memory > 4:  # If we have decent GPU memory
                device_map = "auto"
                torch_dtype = torch.float16
                logger.info("Using auto device mapping and float16 precision")
            else:
                # For smaller GPUs, use 8-bit quantization
                load_in_8bit = True
                logger.info("Using 8-bit quantization for limited GPU memory")
        else:
            logger.info("CUDA not available, using CPU")
        
        # Load model with memory optimization
        update_loading_status(f"{'Loading pre-downloaded' if model_cached else 'Downloading'} model", 50)
        log_banner(f"{'LOADING PRE-DOWNLOADED' if model_cached else 'DOWNLOADING AND LOADING'} MODEL: {MODEL_NAME}", "*")
        
        # Import libraries for quantization if needed
        if load_in_8bit:
            try:
                import bitsandbytes as bnb
                logger.info("Using bitsandbytes for 8-bit quantization")
                update_loading_status("Preparing 8-bit quantization", 60)
            except ImportError:
                logger.warning("bitsandbytes not available, falling back to standard loading")
                load_in_8bit = False
        
        # Load the model with appropriate optimizations
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map=device_map,
            load_in_8bit=load_in_8bit, 
            torch_dtype=torch_dtype,
            offload_folder=OFFLOAD_DIR if device_map == "auto" else None,
            offload_state_dict=True if device_map == "auto" else False,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
            trust_remote_code=True,      # Add this parameter
            revision="main",             # Add this parameter
            attn_implementation="eager"  # Add this parameter
        )
        
        update_loading_status("Model loaded, finalizing setup", 90)
        
        # Set model to evaluation mode
        model.eval()
        
        update_loading_status("Model loading complete!", 100)
        log_banner(f"MODEL LOADED SUCCESSFULLY: {MODEL_NAME}", "*")
        model_loaded = True
        
        # Report memory usage
        if torch.cuda.is_available():
            logger.info(f"CUDA Memory allocated after model load: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    except Exception as e:
        error_message = f"Error loading model: {e}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        log_banner(f"MODEL LOADING FAILED\n{str(e)}", "!")
    finally:
        is_model_loading = False

def generate_text(prompt: str, max_new_tokens: int = 100) -> str:
    """
    Generate text with memory-optimized inference.
    
    Args:
        prompt: Input text prompt
        max_new_tokens: Maximum number of new tokens to generate
        
    Returns:
        str: Generated text response
    """
    global model, tokenizer
    
    if model is None or tokenizer is None:
        logger.warning("Model or tokenizer not loaded, returning fallback")
        return ""
    
    try:
        # Log generation start
        logger.info(f"Generating text for prompt of length: {len(prompt)}")
        
        # Ensure prompt is clean of control characters
        clean_prompt = clean_input_text(prompt)
        if clean_prompt != prompt:
            logger.info("Cleaned prompt before generation")
            prompt = clean_prompt
        
        # Encode the prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        
        # Move input to the right device
        if hasattr(model, "device"):
            input_ids = input_ids.to(model.device)
        
        # Clear GPU memory cache before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Generate with careful management of memory
        with torch.no_grad():
            # Configure generation parameters for better quality while managing memory
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,  # Add this parameter
                num_return_sequences=1,
            )
        
        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generation complete, response length: {len(response)}")
        
        # Extract just the newly generated content
        if len(response) > len(prompt) and response.startswith(prompt):
            generated_text = response[len(prompt):].strip()
        else:
            generated_text = response.strip()
            
        return generated_text
        
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        logger.error(traceback.format_exc())
        return ""
    finally:
        # Clean up memory after generation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def get_smart_response(prompt, history, stage_config):
    """
    Get a smart response using a hybrid approach:
    1. Try using the model if it's loaded
    2. Fall back to rule-based responses
    3. Ensure consistent quality
    
    Args:
        prompt: The user's input text
        history: Conversation history
        stage_config: Current stage configuration
        
    Returns:
        str: A response text
    """
    global model_loaded, model, tokenizer
    
    # Extract stage info
    stage_instructions = stage_config.get("instructions", "")
    stage_missing = stage_config.get("missing", [])
    
    # Detect language
    detected_language = detect_language(prompt)
    
    # For non-Latin scripts, use language-specific responses
    if detected_language != "latin":
        if detected_language in LANGUAGE_FALLBACKS:
            return random.choice(LANGUAGE_FALLBACKS[detected_language])
        return random.choice(FALLBACK_RESPONSES)
    
    # Try the model first if it's loaded
    response_text = ""
    if model_loaded and model is not None and tokenizer is not None:
        try:
            # Prepare context from history
            context = ""
            if history:
                recent_history = history[-6:]  # Last 3 exchanges
                context = "\n".join(recent_history) + "\n"
            
            # Create a streamlined prompt for better reliability
            llm_prompt = (
                f"You're a character lost in darkness with no memory of your identity. "
                f"Your goal is to {stage_instructions}\n"
                f"{context}"
                f"User: {prompt}\n"
                f"Your response (brief, 1-2 sentences):"
            )
            
            # Try generating with the model
            response_text = generate_text(llm_prompt, max_new_tokens=50)
            
            # Clean up the response
            if response_text:
                # Remove any dialogue markers
                dialogue_markers = ["User:", "Character:", "Your response:"]
                for marker in dialogue_markers:
                    if marker in response_text:
                        response_text = response_text.split(marker)[0].strip()
                
                # Limit to first two sentences if long
                sentences = response_text.split('.')
                if len(sentences) > 2 and len(response_text) > 100:
                    response_text = '.'.join(sentences[:2]).strip() + '.'
        except Exception as e:
            logger.error(f"Error generating with model: {e}")
            response_text = ""
    
    # If model failed or didn't generate adequate response, use rule-based approach
    if not response_text or len(response_text) < 10:
        # Key themes in the prompt
        identity_keywords = ["who", "you", "your", "name", "identity", "yourself", "remember", "memory"]
        location_keywords = ["where", "place", "location", "here", "darkness", "void", "trapped"]
        help_keywords = ["help", "save", "rescue", "free", "escape", "assist"]
        relation_keywords = ["know you", "met", "before", "friend", "together", "relationship", "connected"]
        
        # Check which theme is most relevant
        identity_score = sum(1 for word in identity_keywords if word.lower() in prompt.lower())
        location_score = sum(1 for word in location_keywords if word.lower() in prompt.lower())
        help_score = sum(1 for word in help_keywords if word.lower() in prompt.lower())
        relation_score = sum(1 for word in relation_keywords if word.lower() in prompt.lower())
        
        # Select response based on theme
        if identity_score > location_score and identity_score > help_score and identity_score > relation_score:
            response_text = random.choice(IDENTITY_RESPONSES)
        elif location_score > identity_score and location_score > help_score and location_score > relation_score:
            response_text = random.choice(LOCATION_RESPONSES)
        elif help_score > identity_score and help_score > location_score and help_score > relation_score:
            response_text = random.choice(HELP_RESPONSES)
        elif relation_score > identity_score and relation_score > location_score and relation_score > help_score:
            response_text = random.choice(RELATION_RESPONSES)
        else:
            # If no clear theme, use general responses
            response_text = random.choice(FALLBACK_RESPONSES)
    
    return response_text

@app.on_event("startup")
async def startup_event():
    """Start loading the model in the background when the app starts."""
    log_banner("APPLICATION STARTUP - BEGINNING MODEL LOAD IN BACKGROUND", "*")
    background_tasks = BackgroundTasks()
    background_tasks.add_task(load_model_async)
    await background_tasks()

@app.get("/healthz")
def healthz():
    """Health-check endpoint for the LLM container."""
    status = {
        "status": "OK", 
        "model_loaded": model_loaded,
        "loading_status": {
            "is_loading": is_model_loading,
            "stage": loading_stage,
            "progress": loading_progress
        }
    }
    
    if not model_loaded and is_model_loading:
        logger.info(f"Health check: Model still loading - {loading_stage} ({loading_progress}%)")
    
    return status

@app.post("/generate")
async def generate(request: Request, background_tasks: BackgroundTasks):
    """
    Process user input and generate a response.
    If the model isn't loaded yet, queue it up and use fallbacks meanwhile.
    Now with support for detecting non-Latin script and providing appropriate fallbacks.
    """
    start_time = time.time()
    
    try:
        # If model isn't loaded yet, start loading it
        if not model_loaded and not is_model_loading:
            logger.info("Model not loaded, starting background load")
            background_tasks.add_task(load_model_async)
        
        # Parse request data
        data = await request.json()
        raw_prompt = data.get("prompt", "")
        if not raw_prompt:
            raw_prompt = "Hello, is anyone there?"
        
        # Clean the input text of any invisible control characters
        prompt = clean_input_text(raw_prompt)
        if prompt != raw_prompt:
            logger.info(f"Cleaned input from '{raw_prompt}' to '{prompt}'")
        else:
            logger.info(f"Received prompt: '{prompt}'")
        
        session_id = data.get("session_id", str(uuid4()))
        
        # Get or create session
        if session_id not in game_sessions:
            game_sessions[session_id] = {"history": [], "stage": 0}
        session = game_sessions[session_id]
        
        # Load game progress
        progress = load_game_progress()
        current_stage = str(session["stage"])
        stage_config = progress.get("stages", {}).get(current_stage, {})
        stage_instructions = stage_config.get("instructions", "")
        stage_missing = stage_config.get("missing", [])
        
        # Get the language
        detected_language = detect_language(prompt)
        logger.info(f"Detected language: {detected_language}")
        
        # Get response using our hybrid approach
        response_text = get_smart_response(prompt, session["history"], stage_config)
        logger.info(f"Smart response generated: '{response_text}'")
        
        # Get loading status message for debug info
        if not model_loaded and is_model_loading:
            model_status = f"Model loading: {loading_stage} ({loading_progress}%)"
        elif not model_loaded:
            model_status = "Model not loaded yet"
        else:
            model_status = f"Model loaded, Language: {detected_language}"
        
        # Construct the response object
        structured_response = {
            "user_response": response_text,
            "voice_response": response_text,
            "stage_decision": "",
            "video_scene": "A mysterious figure emerges slightly from the darkness.",
            "debug_info": f"{model_status}, Time: {time.time() - start_time:.2f}s"
        }
        
        # Check for stage advancement
        if stage_missing and all(detail.lower() in (prompt.lower() + " " + response_text.lower()) for detail in stage_missing):
            structured_response["stage_decision"] = "advance"
        
        # Update session history
        session["history"].append(f"User: {prompt}")
        session["history"].append(f"Character: {response_text}")
        
        # Handle stage advancement
        if structured_response.get("stage_decision", "").lower() == "advance":
            session["stage"] += 1
            update_game_progress(session["stage"] - 1, "Advanced based on conversation")
        
        # Return the final response
        return {
            "session_id": session_id,
            "stage": session["stage"],
            "user_response": structured_response["user_response"],
            "voice_response": structured_response["voice_response"],
            "stage_decision": structured_response["stage_decision"],
            "video_scene": structured_response["video_scene"],
            "debug_info": structured_response["debug_info"],
            "history": session["history"][-6:]  # Return only recent history
        }
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        logger.error(traceback.format_exc())
        
        # Return a fallback response
        return {
            "session_id": "error_session",
            "stage": 0,
            "user_response": "I sense your presence, but I'm having trouble responding. The darkness seems to be affecting my thoughts.",
            "voice_response": "I sense your presence, but I'm having trouble responding. The darkness seems to be affecting my thoughts.",
            "stage_decision": "",
            "video_scene": "A figure stands motionless in the shadows.",
            "debug_info": f"Error: {str(e)}",
            "history": []
        }