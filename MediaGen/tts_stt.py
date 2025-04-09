import os
import base64
import requests
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.cloud import speech
import json
from cryptography.fernet import Fernet

# === LOAD ENCRYPTED SERVICE ACCOUNT KEY ===
def load_service_account_key():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    key_path = os.path.join(current_dir, 'dec.key')
    encrypted_json_path = os.path.join(current_dir, 'pyencrypted_key.json')

    # Load the secret key
    with open(key_path, 'rb') as key_file:
        key = key_file.read()

    # Decrypt JSON key file on-the-fly
    with open(encrypted_json_path, 'rb') as encrypted_file:
        encrypted_json = encrypted_file.read()

    decrypted_json_bytes = Fernet(key).decrypt(encrypted_json)
    decrypted_json = json.loads(decrypted_json_bytes.decode('utf-8'))

    return decrypted_json  # returns a dictionary object

# === CONFIG ===
SERVICE_ACCOUNT_INFO = load_service_account_key()
GOOGLE_TTS_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"

print("✅ Key decrypted successfully. Ready to use.")

# === INIT GOOGLE CREDENTIALS ===
google_credentials = service_account.Credentials.from_service_account_info(
    SERVICE_ACCOUNT_INFO, scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
google_credentials.refresh(Request())
access_token = google_credentials.token
stt_client = speech.SpeechClient(credentials=google_credentials)

# === TEXT TO SPEECH (TTS) ===
async def generate_audio_google(text: str, output_path: str):
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "X-Goog-User-Project": google_credentials.project_id
    }
    payload = {
        "input": {"text": text},
        "voice": {
            "languageCode": "en-US",
            "name": "en-US-Standard-C"  # Change voice if needed
        },
        "audioConfig": {"audioEncoding": "LINEAR16"}
    }
    response = requests.post(GOOGLE_TTS_URL, headers=headers, json=payload)
    if response.status_code == 200:
        audio_data = base64.b64decode(response.json()["audioContent"])
        with open(output_path, "wb") as f:
            f.write(audio_data)
    else:
        raise Exception(f"TTS generation failed: {response.status_code} {response.text}")

# === SPEECH TO TEXT (STT) ===
async def transcribe_audio(audio_path: str, sample_rate: int = 24000) -> str:
    with open(audio_path, "rb") as f:
        content = f.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code="en-US",  # English as default
        enable_automatic_punctuation=True
    )
    response = stt_client.recognize(config=config, audio=audio)
    transcript = " ".join([res.alternatives[0].transcript for res in response.results])
    return transcript