import os
from google.cloud import speech
import asyncio

# Existing imports...

async def transcribe_audio(audio_path: str, sample_rate: int = 24000) -> str:
    """
    Transcribe audio file to text, handling both short and long audio files.
    
    Args:
        audio_path: Path to the audio file
        sample_rate: Sample rate of the audio file in Hz
        
    Returns:
        str: Transcribed text
    """
    # Check file size or duration to decide which API to use
    file_size = os.path.getsize(audio_path)
    file_duration_estimate = file_size / (sample_rate * 2)  # Rough estimate (16-bit samples = 2 bytes per sample)
    
    # Read audio content
    with open(audio_path, "rb") as f:
        content = f.read()
    
    # Configure recognition settings
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code="en-US",
        enable_automatic_punctuation=True
    )
    
    # If file is likely longer than 60 seconds, use async API
    if file_duration_estimate > 55:  # Use 55 seconds as threshold to be safe
        logger.info(f"Long audio detected ({file_duration_estimate:.2f}s estimate), using async transcription")
        
        # Upload the file to Google Cloud Storage (GCS) if needed
        # For this example, we'll use in-memory approach with base64 encoding
        import base64
        
        # Create async recognition request
        audio_content_b64 = base64.b64encode(content).decode('utf-8')
        
        async_config = {
            "config": {
                "encoding": "LINEAR16",
                "sampleRateHertz": sample_rate,
                "languageCode": "en-US",
                "enableAutomaticPunctuation": True
            },
            "audio": {
                "content": audio_content_b64
            }
        }
        
        # Use the REST API for async recognition
        import requests
        import time
        import json
        
        # Get credentials
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_file(
            os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        )
        access_token = credentials.token
        
        # Create the request
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        # Send async request
        response = requests.post(
            "https://speech.googleapis.com/v1/speech:longrunningrecognize",
            headers=headers,
            data=json.dumps(async_config)
        )
        
        if response.status_code != 200:
            raise Exception(f"Async transcription request failed: {response.status_code} {response.text}")
        
        operation_name = response.json()["name"]
        
        # Poll for completion
        max_polls = 30
        for i in range(max_polls):
            poll_response = requests.get(
                f"https://speech.googleapis.com/v1/operations/{operation_name}",
                headers=headers
            )
            
            poll_data = poll_response.json()
            if "done" in poll_data and poll_data["done"]:
                # Operation is complete
                if "response" in poll_data and "results" in poll_data["response"]:
                    results = poll_data["response"]["results"]
                    transcript = " ".join([
                        result["alternatives"][0]["transcript"]
                        for result in results
                        if "alternatives" in result and len(result["alternatives"]) > 0
                    ])
                    return transcript
                else:
                    raise Exception(f"Unexpected response format: {poll_data}")
            
            # Wait before polling again
            await asyncio.sleep(2)
        
        raise Exception(f"Async transcription timed out after {max_polls * 2} seconds")
    
    # For shorter audio, use the synchronous API
    else:
        logger.info(f"Short audio detected ({file_duration_estimate:.2f}s estimate), using sync transcription")
        audio = speech.RecognitionAudio(content=content)
        response = stt_client.recognize(config=config, audio=audio)
        transcript = " ".join([res.alternatives[0].transcript for res in response.results])
        return transcript