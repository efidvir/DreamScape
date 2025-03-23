let stage = 0;
const status = document.getElementById('status');
const video = document.getElementById('bg-video');
const audio = document.getElementById('bg-audio');

async function sendText() {
    const input = document.getElementById("textInput").value;
    if (!input) return;

    status.innerText = "Thinking...";

    const res = await fetch("/api/interact", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            input,
            stage,
            timestamp: Date.now()
        })
    });

    const data = await res.json();

    if (data.videoUrl) {
        video.src = data.videoUrl;
        video.play();
    }

    if (data.audioUrl) {
        audio.src = data.audioUrl;
        audio.play();
    }

    if (data.nextStage !== undefined) {
        stage = data.nextStage;
    }

    status.innerText = "";
}

// Speech recognition
let recognition;
let isListening = false;

if ('webkitSpeechRecognition' in window) {
    recognition = new webkitSpeechRecognition();
    recognition.lang = 'en-US';
    recognition.onresult = event => {
        const transcript = event.results[0][0].transcript;
        document.getElementById("textInput").value = transcript;
        sendText();
    };
}

function toggleVoice() {
    if (!recognition) return;
    if (isListening) {
        recognition.stop();
        isListening = false
