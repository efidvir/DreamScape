// Global AudioContext re-used for all operations
let globalAudioContext = null;
function getAudioContext() {
  if (!globalAudioContext) {
    globalAudioContext = new (window.AudioContext || window.webkitAudioContext)();
    console.log("Global AudioContext created.");
  }
  return globalAudioContext;
}

// Resume AudioContext on user gesture
document.addEventListener("click", () => {
  const ctx = getAudioContext();
  if (ctx.state === "suspended") {
    ctx.resume().then(() => {
      console.log("Global AudioContext resumed on user gesture.");
    });
  }
});

// === Global Variables for Waveform Visualization ===
let audioContext; // will use globalAudioContext
let analyser;
let source;
let animationId;
let currentMode = null; // "mic" for recording, "transcript" flow after recording
let waveformColor = "green"; // green when recording; purple during TTS playback

// Canvas setup
const canvas = document.getElementById("waveCanvas");
const ctx = canvas.getContext("2d");
function resizeCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}
window.addEventListener("resize", resizeCanvas);
resizeCanvas();

function drawWaveform() {
  animationId = requestAnimationFrame(drawWaveform);
  if (!analyser) return;
  const bufferLength = analyser.fftSize;
  const dataArray = new Uint8Array(bufferLength);
  analyser.getByteTimeDomainData(dataArray);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 2;
  ctx.shadowBlur = 20;
  // Use purple for TTS playback, green for mic recording.
  ctx.shadowColor = waveformColor === "purple" 
    ? "rgba(160,32,240,0.6)" 
    : "rgba(0,255,0,0.6)";
  ctx.strokeStyle = waveformColor === "purple" 
    ? "rgba(160,32,240,0.8)" 
    : "rgba(0,255,0,0.8)";
  ctx.beginPath();
  const sliceWidth = canvas.width / analyser.fftSize;
  let x = 0;
  for (let i = 0; i < analyser.fftSize; i++) {
    const v = dataArray[i] / 128.0;
    const y = (v * canvas.height) / 2;
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
    x += sliceWidth;
  }
  ctx.lineTo(canvas.width, canvas.height / 2);
  ctx.stroke();
}

function startMicVisualizationFromStream(stream) {
  audioContext = getAudioContext();
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 2048;
  source = audioContext.createMediaStreamSource(stream);
  source.connect(analyser);
  currentMode = "mic";
  waveformColor = "green";
  drawWaveform();
}

function cleanupAudio() {
  try {
    if (source) source.disconnect();
    if (analyser) analyser.disconnect();
  } catch (e) {
    console.error("Error during cleanup:", e);
  }
  currentMode = null;
}
function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// Append a chat bubble into the chat container (#chatMessages)
function appendChatBubble(sender, text) {
  const chatContainer = document.getElementById("chatMessages");
  if (!chatContainer) {
    console.warn("Chat container not found!");
    return;
  }
  const bubble = document.createElement("div");
  bubble.classList.add("chat-bubble", sender); // "user" or "llm"
  bubble.textContent = text;
  chatContainer.appendChild(bubble);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

// === Continuous Voice Chat with Silence Detection ===
let webrtcId = Math.random().toString(36).substring(2, 10);
let audioChunksForFastRTC = [];
let recordingActive = false;
let silenceThreshold = 0.05; // Adjust threshold as needed
let silenceTimeout = 2000;   // ms of silence before sending audio
let silenceTimer = null;
let mediaRecorderRTC;
let micStreamRTC;

async function startFastRTCRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({ 
    audio: { sampleRate: 44100, channelCount: 1 } 
  });
  micStreamRTC = stream;
  recordingActive = true;
  audioChunksForFastRTC = [];
  mediaRecorderRTC = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });
  mediaRecorderRTC.ondataavailable = (event) => {
    if (event.data.size > 0) {
      audioChunksForFastRTC.push(event.data);
    }
  };
  startMicVisualizationFromStream(stream);
  
  const silenceAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const silenceAnalyser = silenceAudioCtx.createAnalyser();
  const silenceSource = silenceAudioCtx.createMediaStreamSource(stream);
  silenceSource.connect(silenceAnalyser);
  silenceAnalyser.fftSize = 2048;
  const dataArrayRTC = new Uint8Array(silenceAnalyser.fftSize);
  
  function checkSilence() {
    silenceAnalyser.getByteTimeDomainData(dataArrayRTC);
    let sum = 0;
    for (let i = 0; i < dataArrayRTC.length; i++) {
      sum += Math.abs(dataArrayRTC[i] - 128);
    }
    let avg = sum / dataArrayRTC.length / 128;
    if (avg < silenceThreshold) {
      if (!silenceTimer) {
        console.log("Silence detected, will send audio after timeout...");
        silenceTimer = setTimeout(() => {
          sendAudioToFastRTC();
          silenceTimer = null;
        }, silenceTimeout);
      }
    } else {
      if (silenceTimer) {
        console.log("User resumed speaking, clearing silence timer.");
        clearTimeout(silenceTimer);
        silenceTimer = null;
      }
    }
  }
  setInterval(checkSilence, 100);
  mediaRecorderRTC.start();
}

async function sendAudioToFastRTC() {
  if (audioChunksForFastRTC.length === 0) return;
  mediaRecorderRTC.stop();
  micStreamRTC.getTracks().forEach(track => track.stop());
  recordingActive = false;
  const mimeType = mediaRecorderRTC.mimeType || "audio/webm";
  const audioBlob = new Blob(audioChunksForFastRTC, { type: mimeType });
  console.log("Recorded audio blob size:", audioBlob.size, "MIME:", mimeType);
  audioChunksForFastRTC = [];
  const formData = new FormData();
  formData.append("audio", audioBlob);
  formData.append("webrtc_id", webrtcId);
  
  try {
    // Send the recorded audio to /input_hook.
    const response = await fetch("/input_hook", {
      method: "POST",
      body: formData
    });
    if (!response.ok) {
      throw new Error(`Server error: ${response.statusText}`);
    }
    const data = await response.json();
    console.log("Response received:", data);
    
    // Append chat bubbles: one for user transcript, one for LLM response.
    appendChatBubble("user", data.transcript);
    appendChatBubble("llm", data.llm_response);
    
    // Optionally, display the transcript in a popup.
    displayTranscript(data.transcript);
    
    // If a TTS URL is provided, play it.
    if (data.tts_audio_url) {
      playTTSAudio(data.tts_audio_url);
    }
  } catch (e) {
    console.error("Error sending audio for transcript:", e);
  }
}

// Play TTS audio and trigger purple waveform animation during playback.
function playTTSAudio(url) {
  const audio = new Audio(url);
  waveformColor = "purple"; // Switch to purple during playback
  
  audio.addEventListener("ended", () => {
    waveformColor = "green"; // Revert to green after playback ends
  });
  audio.play().catch((e) => {
    console.error("Error playing TTS audio:", e);
    waveformColor = "green";
  });
}

// Optional: Popup for transcript display.
function displayTranscript(transcript) {
  console.log("Transcript:", transcript);
  // You could implement a popup here if desired.
}

// === Mic Toggle Control ===
document.addEventListener("DOMContentLoaded", () => {
  const micCheckbox = document.querySelector(".mic-toggle input");
  micCheckbox.addEventListener("change", () => {
    if (micCheckbox.checked) {
      if (!recordingActive) {
        startFastRTCRecording();
      }
    } else {
      if (recordingActive) {
        mediaRecorderRTC.stop();
        micStreamRTC.getTracks().forEach(track => track.stop());
        recordingActive = false;
      }
    }
  });
  if (micCheckbox.checked) {
    startFastRTCRecording();
  }
  setupKeepaliveSSE();
});

function setupKeepaliveSSE() {
  const evtSource = new EventSource("/keepalive");
  evtSource.onmessage = function(event) {
    try {
      const dataStr = event.data.replace(/'/g, '"');
      const statusObj = JSON.parse(dataStr);
      updateCircleColor("status_frontend", statusObj.frontend);
      updateCircleColor("status_backend", statusObj.backend);
      updateCircleColor("status_mediagen", statusObj.mediagen);
      updateCircleColor("status_llm", statusObj.llm);
    } catch (err) {
      console.error("Error parsing keepalive data:", err, event.data);
    }
  };
  evtSource.onerror = function(err) {
    console.error("SSE /keepalive error:", err);
  };
}

function updateCircleColor(circleId, status) {
  const circle = document.getElementById(circleId);
  if (!circle) return;
  if (status === "alive") {
    circle.style.backgroundColor = "green";
  } else if (status === "dead") {
    circle.style.backgroundColor = "red";
  } else {
    circle.style.backgroundColor = "gray";
  }
}
