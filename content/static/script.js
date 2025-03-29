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
let waveformColor = "green"; // default: green when recording

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
  ctx.shadowColor = waveformColor === "green" 
    ? "rgba(0,255,0,0.6)" 
    : "rgba(160,32,240,0.6)";
  ctx.strokeStyle = waveformColor === "green" 
    ? "rgba(0,255,0,0.8)" 
    : "rgba(160,32,240,0.8)";
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

// Popup for transcript display at top left with a container frame
function showTranscriptPopup(transcript) {
  let popup = document.getElementById("transcript-popup");
  if (!popup) {
    popup = document.createElement("div");
    popup.id = "transcript-popup";
    // Styling similar to the reference design
    popup.style.position = "fixed";
    popup.style.top = "20px";
    popup.style.left = "20px";
    popup.style.backgroundColor = "rgba(0, 0, 0, 0.85)";
    popup.style.color = "#fff";
    popup.style.padding = "15px 20px";
    popup.style.border = "2px solid #fff";
    popup.style.borderRadius = "8px";
    popup.style.boxShadow = "0 4px 8px rgba(0, 0, 0, 0.3)";
    popup.style.zIndex = "9999";
    popup.style.fontSize = "1.1em";
    popup.style.maxWidth = "300px";
    document.body.appendChild(popup);
  }
  popup.textContent = transcript;
  popup.style.display = "block";
  popup.style.opacity = "1";
  // Fade out after 5 seconds
  setTimeout(() => {
    popup.style.transition = "opacity 1s ease-out";
    popup.style.opacity = "0";
    setTimeout(() => {
      popup.style.display = "none";
    }, 1000);
  }, 5000);
}

function displayTranscript(transcript) {
  console.log("Transcript:", transcript);
  showTranscriptPopup(transcript);
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
  // Request explicit audio constraints to help capture audio correctly
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
    // Note: The endpoint is now "/generate/transcript" to receive a proper transcript.
    const response = await fetch("/generate/transcript", {
      method: "POST",
      body: formData
    });
    if (!response.ok) {
      throw new Error(`Server error: ${response.statusText}`);
    }
    const data = await response.json();
    console.log("Transcript received:", data.transcript);
    displayTranscript(data.transcript);
  } catch (e) {
    console.error("Error sending audio for transcript:", e);
  }
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
