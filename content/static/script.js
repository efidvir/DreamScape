// === Global Variables for Waveform Visualization ===
let audioContext;
let analyser;
let source;
let animationId;
let currentMode = null; // "mic" for recording, "tts" for playback
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

// Draw the waveform on the canvas with glow effect
function drawWaveform() {
  animationId = requestAnimationFrame(drawWaveform);
  if (!analyser) return;

  const bufferLength = analyser.fftSize;
  const dataArray = new Uint8Array(bufferLength);
  analyser.getByteTimeDomainData(dataArray);

  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Set up glow effect based on current mode: green for mic, purple for TTS
  ctx.lineWidth = 2;
  ctx.shadowBlur = 20;
  ctx.shadowColor = waveformColor === "green" 
    ? "rgba(0,255,0,0.6)" 
    : "rgba(160,32,240,0.6)";
  ctx.strokeStyle = waveformColor === "green" 
    ? "rgba(0,255,0,0.8)" 
    : "rgba(160,32,240,0.8)";

  ctx.beginPath();
  const sliceWidth = canvas.width / bufferLength;
  let x = 0;
  for (let i = 0; i < bufferLength; i++) {
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

// Initialize waveform visualization for microphone recording
function startMicVisualizationFromStream(stream) {
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 2048;
  source = audioContext.createMediaStreamSource(stream);
  source.connect(analyser);
  currentMode = "mic";
  waveformColor = "green";
  drawWaveform();
}

// Clean up audio nodes and animation
function cleanupAudio() {
  try {
    if (source) source.disconnect();
    if (analyser) analyser.disconnect();
    if (audioContext) audioContext.close();
  } catch (e) {
    console.error("Error during cleanup:", e);
  }
  currentMode = null;
}
function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// Initialize TTS visualization (waveform turns purple)
function startTTSVisualization(audioElement) {
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 2048;
  source = audioContext.createMediaElementSource(audioElement);
  source.connect(analyser);
  currentMode = "tts";
  waveformColor = "purple";
  drawWaveform();
}
function stopTTSVisualization() {
  if (animationId) {
    cancelAnimationFrame(animationId);
  }
  cleanupAudio();
  clearCanvas();
}

// === Continuous Voice Chat with Silence Detection ===

let webrtcId = Math.random().toString(36).substring(2, 10);
let audioChunksForFastRTC = [];
let recordingActive = false;
let silenceThreshold = 0.05; // Adjust threshold as needed
let silenceTimeout = 2000;   // ms to consider as silence (end of query)
let silenceTimer = null;
let mediaRecorderRTC;
let micStreamRTC;

async function startFastRTCRecording() {
  // Get microphone stream
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  micStreamRTC = stream;
  recordingActive = true;
  audioChunksForFastRTC = [];

  // Set up MediaRecorder for capturing audio chunks
  mediaRecorderRTC = new MediaRecorder(stream);
  mediaRecorderRTC.ondataavailable = (event) => {
    if (event.data.size > 0) {
      audioChunksForFastRTC.push(event.data);
    }
  };

  // Start visualization in mic mode (green)
  startMicVisualizationFromStream(stream);

  // Set up silence detection with a separate analyser
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
        // Silence detected, so wait a bit then send audio
        silenceTimer = setTimeout(() => {
          sendAudioToFastRTC();
          silenceTimer = null;
        }, silenceTimeout);
      }
    } else {
      // Reset the silence timer if user is still speaking
      if (silenceTimer) {
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
  // Stop recording and microphone stream
  mediaRecorderRTC.stop();
  micStreamRTC.getTracks().forEach(track => track.stop());
  recordingActive = false;

  // Combine recorded chunks into a blob
  const audioBlob = new Blob(audioChunksForFastRTC, { type: "audio/wav" });
  audioChunksForFastRTC = [];
  const formData = new FormData();
  formData.append("audio", audioBlob);
  formData.append("webrtc_id", webrtcId);

  try {
    // Send audio query to backend
    const response = await fetch("/input_hook", {
      method: "POST",
      body: formData
    });
    if (!response.ok) {
      throw new Error(`Server error: ${response.statusText}`);
    }
    // Receive the TTS audio response as a blob
    const ttsBlob = await response.blob();
    const ttsUrl = URL.createObjectURL(ttsBlob);
    playTTS(ttsUrl);
  } catch (e) {
    console.error("Error sending audio to backend:", e);
  }
}

function playTTS(ttsUrl) {
  const audio = new Audio(ttsUrl);
  // Disable the mic toggle during playback
  const micCheckbox = document.querySelector(".mic-toggle input");
  micCheckbox.disabled = true;
  audio.onplay = () => {
    // Change waveform color to purple for TTS playback
    startTTSVisualization(audio);
  };
  audio.onended = () => {
    // Once TTS finishes, stop visualization, re-enable mic, and restart recording
    stopTTSVisualization();
    micCheckbox.disabled = false;
    if (micCheckbox.checked) {
      startFastRTCRecording();
    }
  };
  audio.play();
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
  // Automatically start recording if mic toggle is enabled on page load
  if (micCheckbox.checked) {
    startFastRTCRecording();
  }

  // **NEW**: Subscribe to keepalive SSE
  setupKeepaliveSSE();
});

/**
 * Sets up an SSE connection to /keepalive to get container statuses.
 */
function setupKeepaliveSSE() {
  const evtSource = new EventSource("/keepalive");
  evtSource.onmessage = function(event) {
    try {
      // Data might look like: {frontend:'alive', backend:'alive', mediagen:'dead', llm:'alive'}
      // We'll parse it out. If your server sends double quotes, you can do JSON.parse(event.data).
      const dataStr = event.data.replace(/'/g, '"');
      const statusObj = JSON.parse(dataStr);

      updateCircleColor("status_frontend",  statusObj.frontend);
      updateCircleColor("status_backend",   statusObj.backend);
      updateCircleColor("status_mediagen",  statusObj.mediagen);
      updateCircleColor("status_llm",       statusObj.llm);
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
