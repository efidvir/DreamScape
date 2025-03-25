// === Global Variables for Waveform Visualization ===
let audioContext;
let analyser;
let source;
let animationId;
let currentMode = null; // "mic" or "tts"
let waveformColor = "green"; // default

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

  // Set up glow effect
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

// Initialize waveform visualization using a provided stream
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

// === TTS Visualization Functions ===
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
let silenceThreshold = 0.05; // Adjust as needed
let silenceTimeout = 2000;   // ms to consider as silence
let silenceTimer = null;
let mediaRecorderRTC;
let micStreamRTC;

async function startFastRTCRecording() {
  // Get microphone stream
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  micStreamRTC = stream;
  recordingActive = true;
  audioChunksForFastRTC = [];

  // Set up MediaRecorder for recording audio chunks
  mediaRecorderRTC = new MediaRecorder(stream);
  mediaRecorderRTC.ondataavailable = (event) => {
    if (event.data.size > 0) {
      audioChunksForFastRTC.push(event.data);
    }
  };

  // Start waveform visualization using the same stream
  startMicVisualizationFromStream(stream);

  // Set up silence detection
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
        silenceTimer = setTimeout(() => {
          sendAudioToFastRTC();
          silenceTimer = null;
        }, silenceTimeout);
      }
    } else {
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
  mediaRecorderRTC.stop();
  micStreamRTC.getTracks().forEach(track => track.stop());
  recordingActive = false;

  // Combine recorded chunks into a blob.
  const audioBlob = new Blob(audioChunksForFastRTC, { type: "audio/wav" });
  audioChunksForFastRTC = [];
  const formData = new FormData();
  formData.append("audio", audioBlob);
  formData.append("webrtc_id", webrtcId);

  try {
    // Call the backend orchestrator endpoint which returns TTS audio as a blob.
    const response = await fetch("/input_hook", {
      method: "POST",
      body: formData
    });
    if (!response.ok) {
      throw new Error(`Server error: ${response.statusText}`);
    }
    const ttsBlob = await response.blob();
    const ttsUrl = URL.createObjectURL(ttsBlob);
    playTTS(ttsUrl);
  } catch (e) {
    console.error("Error sending audio to FastRTC:", e);
  }

  // After TTS playback, resume recording if the mic is still enabled.
  setTimeout(() => {
    const micCheckbox = document.querySelector(".mic-toggle input");
    if (micCheckbox.checked) {
      startFastRTCRecording();
    }
  }, 1000);
}

// === TTS Playback Function ===
function playTTS(ttsUrl) {
  const audio = new Audio(ttsUrl);
  const micCheckbox = document.querySelector(".mic-toggle input");
  micCheckbox.disabled = true; // Disable mic during TTS playback
  audio.onplay = () => startTTSVisualization(audio);
  audio.onended = () => {
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
});
