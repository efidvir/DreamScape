// Global AudioContext re-used for all operations
let globalAudioContext = null;
function getAudioContext() {
  if (!globalAudioContext) {
    globalAudioContext = new (window.AudioContext || window.webkitAudioContext)();
    console.log("Global AudioContext created.");
  }
  return globalAudioContext;
}

// Resume AudioContext on any user gesture (click)
document.addEventListener("click", () => {
  const ctx = getAudioContext();
  if (ctx.state === "suspended") {
    ctx.resume().then(() => {
      console.log("Global AudioContext resumed on user gesture.");
    });
  }
});

// === Global Variables for Waveform Visualization ===
let audioContext; // will use the globalAudioContext
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

// Initialize waveform visualization for microphone recording
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

// (For TTS visualization, we now create a dedicated analyser when playing TTS)
function startTTSVisualizationWithAnalyser(ttsAnalyser) {
  // Set our global analyser to the one for TTS playback and start drawing
  analyser = ttsAnalyser;
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
let silenceTimeout = 2000;   // ms of silence before sending audio
let silenceTimer = null;
let mediaRecorderRTC;
let micStreamRTC;

async function startFastRTCRecording() {
  // Request explicit constraints to help capture audio correctly
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
    const response = await fetch("/debug_response", {
      method: "POST",
      body: formData
    });
    if (!response.ok) {
      throw new Error(`Server error: ${response.statusText}`);
    }
    const ttsBlob = await response.blob();
    console.log("Received TTS blob size:", ttsBlob.size);
    const ttsUrl = URL.createObjectURL(ttsBlob);
    playTTS(ttsUrl);
  } catch (e) {
    console.error("Error sending audio to backend:", e);
  }
}

// Updated playTTS using Web Audio API for playback and visualization
function playTTS(ttsUrl) {
  console.log("Attempting to play TTS via Web Audio API from URL:", ttsUrl);
  const ctx = getAudioContext();
  fetch(ttsUrl)
    .then(response => response.arrayBuffer())
    .then(arrayBuffer => ctx.decodeAudioData(arrayBuffer))
    .then(audioBuffer => {
      const sourceNode = ctx.createBufferSource();
      sourceNode.buffer = audioBuffer;
      
      // Create an analyser for TTS playback visualization
      const ttsAnalyser = ctx.createAnalyser();
      ttsAnalyser.fftSize = 2048;
      
      // Connect the source to the analyser and then to the destination
      sourceNode.connect(ttsAnalyser);
      ttsAnalyser.connect(ctx.destination);
      
      // Set the waveform color to purple and start visualization using this analyser
      startTTSVisualizationWithAnalyser(ttsAnalyser);
      
      sourceNode.start(0);
      console.log("TTS audio playback started via Web Audio API.");
      
      sourceNode.onended = () => {
        console.log("TTS audio playback ended via Web Audio API.");
        cancelAnimationFrame(animationId);
        clearCanvas();
        const micCheckbox = document.querySelector(".mic-toggle input");
        micCheckbox.disabled = false;
        if (micCheckbox.checked) {
          startFastRTCRecording();
        }
      };
    })
    .catch(err => {
      console.error("Error decoding/playing TTS audio:", err);
    });
}

// Test function to play background music (via HTML5 Audio)
function playTestAudio() {
  const testAudioUrl = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3";
  const testAudio = new Audio(testAudioUrl);
  testAudio.volume = 1.0;
  testAudio.onplay = () => {
    console.log("Test audio is playing.");
  };
  testAudio.onerror = (err) => {
    console.error("Error playing test audio:", err);
  };
  testAudio.play().then(() => {
    console.log("Test audio play() resolved.");
  }).catch((err) => {
    console.error("Test audio play() failed:", err);
  });
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
  // Uncomment the line below to test background music playback:
  // playTestAudio();
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
