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

// Ensure global tracking variables are properly initialized
if (typeof window.audioBeingPlayed === 'undefined') {
  window.audioBeingPlayed = {};
}
if (typeof window.audioPlayed === 'undefined') {
  window.audioPlayed = new Set();
}
// Add a flag to track the very first audio playback
if (typeof window.firstAudioPlayed === 'undefined') {
  window.firstAudioPlayed = false;
}
// Add a lock mechanism for audio playback
if (typeof window.audioPlaybackLock === 'undefined') {
  window.audioPlaybackLock = false;
}

// TTS voice selection variables
let availableVoices = [];
let selectedVoice = null;
let ttsSpeed = 1.0;

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

// Function to clean up stale audio tracking data
function cleanupAudioTrackingData() {
  // Clear any tracking entries older than 5 minutes
  const fiveMinutesAgo = Date.now() - 5 * 60 * 1000;
  
  // Clean up task tracking
  for (const taskId in window.audioBeingPlayed) {
    const timestamp = window.audioBeingPlayed[taskId];
    if (typeof timestamp === 'number' && timestamp < fiveMinutesAgo) {
      delete window.audioBeingPlayed[taskId];
    }
  }
  
  console.log("Cleaned up audio tracking data");
}

// Run cleanup every 5 minutes
setInterval(cleanupAudioTrackingData, 5 * 60 * 1000);

// Function to check if critical services are healthy
function areServicesHealthy() {
  const llmStatus = document.getElementById('status_llm');
  const mediagenStatus = document.getElementById('status_mediagen');
  
  // Check if both elements exist and have green background color
  const isLlmHealthy = llmStatus && window.getComputedStyle(llmStatus).backgroundColor === 'rgb(0, 128, 0)';
  const isMediagenHealthy = mediagenStatus && window.getComputedStyle(mediagenStatus).backgroundColor === 'rgb(0, 128, 0)';
  
  return isLlmHealthy && isMediagenHealthy;
}

// Function to update microphone toggle state based on service health
function updateMicToggleBasedOnHealth() {
  const micCheckbox = document.querySelector(".mic-toggle input");
  const servicesHealthy = areServicesHealthy();
  
  if (micCheckbox) {
    if (!servicesHealthy) {
      // Disable microphone if services are unhealthy
      if (micCheckbox.checked) {
        console.log("Disabling microphone because critical services are unhealthy");
        
        // Stop any active recording
        if (recordingActive) {
          mediaRecorderRTC.stop();
          micStreamRTC.getTracks().forEach(track => track.stop());
          recordingActive = false;
        }
        
        // Uncheck and disable the microphone toggle
        micCheckbox.checked = false;
        micCheckbox.disabled = true;
        
        // Add visual indication that microphone is disabled
        const micToggle = document.querySelector(".mic-toggle");
        if (micToggle) {
          micToggle.classList.add("disabled");
          micToggle.title = "Microphone disabled because LLM or MediaGen services are unavailable";
        }
      }
    } else {
      // Re-enable microphone if services are healthy
      if (micCheckbox.disabled) {
        console.log("Re-enabling microphone because critical services are healthy");
        
        // Enable the microphone toggle
        micCheckbox.disabled = false;
        
        // Remove disabled visual indication
        const micToggle = document.querySelector(".mic-toggle");
        if (micToggle) {
          micToggle.classList.remove("disabled");
          micToggle.title = "Toggle microphone";
        }
      }
    }
  }
}

// Function to check if the microphone toggle is enabled and services are healthy
function isMicrophoneEnabled() {
  const micCheckbox = document.querySelector(".mic-toggle input");
  return micCheckbox && micCheckbox.checked && areServicesHealthy();
}

// Setup periodic service health check
function setupServiceHealthCheck() {
  // Check service health every 5 seconds
  setInterval(updateMicToggleBasedOnHealth, 5000);
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
  // Check if microphone toggle is enabled and services are healthy
  if (!isMicrophoneEnabled()) {
    console.log("Microphone toggle is disabled or services are unhealthy, not starting recording");
    return;
  }

  try {
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
      if (!recordingActive) return;
      
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
    
    const silenceCheckInterval = setInterval(checkSilence, 100);
    
    // Clean up interval when recording stops
    mediaRecorderRTC.onstop = () => {
      clearInterval(silenceCheckInterval);
    };
    
    mediaRecorderRTC.start();
    console.log("FastRTC recording started");
  } catch (err) {
    console.error("Error starting FastRTC recording:", err);
    recordingActive = false;
  }
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
  
  // Add TTS voice parameters if selected
  if (selectedVoice) {
    formData.append("voice_model", selectedVoice);
  }
  formData.append("tts_speed", ttsSpeed.toString());
  
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
    appendChatBubble("llm", data.llm_user_response);
    
    // Save the task ID if present
    if (data.task_id) {
      console.log(`Received task ID: ${data.task_id}`);
      localStorage.setItem('lastTaskId', data.task_id);
    }
    
    // Handle audio in one place only - prioritize direct URLs over polling
    if (data.tts_audio_url) {
      console.log(`Direct TTS URL available: ${data.tts_audio_url}`);
      playTTSAudio(data.tts_audio_url);
    } else if (data.task_id) {
      console.log(`No direct TTS URL, will poll using task ID: ${data.task_id}`);
      pollForAudio(data.task_id);
    }
    
    // Notify video manager for videos only (audio disabled in video_manager.js)
    if (window.handleNewConversationResponse) {
      window.handleNewConversationResponse(data);
    }
  } catch (e) {
    console.error("Error sending audio for transcript:", e);
  }
}

// Improved playTTSAudio with proper analyzer connection to visualize the waveform during playback
function playTTSAudio(url) {
  if (!url) {
    console.error("No audio URL provided for TTS playback");
    return;
  }

  // Extract base URL without query parameters
  const baseUrl = url.split('?')[0];

  // Check if audio playback is locked
  if (window.audioPlaybackLock) {
    console.log("Audio playback is currently locked, queueing for later");
    setTimeout(() => playTTSAudio(url), 500);
    return;
  }

  // Check if this audio has already been played recently
  if (window.audioPlayed.has(baseUrl) && window.firstAudioPlayed) {
    console.log(`Audio ${baseUrl} was already played, skipping duplicate playback`);
    return;
  }

  // Lock the audio playback
  window.audioPlaybackLock = true;

  // Mark as played
  window.audioPlayed.add(baseUrl);
  window.firstAudioPlayed = true;
  
  console.log("Playing TTS audio from URL:", url);

  // Clear from played set after 10 seconds to allow future playback if needed
  setTimeout(() => {
    window.audioPlayed.delete(baseUrl);
  }, 10000);

  // Add a cache buster to prevent browser caching
  const cacheBuster = new Date().getTime();
  const audioUrl = url.includes('?') ? `${url}&cb=${cacheBuster}` : `${url}?cb=${cacheBuster}`;

  const audio = new Audio(audioUrl);
  
  // Cleanup existing audio connections
  cleanupAudio();
  
  // Change waveform color to purple for TTS
  waveformColor = "purple";
  
  // Set up audio visualization when the audio can be played
  audio.addEventListener('canplay', function() {
    try {
      // Create new audio context if needed
      audioContext = getAudioContext();
      
      // Set up analyzer
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      
      // Connect audio element to analyzer
      source = audioContext.createMediaElementSource(audio);
      source.connect(analyser);
      
      // Also connect to destination (speakers) so we can hear it
      source.connect(audioContext.destination);
      
      console.log("TTS audio connected to analyzer for visualization");
      
      // Ensure waveform drawing is active
      if (!animationId) {
        drawWaveform();
      }
    } catch (e) {
      console.error("Error connecting TTS audio to analyzer:", e);
    }
  }, { once: true });

  // When audio playback ends
  audio.addEventListener("ended", () => {
    console.log("TTS audio playback completed");
    
    // Revert to green waveform
    waveformColor = "green";
    
    // Release the lock
    window.audioPlaybackLock = false;
    
    // Clean up audio connections
    cleanupAudio();
    
    // Only restart recording if the microphone toggle is enabled and services are healthy
    if (isMicrophoneEnabled() && !recordingActive) {
      console.log("Microphone toggle is enabled and services are healthy, starting recording");
      startFastRTCRecording();
    } else {
      console.log("Microphone toggle is disabled or services are unhealthy, not starting recording");
    }
  });

  // Handle playback errors
  audio.addEventListener("error", (e) => {
    console.error("Error playing TTS audio:", e);
    
    // Revert to green waveform
    waveformColor = "green";
    
    // Release the lock on error
    window.audioPlaybackLock = false;
    
    // Clean up audio connections
    cleanupAudio();
    
    // Only restart recording if the microphone toggle is enabled and services are healthy
    if (isMicrophoneEnabled() && !recordingActive) {
      console.log("Microphone toggle is enabled and services are healthy, starting recording after error");
      startFastRTCRecording();
    }
  });

  // Start audio playback
  audio.play().catch((e) => {
    console.error("Error starting TTS audio playback:", e);
    
    // Revert to green waveform
    waveformColor = "green";
    
    // Release the lock on error
    window.audioPlaybackLock = false;
    
    // Clean up audio connections
    cleanupAudio();
    
    // Only restart recording if the microphone toggle is enabled and services are healthy
    if (isMicrophoneEnabled() && !recordingActive) {
      console.log("Microphone toggle is enabled and services are healthy, starting recording after error");
      startFastRTCRecording();
    }
  });
}

// Improved polling for audio with better error handling and duplicate prevention
async function pollForAudio(taskId) {
  if (!taskId) return;
  
  // Prevent duplicate polling
  if (window.audioBeingPlayed[taskId]) {
    console.log(`Already polling/playing audio for task ${taskId}, skipping duplicate polling`);
    return;
  }
  
  // Mark as being handled with timestamp
  window.audioBeingPlayed[taskId] = Date.now();
  
  console.log(`Starting to poll for audio with task ID: ${taskId}`);
  
  try {
    // Poll for audio status
    for (let attempt = 1; attempt <= 30; attempt++) {
      try {
        console.log(`Polling attempt ${attempt} for audio: ${taskId}`);
        const response = await fetch(`/audio_status/${taskId}`);
        
        if (!response.ok) {
          console.warn(`Audio status endpoint returned ${response.status}, will retry...`);
          await new Promise(resolve => setTimeout(resolve, 1000));
          continue;
        }
        
        const data = await response.json();
        console.log(`Audio status for ${taskId}:`, data);
        
        if (data.status === "completed" && data.audio_url) {
          console.log(`Audio ready for playback: ${data.audio_url}`);
          
          // Wait a moment before playing to ensure any other
          // parallel playback attempts have been processed
          await new Promise(resolve => setTimeout(resolve, 100));
          
          // Check if this audio URL has been played recently
          const baseUrl = data.audio_url.split('?')[0];
          if (window.audioPlayed.has(baseUrl) && window.firstAudioPlayed) {
            console.log(`Audio ${baseUrl} was already played, won't play again`);
          } else {
            // Play the TTS audio with visualization
            playTTSAudio(data.audio_url);
          }
          
          return; // Exit polling loop
        }
        
        // Wait 1 second before next attempt
        await new Promise(resolve => setTimeout(resolve, 1000));
      } catch (err) {
        console.error(`Error polling audio status (attempt ${attempt}):`, err);
        // Wait before retry
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
    
    console.warn(`Audio polling timed out for task ID: ${taskId}`);
  } finally {
    // Always clean up, even if there's an error
    delete window.audioBeingPlayed[taskId];
  }
}

// Optional: Popup for transcript display.
function displayTranscript(transcript) {
  console.log("Transcript:", transcript);
  // You could implement a popup here if desired.
}

// === Voice Selection Functions ===
async function fetchAvailableVoices() {
  try {
    const response = await fetch('/api/voices');
    if (!response.ok) {
      throw new Error('Failed to fetch voices');
    }
    const data = await response.json();
    if (data.success && data.voices) {
      availableVoices = data.voices;
      populateVoiceSelector();
    }
  } catch (error) {
    console.error('Error fetching voices:', error);
  }
}

function populateVoiceSelector() {
  const voiceSelector = document.getElementById('voice-selector');
  if (!voiceSelector) return;
  
  // Clear existing options
  voiceSelector.innerHTML = '';
  
  // Add a default option
  const defaultOption = document.createElement('option');
  defaultOption.value = '';
  defaultOption.textContent = 'Default voice';
  voiceSelector.appendChild(defaultOption);
  
  // Group voices by language
  const voicesByLanguage = {};
  availableVoices.forEach(voice => {
    if (!voicesByLanguage[voice.language]) {
      voicesByLanguage[voice.language] = [];
    }
    voicesByLanguage[voice.language].push(voice);
  });
  
  // Create option groups by language
  Object.keys(voicesByLanguage).sort().forEach(language => {
    const group = document.createElement('optgroup');
    group.label = `Language: ${language}`;
    
    voicesByLanguage[language].forEach(voice => {
      const option = document.createElement('option');
      option.value = voice.model_name;
      option.textContent = `${voice.speaker} (${voice.language})`;
      group.appendChild(option);
    });
    
    voiceSelector.appendChild(group);
  });
}

function handleVoiceSelection(event) {
  selectedVoice = event.target.value;
  console.log(`Selected voice: ${selectedVoice}`);
}

function handleSpeedChange(event) {
  ttsSpeed = parseFloat(event.target.value);
  document.getElementById('speed-value').textContent = ttsSpeed.toFixed(1);
  console.log(`TTS speed set to: ${ttsSpeed}`);
}

// === Mic Toggle Control ===
document.addEventListener("DOMContentLoaded", () => {
  const micCheckbox = document.querySelector(".mic-toggle input");
  if (micCheckbox) {
    micCheckbox.addEventListener("change", () => {
      // First check if services are healthy before allowing toggle
      if (!areServicesHealthy()) {
        console.log("Cannot enable microphone because critical services are unhealthy");
        micCheckbox.checked = false;
        
        // Show an alert to the user
        alert("Cannot enable microphone because LLM or MediaGen services are unavailable. Please check service status indicators.");
        
        return;
      }
      
      if (micCheckbox.checked) {
        console.log("Microphone toggle enabled");
        if (!recordingActive) {
          startFastRTCRecording();
        }
      } else {
        console.log("Microphone toggle disabled");
        if (recordingActive) {
          // Explicitly stop recording when toggle is turned off
          mediaRecorderRTC.stop();
          micStreamRTC.getTracks().forEach(track => track.stop());
          recordingActive = false;
        }
      }
    });
    
    // Add CSS for disabled state
    const style = document.createElement('style');
    style.textContent = `
      .mic-toggle.disabled {
        opacity: 0.5;
        cursor: not-allowed;
      }
    `;
    document.head.appendChild(style);
    
    // Check service health on load before starting recording
    if (micCheckbox.checked) {
      if (areServicesHealthy()) {
        console.log("Microphone toggle initially enabled and services healthy, starting recording");
        startFastRTCRecording();
      } else {
        console.log("Microphone toggle initially enabled but services unhealthy, disabling toggle");
        micCheckbox.checked = false;
        micCheckbox.disabled = true;
        
        const micToggle = document.querySelector(".mic-toggle");
        if (micToggle) {
          micToggle.classList.add("disabled");
          micToggle.title = "Microphone disabled because LLM or MediaGen services are unavailable";
        }
      }
    } else {
      console.log("Microphone toggle initially disabled, not starting recording");
    }
  }
  
  // Set up voice selector event listener
  const voiceSelector = document.getElementById('voice-selector');
  if (voiceSelector) {
    voiceSelector.addEventListener('change', handleVoiceSelection);
  }
  
  // Set up speed control event listener
  const speedControl = document.getElementById('speed-control');
  if (speedControl) {
    speedControl.addEventListener('input', handleSpeedChange);
    // Initialize the display value
    const speedValue = document.getElementById('speed-value');
    if (speedValue) {
      speedValue.textContent = speedControl.value;
    }
  }
  
  // Fetch available voices when page loads
  fetchAvailableVoices();
  
  // Setup services
  setupKeepaliveSSE();
  setupServiceHealthCheck();
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

// Modify the updateCircleColor function to check service health after color update
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
  
  // Check if we need to update microphone state whenever LLM or MediaGen status changes
  if (circleId === 'status_llm' || circleId === 'status_mediagen') {
    updateMicToggleBasedOnHealth();
  }
}

// Improved processExistingResponseFunction to avoid audio duplication
function processExistingResponseFunction(data) {
  // Your existing code to process the response
  console.log("Processing existing response:", data);
  
  // If this data contains a TTS URL, handle it only once
  if (data.tts_audio_url) {
    const baseUrl = data.tts_audio_url.split('?')[0];
    if (!window.audioPlayed.has(baseUrl) || !window.firstAudioPlayed) {
      console.log("Playing TTS from processExistingResponseFunction");
      playTTSAudio(data.tts_audio_url);
    } else {
      console.log("Audio already played, skipping duplicate in processExistingResponseFunction");
    }
  } else if (data.task_id && !window.audioBeingPlayed[data.task_id]) {
    // Only poll if we're not already polling for this task
    console.log("Starting audio polling from processExistingResponseFunction");
    pollForAudio(data.task_id);
  }
  
  // Notify video manager for video only (audio is disabled there)
  if (window.handleNewConversationResponse) {
    window.handleNewConversationResponse(data);
  }
}

// Function to ensure the VideoManager is notified of new responses
function notifyVideoManager(data) {
  console.log("Notifying VideoManager of new response:", data);
  
  if (window.handleNewConversationResponse && data && data.task_id) {
      window.handleNewConversationResponse(data);
      
      // Also manually check for video updates
      if (typeof checkForVideoUpdate === 'function') {
          setTimeout(() => checkForVideoUpdate(data.task_id), 1000);
      }
  }
}

// Hook into any existing response processing functions
document.addEventListener('DOMContentLoaded', function() {
  // Hook into any function that processes API responses
  if (window.processExistingResponseFunction) {
      const originalFn = window.processExistingResponseFunction;
      window.processExistingResponseFunction = function(data) {
          // Call the original function
          originalFn(data);
          
          // Notify the VideoManager
          notifyVideoManager(data);
      };
  }
  
  // Look for the appendChatBubble function, which is likely called when new messages are added
  if (window.appendChatBubble) {
      const originalAppendChatBubble = window.appendChatBubble;
      window.appendChatBubble = function(sender, text) {
          // Call the original function
          originalAppendChatBubble(sender, text);
          
          // If this is an LLM response, it might be related to a task
          if (sender === 'llm') {
              // Check our most recent task ID from localStorage
              const lastTaskId = localStorage.getItem('lastTaskId');
              if (lastTaskId) {
                  console.log(`Chat bubble added, checking video for last task: ${lastTaskId}`);
                  setTimeout(() => checkForVideoUpdate(lastTaskId), 1000);
              }
          }
      };
  }
});

// Intercept form submissions to capture task IDs
document.addEventListener('submit', function(e) {
  const form = e.target;
  if (form.getAttribute('action') === '/input_hook' || form.getAttribute('action')?.includes('/generate/')) {
      console.log("Form submission intercepted");
      
      // Save the timestamp to help identify the response
      localStorage.setItem('lastFormSubmission', Date.now());
  }
});

// Periodic check for newly finished videos
setInterval(function() {
  const lastTaskId = localStorage.getItem('lastTaskId');
  if (lastTaskId) {
      console.log(`Periodic check for video update: ${lastTaskId}`);
      checkForVideoUpdate(lastTaskId);
  }
}, 10000); // Check every 10 seconds