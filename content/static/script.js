let mediaRecorder;
let audioChunks = [];
let waveSurfer;

async function startRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);
  audioChunks = [];

  mediaRecorder.ondataavailable = (event) => {
    audioChunks.push(event.data);
  };

  mediaRecorder.onstop = async () => {
    const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
    const formData = new FormData();
    formData.append("audio", audioBlob);

    const response = await fetch("/api/audio", {
      method: "POST",
      body: formData,
    });
    const result = await response.json();
    displayResponse(result.response);
  };

  mediaRecorder.start();
}

function stopRecording() {
  if (mediaRecorder) {
    mediaRecorder.stop();
  }
}

function displayResponse(response) {
  if (!waveSurfer) {
    waveSurfer = WaveSurfer.create({
      container: "#waveform",
      waveColor: "violet",
      interact: false,
      cursorWidth: 0,
    });
  }
  console.log("AI Response:", response);
}

function toggleLanguage() {
  console.log("Language toggled");
}

function openSettings() {
  console.log("Settings opened");
}
document.addEventListener("DOMContentLoaded", () => {
    const micCheckbox = document.querySelector(".mic-toggle input");
    micCheckbox.addEventListener("change", () => {
      if (micCheckbox.checked) {
        console.log("Microphone active!");
        // Insert real logic for enabling mic
      } else {
        console.log("Microphone muted!");
        // Insert real logic for muting mic
      }
    });
  });
  