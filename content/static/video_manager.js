class VideoManager {
    constructor() {
        this.videoTrackingIntervals = {};
        this.videoElement = document.getElementById('video-player');
        if (!this.videoElement) {
            console.error("Video player element not found!");
        }
    }

    startTrackingVideo(taskId) {
        console.log(`Starting to track video for task: ${taskId}`);
        
        // Clear any existing interval for this task
        if (this.videoTrackingIntervals[taskId]) {
            clearInterval(this.videoTrackingIntervals[taskId]);
        }
        
        // Set up polling interval to check video status
        this.videoTrackingIntervals[taskId] = setInterval(() => {
            this.checkVideoStatus(taskId);
        }, 2000); // Check every 2 seconds
    }
    
    stopTrackingVideo(taskId) {
        if (this.videoTrackingIntervals[taskId]) {
            clearInterval(this.videoTrackingIntervals[taskId]);
            delete this.videoTrackingIntervals[taskId];
            console.log(`Stopped tracking video for task: ${taskId}`);
        }
    }
    
    async checkVideoStatus(taskId) {
        try {
            const response = await fetch(`/video_status/${taskId}`);
            const data = await response.json();
            
            console.log(`Video status for ${taskId}:`, data);
            
            if (data.status === "completed" && data.video_url) {
                // Video is ready, play it
                this.playVideo(data.video_url);
                this.stopTrackingVideo(taskId);
            }
        } catch (error) {
            console.error(`Error checking video status for ${taskId}:`, error);
        }
    }
    
    playVideo(videoUrl) {
        if (!this.videoElement) {
            this.videoElement = document.getElementById('video-player');
            if (!this.videoElement) {
                console.error("Video player element still not found!");
                return;
            }
        }
        
        console.log(`Playing video from URL: ${videoUrl}`);
        
        // Update the video source and play
        this.videoElement.src = videoUrl;
        this.videoElement.style.display = 'block';
        
        // Play the video with error handling
        this.videoElement.load();
        const playPromise = this.videoElement.play();
        
        if (playPromise !== undefined) {
            playPromise.catch(error => {
                console.error('Error playing video:', error);
            });
        }
    }
}

// Initialize the video manager when the page loads
window.addEventListener('DOMContentLoaded', () => {
    window.videoManager = new VideoManager();
    console.log('VideoManager initialized');
});

function checkForVideoUpdate(taskId) {
    if (window.videoManager) {
        window.videoManager.checkVideoStatus(taskId);
    }
}

/**
 * MODIFIED: This function now only handles video tracking, not audio
 * to prevent duplicate audio playback
 */
function handleNewConversationResponse(response) {
    console.log('Handling conversation response:', response);
    
    if (window.videoManager && response && response.task_id) {
        // Only handle video tracking, not audio
        window.videoManager.startTrackingVideo(response.task_id);
    }
    
    // Audio handling is disabled to prevent duplicate playback
    // Audio will be handled by the main script.js
}

/**
 * DISABLED: This function now only logs a message and does nothing
 * to prevent duplicate audio playback
 */
function playTTSWhenReady(response) {
    console.log('Audio handling disabled in video_manager.js - main script will handle audio');
    // Audio handling is disabled to prevent duplicate playback
}

/**
 * DISABLED: This function now only logs a message and does nothing
 * to prevent duplicate audio playback
 */
function pollAudioStatus(taskId) {
    console.log(`Audio polling disabled in video_manager.js - main script will handle audio for task: ${taskId}`);
    // Audio polling is disabled to prevent duplicate playback
}

/**
 * DISABLED: This function now only logs a message and does nothing
 * to prevent duplicate audio playback
 */
function playAudioSequence(urls, index) {
    console.log('Audio playback disabled in video_manager.js - main script will handle audio', urls);
    // Audio playback is disabled to prevent duplicate playback
}

// Original functions preserved below for reference but disabled
/*
function playTTSWhenReady(response) {
    const audioUrls = response.tts_audio_urls || (response.tts_audio_url ? [response.tts_audio_url] : []);
    
    if (audioUrls.length === 0) {
        // Check if we need to poll for audio status
        if (response.task_id) {
            pollAudioStatus(response.task_id);
        }
        return;
    }
    
    console.log('Playing TTS audio with URLs:', audioUrls);
    
    // Queue the audio files for sequential playback
    playAudioSequence(audioUrls, 0);
}

function pollAudioStatus(taskId) {
    console.log(`Starting to poll audio status for task: ${taskId}`);
    
    // Set up interval to check audio status
    const audioCheckInterval = setInterval(async () => {
        try {
            const response = await fetch(`/audio_status/${taskId}`);
            const data = await response.json();
            
            console.log(`Audio status for ${taskId}:`, data);
            
            if (data.status === "completed" && data.audio_url) {
                // Audio is ready, play it
                playAudioSequence([data.audio_url], 0);
                clearInterval(audioCheckInterval);
            }
        } catch (error) {
            console.error(`Error checking audio status for ${taskId}:`, error);
        }
    }, 1000); // Check every 1 second
    
    // Stop polling after 30 seconds to prevent indefinite polling
    setTimeout(() => {
        clearInterval(audioCheckInterval);
        console.log(`Stopped polling audio status for task: ${taskId} after timeout`);
    }, 30000);
}

function playAudioSequence(urls, index) {
    if (index >= urls.length) {
        console.log('Audio sequence playback completed');
        return;
    }
    
    const url = urls[index];
    console.log(`Playing audio ${index + 1}/${urls.length}: ${url}`);
    
    const audio = new Audio(url);
    
    // Set waveform color to purple during playback
    window.waveformColor = "purple";
    
    // Listen for playback completion
    audio.addEventListener("ended", () => {
        // Play the next audio in the sequence
        playAudioSequence(urls, index + 1);
    });
    
    // Set up error handling
    audio.addEventListener("error", (e) => {
        console.error("Error playing TTS audio:", e);
        
        // Reset waveform color and try the next audio
        window.waveformColor = "green";
        playAudioSequence(urls, index + 1);
    });
    
    // When all audio has finished playing, reset the waveform color
    if (index === urls.length - 1) {
        audio.addEventListener("ended", () => {
            window.waveformColor = "green";
            console.log("Audio playback complete, ready for user input");
        });
    }
    
    // Start playback
    audio.play().catch((e) => {
        console.error("Error starting audio playback:", e);
        window.waveformColor = "green";
        playAudioSequence(urls, index + 1);
    });
}
*/