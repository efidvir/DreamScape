// video-manager.js - New file to add to your project
class VideoManager {
    constructor() {
        this.videoElement = document.getElementById('scene-video');
        this.currentTaskId = null;
        this.isPolling = false;
        this.defaultVideo = '/static/media/default-scene.mp4';  // Optional default video
        this.checkInterval = 2000;  // Check every 2 seconds
        
        // Initialize if the video element exists
        if (this.videoElement) {
            // Set default video if available
            this.setDefaultVideo();
            
            // Initialize event listeners
            this.initEventListeners();
            
            console.log('Video Manager initialized');
        } else {
            console.error('Scene video element not found');
        }
    }
    
    initEventListeners() {
        // Listen for video errors
        this.videoElement.addEventListener('error', (e) => {
            console.error('Video failed to load', e);
            this.setDefaultVideo();
        });
        
        // When video ends, ensure it loops (though loop attribute should handle this)
        this.videoElement.addEventListener('ended', () => {
            this.videoElement.play().catch(e => console.error('Error replaying video:', e));
        });
    }
    
    setDefaultVideo() {
        // Only set default if provided
        if (this.defaultVideo) {
            fetch(this.defaultVideo, { method: 'HEAD' })
                .then(response => {
                    if (response.ok) {
                        this.videoElement.src = this.defaultVideo;
                        this.videoElement.load();
                        this.videoElement.play().catch(e => console.error('Error playing default video:', e));
                    }
                })
                .catch(error => console.error('Default video not available', error));
        }
    }
    
    startTrackingVideo(taskId) {
        if (!this.videoElement) {
            console.error('Video element not found');
            return;
        }
        
        this.currentTaskId = taskId;
        
        // Stop any existing polling
        if (this.isPolling) {
            this.stopPolling();
        }
        
        // Start polling for video status
        this.isPolling = true;
        this.pollForVideo();
        
        console.log(`Started tracking video for task: ${taskId}`);
    }
    
    stopPolling() {
        this.isPolling = false;
        if (this.pollingTimeout) {
            clearTimeout(this.pollingTimeout);
            this.pollingTimeout = null;
        }
    }
    
    pollForVideo() {
        if (!this.isPolling || !this.currentTaskId) {
            return;
        }
        
        fetch(`/mediagen/video_status/${this.currentTaskId}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('Video status:', data);
                
                // Check if video is ready
                if (data.status === 'completed' && data.video_url) {
                    // Video is ready, load it
                    this.loadVideo(data.video_url);
                    // We can stop polling once video is found
                    this.stopPolling();
                } else if (data.status.startsWith('error')) {
                    // Video generation failed
                    console.error(`Video generation failed: ${data.status}`);
                    this.stopPolling();
                } else {
                    // Continue polling
                    this.pollingTimeout = setTimeout(() => this.pollForVideo(), this.checkInterval);
                }
            })
            .catch(error => {
                console.error('Error checking video status:', error);
                // Continue polling even on error, in case it's temporary
                this.pollingTimeout = setTimeout(() => this.pollForVideo(), this.checkInterval);
            });
    }
    
    loadVideo(videoUrl) {
        // Add timestamp to prevent caching
        const url = `${videoUrl}?t=${new Date().getTime()}`;
        console.log(`Loading video from: ${url}`);
        
        // Set the video source and play it
        this.videoElement.src = url;
        this.videoElement.load();
        
        // Only after metadata is loaded, start playing
        this.videoElement.onloadedmetadata = () => {
            this.videoElement.play()
                .then(() => console.log('Video playing'))
                .catch(error => console.error('Error playing video:', error));
        };
    }
}

// Initialize the video manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.videoManager = new VideoManager();
});

// Handle new conversation responses - can be called from main.js
function handleNewConversationResponse(response) {
    if (window.videoManager && response && response.task_id) {
        window.videoManager.startTrackingVideo(response.task_id);
    }
}

// Make the function available globally
window.handleNewConversationResponse = handleNewConversationResponse;