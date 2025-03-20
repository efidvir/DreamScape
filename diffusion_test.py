import torch
from diffusers import (
    StableDiffusionPipeline,
    StableVideoDiffusionPipeline
)
import imageio

# Models
img_model_id = "runwayml/stable-diffusion-v1-5"
video_model_id = "stabilityai/stable-video-diffusion-img2vid-xt"

# Load image generation pipeline (text → image)
img_pipe = StableDiffusionPipeline.from_pretrained(
    img_model_id,
    torch_dtype=torch.float16,
).to("cuda")

# Load video generation pipeline (image → video)
vid_pipe = StableVideoDiffusionPipeline.from_pretrained(
    video_model_id,
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# Generate image from text prompt
def generate_image(prompt, output_image_path):
    generator = torch.Generator("cuda").manual_seed(42)
    image = img_pipe(prompt, guidance_scale=7.5, generator=generator).images[0]
    image.save(output_image_path)
    print(f"✅ Image saved at {output_image_path}")
    return image

# Generate video from an image
def generate_video_from_image(image, output_video_path, num_frames=25, fps=7):
    generator = torch.Generator("cuda").manual_seed(42)
    frames = vid_pipe(
        image=image,
        num_frames=num_frames,
        generator=generator,
        decode_chunk_size=8
    ).frames

    # Debug: Print the shape of the frames
    for i, frame in enumerate(frames):
        print(f"Frame {i} shape: {frame.shape}")

    # Ensure frames have the correct number of channels
    frames = [frame if frame.shape[2] in [1, 2, 3, 4] else frame[:, :, :3] for frame in frames]

    imageio.mimsave(output_video_path, frames, fps=fps)
    print(f"✅ Video saved at {output_video_path}")

# Full pipeline: text prompt → image → video
def full_pipeline(prompt, image_path, video_path):
    generated_image = generate_image(prompt, image_path)
    generate_video_from_image(generated_image, video_path)

# Run the pipeline
if __name__ == "__main__":
    text_prompt = "An astronaut meditating on a serene, colorful alien planet"
    output_image_path = "generated_image.png"
    output_video_path = "generated_video.mp4"

    full_pipeline(text_prompt, output_image_path, output_video_path)
