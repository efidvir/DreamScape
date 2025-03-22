import torch
from diffusers import StableDiffusionPipeline
import numpy as np
import cv2
import os
import subprocess
import json

# Load Stable Diffusion
sd_model = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(sd_model, torch_dtype=torch.float16).to("cuda")

# Generate Image from Text Prompt
prompt = "A futuristic cyberpunk city with neon lights"
image_path = "generated_scene.jpg"

if not os.path.exists(image_path):
    image = pipe(prompt).images[0]
    image.save(image_path)
    print(f"✅ Image saved at {image_path}")
else:
    print(f"ℹ️ Image already exists at {image_path}, skipping image generation.")

# Prepare NeRF dataset
def generate_nerf_from_image(image_path):
    nerf_dataset_path = "output_nerf"
    images_path = os.path.join(nerf_dataset_path, "images")
    os.makedirs(images_path, exist_ok=True)

    # Copy and resize the image
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (800, 800))
    scene_img_path = os.path.join(images_path, "scene.png")
    cv2.imwrite(scene_img_path, img_resized)
    print(f"✅ Scene image saved at {scene_img_path}")

    # Create minimal transforms.json required by Instant-NGP
    transforms = {
        "camera_angle_x": 0.691111,
        "frames": [
            {
                "file_path": "images/scene.png",
                "transform_matrix": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ]
            }
        ]
    }

    transforms_path = os.path.join(nerf_dataset_path, "transforms.json")
    with open(transforms_path, 'w') as f:
        json.dump(transforms, f, indent=4)
    print(f"✅ Transforms file saved at {transforms_path}")

    print("✅ NeRF dataset prepared successfully!")

# Generate the correct dataset for Instant-NGP
generate_nerf_from_image(image_path)

# Render in Instant-NGP GUI
def render_nerf():
    instant_ngp_executable = r"C:\DreamScape\Instant-NGP-for-RTX-3000-and-4000\instant-ngp.exe"
    command = [instant_ngp_executable, "--scene=output_nerf"]  # Pointing to the directory containing transforms.json

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        raise

# Open Instant-NGP GUI with prepared NeRF Scene
render_nerf()