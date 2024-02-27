CONTROLNET_COMMANDS = [
    "apt-get update && apt-get install ffmpeg libsm6 libxext6 git -y",
    "apt-get update && apt-get install wget -y",
    "wget https://civitai.com/api/download/models/182077",
    "pip install diffusers --upgrade",
    "pip install controlnet_aux invisible_watermark transformers accelerate safetensors xformers==0.0.22 omegaconf pydantic cloudinary",
    "mv 182077 Starlight.safetensors",
]

NORMAL_CONTROLNET_COMMANDS = [
            "apt-get update && apt-get install ffmpeg libsm6 libxext6  -y",
            "pip install diffusers transformers accelerate opencv-python Pillow xformers controlnet_aux matplotlib cloudinary",
        ]
