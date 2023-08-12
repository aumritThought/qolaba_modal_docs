from modal import Image, Secret, Stub, method

model_schema={
    "model_id":"",
    "container_idle_timeout":600,
    "name":"SDXL_image2image",
}
stub = Stub(model_schema["name"])
image = (
    Image.debian_slim(python_version="3.10")
    .pip_install("Pillow", "requests")
    )

stub.image = image

@stub.cls(container_idle_timeout=model_schema["container_idle_timeout"])
class stableDiffusion:
    @method()
    def run_inference(self, img, prompt,guidance_scale,batch, strength, style_preset, height, width):
        import base64, os, requests, io
        from PIL import Image
        # height, width = hwcombined.split("X")
        engine_id = "stable-diffusion-xl-1024-v1-0"
        api_host = os.getenv("API_HOST", "https://api.stability.ai")
        api_key = "sk-q7ueICsPrJJcrYmXV0Ey4Gm7SGirnMyIbXFE6Ndjj1AjJM0i"
        image=img.resize((int(width),int(height)))
        filtered_image = io.BytesIO()
        image.save(filtered_image, "JPEG")

        response = requests.post(
            f"{api_host}/v1/generation/{engine_id}/image-to-image",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            files={
                "init_image": filtered_image.getvalue()
            },
            data={
                "image_strength": 1-strength,
                "init_image_mode": "IMAGE_STRENGTH",
                "text_prompts[0][text]": prompt,
                "cfg_scale": guidance_scale,
                "samples": batch,
                "steps": 30,
                "style_preset":style_preset
            }
        )

        Has_NSFW_Content=[False]*batch
        if response.status_code != 200:
            if(response.json()["message"]=="Invalid prompts detected"):
                Has_NSFW_Content=[True]*batch
                return {"images":None,  "Has_NSFW_Content":Has_NSFW_Content}
            else:
                raise Exception("Non-200 response: " + str(response.text))
        else:
            data = response.json()
            images=[]
            for i, image in enumerate(data["artifacts"]):
                images.append(Image.open(io.BytesIO(base64.b64decode(image["base64"]))))


            return {"images":images,  "Has_NSFW_Content":Has_NSFW_Content}

