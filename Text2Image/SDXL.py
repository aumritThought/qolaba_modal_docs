from modal import Image, Secret, Stub, method

stub = Stub("SDXL")
image = (
    Image.debian_slim(python_version="3.10")
    .pip_install("Pillow", "requests")
    )

stub.image = image

@stub.cls()
class stableDiffusion:
    @method()
    def run_inference(self,prompt,height,width,num_inference_steps,guidance_scale,negative_prompt,batch):

        from PIL import Image
        import base64, os, requests,io
        

        engine_id = "stable-diffusion-xl-beta-v2-2-2"
        api_host = os.getenv('API_HOST', 'https://api.stability.ai')
        api_key = "sk-q7ueICsPrJJcrYmXV0Ey4Gm7SGirnMyIbXFE6Ndjj1AjJM0i"

        if api_key is None:
            raise Exception("Missing Stability API key.")

        response = requests.post(
            f"{api_host}/v1/generation/{engine_id}/text-to-image",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json={
                "text_prompts": [
                    {
                        "text": prompt
                    }
                ],
                "cfg_scale": guidance_scale,
                "clip_guidance_preset": "FAST_BLUE",
                "height": height,
                "width": width,
                "samples": batch,
                "steps": num_inference_steps,
            },
        )

        if response.status_code != 200:
            raise Exception("Non-200 response: " + str(response.text))

        data = response.json()
        images=[]
        for i, image in enumerate(data["artifacts"]):
            images.append(Image.open(io.BytesIO(base64.b64decode(image["base64"]))))

        return images

