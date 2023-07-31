from modal import Image, Stub, method

def download_models():
    from diffusers import KandinskyImg2ImgPipeline, KandinskyPriorPipeline
    import torch
    pipe_prior = KandinskyPriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16
        )
    pipe_prior.to("cuda")

        # create img2img pipeline
    pipe = KandinskyImg2ImgPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16)
    pipe.to("cuda")

stub = Stub("kandinsky_image2image")
image = (
    Image.debian_slim(python_version="3.10")
    .run_commands([
        "pip install diffusers transformers accelerate opencv-python Pillow"
                   ])
    ).run_function(
            download_models,
            gpu="a10g"
        )

stub.image = image

@stub.cls(gpu="a10g", container_idle_timeout=100, memory=10240)
class stableDiffusion:   
    def __enter__(self):
        from diffusers import KandinskyImg2ImgPipeline, KandinskyPriorPipeline
        import torch
        self.pipe_prior = KandinskyPriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16
        )
        self.pipe_prior.to("cuda")

        # create img2img pipeline
        self.pipe = KandinskyImg2ImgPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16)
        self.pipe.to("cuda")

    @method()
    def run_inference(self, img, prompt,guidance_scale,negative_prompt,batch, strength):
        print( img, prompt,guidance_scale,negative_prompt,batch, strength)
        image_embeds, negative_image_embeds = self.pipe_prior(prompt, negative_prompt).to_tuple()

        image = self.pipe(
            prompt,
            image=img,
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
            strength=strength,
            num_images_per_prompt=batch,
            height=img.size[1],
            width=img.size[0],
        ).images

        return {"images":image,  "Has_NSFW_Content":[False]*batch}
