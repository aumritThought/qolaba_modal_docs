from modal import Image, Stub, method


def download_models():
    from diffusers import DiffusionPipeline
    import torch

    pipe_prior = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16)
    pipe_prior.to("cuda")

    t2i_pipe = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16)
    t2i_pipe.to("cuda")

stub = Stub("kandinsky_text2image")
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

@stub.cls(gpu="a10g", container_idle_timeout=600, memory=10240)
class stableDiffusion:   
    def __enter__(self):
        from diffusers import DiffusionPipeline
        import torch

        self.pipe_prior = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16)
        self.pipe_prior.to("cuda")

        self.t2i_pipe = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16)
        self.t2i_pipe.to("cuda")

    @method()
    def run_inference(self,prompt,height,width,num_inference_steps,guidance_scale,negative_prompt,batch):

        image_embeds, negative_image_embeds = self.pipe_prior(prompt, negative_prompt, guidance_scale=1.0).to_tuple()

        image = self.t2i_pipe(prompt, negative_prompt=negative_prompt, image_embeds=image_embeds, negative_image_embeds=negative_image_embeds, height=height, width=width, num_images_per_prompt=batch).images


        return {"images":image,  "Has_NSFW_Content":[False]*batch}
