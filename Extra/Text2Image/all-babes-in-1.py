from modal import Image, Secret, Stub, method

from common_code import give_model_schema, SD_common, download_models

model_schema=give_model_schema()
model_schema["name"]="all-babes-in-1_text2image"

stub = Stub(model_schema["name"])

def download_models_():
    download_models("abbiepam/all-babes-in-1")


image=Image.from_dockerfile( "Dockerfile"
).run_function(
        download_models_,
        gpu="a10g",
    )

stub.image = image

@stub.cls(gpu=model_schema["gpu"], memory=model_schema["memory"], container_idle_timeout=model_schema["container_idle_timeout"], serialized=True)
class stableDiffusion():
    def __enter__(self):
        self.model=SD_common( ", (((((Babes 1.1 , Ex7, 0.5), Ex8, 0.5), KissableLips, 0.5), SexyToons, 0.5), Babes 2.0, 0.5)")
    
    @method()
    def run_inference(self, img, prompt, height,width,num_inference_steps,guidance_scale,negative_prompt,batch, strength):
        data=self.model.run_inference(img, prompt, height,width,num_inference_steps,guidance_scale,negative_prompt,batch, strength)
        return data
