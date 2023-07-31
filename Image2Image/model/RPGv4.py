from modal import Image, Stub, method

from common_code import give_model_schema, SD_common, download_models

model_schema=give_model_schema()
model_schema["name"]="RPG-v4"+"_image2image"
model_schema["extra_prompt"]=""

stub = Stub(model_schema["name"])

def download_models_():
    download_models("danbrown/RPG-v4")


image=Image.from_dockerfile( "Dockerfile"
).run_function(
        download_models_,
        gpu="a10g",
    )

stub.image = image

@stub.cls(gpu=model_schema["gpu"], memory=model_schema["memory"], container_idle_timeout=model_schema["container_idle_timeout"], serialized=True)
class stableDiffusion():
    def __enter__(self):
        self.model=SD_common(model_schema["extra_prompt"])
    
    @method()
    def run_inference(self, img, prompt,guidance_scale,negative_prompt,batch, strength):
        data=self.model.run_inference(img, prompt,guidance_scale,negative_prompt,batch, strength)
        return data
