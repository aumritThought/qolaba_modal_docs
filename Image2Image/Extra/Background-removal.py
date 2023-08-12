from modal import Image, Stub, method


def download_models():
    import cv2

    from PIL import Image
    from transparent_background import Remover

    remover = Remover()

stub = Stub("Background_removal"+"_image2image")
image = (
    Image.debian_slim(python_version="3.10")
    .run_commands([
        "apt-get update && apt-get install ffmpeg libsm6 libxext6  -y",

        "pip install torch torchvision opencv-python timm tqdm kornia gdown pyvirtualcam transparent-background"])
    ).run_function(
            download_models,
            gpu="a10g"
        )

stub.image = image

@stub.cls(gpu="a10g", container_idle_timeout=600, memory=10240)
class stableDiffusion:  
    def __enter__(self):
        import cv2
        
        from PIL import Image
        from transparent_background import Remover

        # Load model
        self.remover = Remover()

    @method()
    def run_inference(self, img, bg_img=None, bg_color=False, r_color=255, g_color=0, b_color=0, blur=False):
        import random, os
        from PIL import Image
        import requests, io
        try:
            response = requests.get(bg_img)
            bg_img = Image.open(io.BytesIO(response.content))
            bg_img=bg_img.resize(img.size)
        except:
            bg_img=None
        
        if(blur==True):
            image = self.remover.process(img, type='blur')
            return {"images":[Image.fromarray(image).convert('RGB')],  "Has_NSFW_Content":[False]*1}
        elif(not(bg_img==None)):
            num = random.random()
            name=str(num)+".png"
            bg_img.save(name)
            image = self.remover.process(img, type=name) # use another image as a background
            os.remove(name)
            return {"images":[Image.fromarray(image).convert('RGB')],  "Has_NSFW_Content":[False]*1}
        elif(bg_color==True):
            image = self.remover.process(img, type=str([r_color, g_color, b_color]))
            return {"images":[Image.fromarray(image).convert('RGB')],  "Has_NSFW_Content":[False]*1}
        else:
            out = self.remover.process(img, type='white')
            return {"images":[Image.fromarray(out).convert('RGB')],  "Has_NSFW_Content":[False]*1}