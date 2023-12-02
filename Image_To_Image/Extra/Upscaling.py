from modal import Image, Stub, method

def download_models():
    import os 
    os.chdir("../Real-ESRGAN")
    os.system("python setup.py develop")
    

def download_models1():
    import os
    os.chdir("../Real-ESRGAN")
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.download_util import load_file_from_url
    from gfpgan import GFPGANer
    from realesrgan import RealESRGANer
    
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4
    file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    model_path = load_file_from_url(
                    url=file_url[0], model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
    upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=None,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False,
            gpu_id=0)
    
    face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=4,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)

stub = Stub("upscaling_image2image")
image = (Image.debian_slim(python_version="3.11")
    .run_commands([
        "apt-get update --fix-missing",
        "pip3 install torch torchvision torchaudio",
        "apt-get update && apt-get install ffmpeg libsm6 libxext6 -y", 
        "apt-get -y install git",
        "git clone https://github.com/xinntao/Real-ESRGAN.git",
        "pip install basicsr",
        "pip install facexlib",
        "pip install gfpgan",
        "pip install -r Real-ESRGAN/requirements.txt",
        ])).run_function(
    download_models,
    gpu="a10g"
).run_function(
    download_models1,
    gpu="a10g"
)

stub.image = image

@stub.cls(gpu="a10g", container_idle_timeout=200)
class stableDiffusion:
    def __enter__(self):
        import time
        st= time.time()
        import os 
        os.chdir("../Real-ESRGAN")
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.utils.download_util import load_file_from_url
        from gfpgan import GFPGANer
        from realesrgan import RealESRGANer

        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
        model_path = load_file_from_url(
                        url=file_url[0], model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
    
        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=None,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False,
            gpu_id=0)

        
        self.face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=4,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=self.upsampler)
        self.container_execution_time=time.time()-st

    
    def generate_image_urls(self, image_data):
        import io, base64, requests
        url = "https://qolaba-server-production-caff.up.railway.app/api/v1/uploadToCloudinary/image"
        image_urls=[]
        for im in range(0, len(image_data["images"])):
            filtered_image = io.BytesIO()
            if(image_data["Has_NSFW_Content"][im]):
                pass
            else:
                image_data["images"][im].save(filtered_image, "JPEG")
                myobj = {
                        "image":"data:image/png;base64,"+(base64.b64encode(filtered_image.getvalue()).decode("utf8"))
                    }
                rps = requests.post(url, json=myobj, headers={'Content-Type': 'application/json'})
                im_url=rps.json()["data"]["secure_url"]
                image_urls.append(im_url)
        return image_urls

    @method()
    def run_inference(self, file_url, upscale, face_upsample):
        import time
        st=time.time()
        from gfpgan import GFPGANer
        import numpy as np
        from PIL import Image

        if face_upsample:
            self.face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=upscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=self.upsampler)
            _, _, output = self.face_enhancer.enhance(np.array(file_url), has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = self.upsampler.enhance(np.array(file_url), outscale=upscale)
        restored_img = Image.fromarray(output)
        image_data = {"images" :  [restored_img], "Has_NSFW_Content" : [False]*1}
        image_urls =self.generate_image_urls(image_data)
        
        self.runtime=time.time()-st
        return {"result":image_urls,  
                "Has_NSFW_Content":  [False]*1, 
                "time": {"startup_time" : self.container_execution_time, "runtime":self.runtime}}
