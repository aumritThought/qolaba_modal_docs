from pathlib import Path

from modal import Image, Secret, Stub, web_endpoint, method

from fastapi import Security, Depends, HTTPException, UploadFile, status, Request
from typing import  Optional 
import io
from fastapi.responses import StreamingResponse
from starlette.status import HTTP_403_FORBIDDEN
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

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

stub = Stub("upscaling_video")
image = (Image.debian_slim(python_version="3.10")
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
        "pip install ffmpeg-python imageio==2.19.3 imageio-ffmpeg==0.4.7",
        ])).run_function(
    download_models,
    gpu="a10g"
).run_function(
    download_models1,
    gpu="a10g"
)

stub.image = image

@stub.cls(gpu="a10g", container_idle_timeout=600, timeout=900)
class stableDiffusion:
    def __enter__(self):
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
            

    @method()
    def run_inference(self, video_url, upscale, face_upsample):

        from gfpgan import GFPGANer
        import numpy as np
        from PIL import Image
        import cv2, imageio, time, base64, requests, os


        def write_video(file_path, frames, fps):

            frames = [frame for frame in frames if frame.size == frames[0].size]

            writer = imageio.get_writer(file_path, fps=int(fps), macro_block_size=None)

            for frame in frames:
                np_frame = np.array(frame)
                writer.append_data(np_frame)

            writer.close()

        vcap = cv2.VideoCapture(video_url)
        fps = vcap.get(cv2.CAP_PROP_FPS)
        length = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))

        if(length>400):
            raise Exception("Video is too large for upscaling")

        if face_upsample:
            self.face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=upscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=self.upsampler)

        all_frames=[]
        while(True):
            ret, frame = vcap.read()
            if frame is not None:
                frame =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if face_upsample:
                    _, _, output = self.face_enhancer.enhance(np.array(frame), has_aligned=False, only_center_face=False, paste_back=True)
                else:
                    output, _ = self.upsampler.enhance(np.array(frame), outscale=upscale)
                all_frames.append(output)
            else:
                print ("Frame is None")
                break

        vcap.release()
        cv2.destroyAllWindows()

        if(len(all_frames)==0):
            raise Exception("Failed in Upscaling Video")
        
        video_file_name = "infinite_zoom_" + str(time.time())
        save_path = video_file_name + ".mp4"

        write_video(save_path, all_frames, fps)

        with open(save_path, "rb") as f:
            url="https://qolaba-server-development-2303.up.railway.app/api/v1/uploadToCloudinary/audiovideo"
            byte=f.read()
            myobj = {"image":"data:audio/mpeg;base64,"+(base64.b64encode(byte).decode("utf8"))}
            
            rps = requests.post(url, json=myobj, headers={'Content-Type': 'application/json'})
            video_url=rps.json()["data"]["secure_url"]
        

        os.remove(save_path)

        return {"video_url":video_url}
