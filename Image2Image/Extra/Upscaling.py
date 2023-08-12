from pathlib import Path

from modal import Image, Secret, Stub, web_endpoint, method

from fastapi import Security, Depends, HTTPException, UploadFile, status, Request
from typing import  Optional 
import io
from fastapi.responses import StreamingResponse
from starlette.status import HTTP_403_FORBIDDEN
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

def download_models():
    class stableDiffusion:
        def set_realesrgan(self):
            import torch
            from basicsr.utils.misc import gpu_is_available
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from basicsr.utils.realesrgan_utils import RealESRGANer
            use_half = False
            if torch.cuda.is_available(): # set False in CPU/MPS mode
                no_half_gpu_list = ['1650', '1660'] # set False for GPUs that don't support f16
                if not True in [gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list]:
                        use_half = True
            model = RRDBNet(
                        num_in_ch=3,
                        num_out_ch=3,
                        num_feat=64,
                        num_block=23,
                        num_grow_ch=32,
                        scale=2,
                    )
            upsampler = RealESRGANer(
                        scale=2,
                        model_path="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
                        model=model,
                        tile=400,
                        tile_pad=40,
                        pre_pad=0,
                        half=use_half)

            if not gpu_is_available():  # CPU
                import warnings
                warnings.warn('Running on CPU now! Make sure your PyTorch version matches your CUDA.'
                                        'The unoptimized RealESRGAN is slow on CPU. '
                                        'If you want to disable it, please remove `--bg_upsampler` and `--face_upsample` in command.',
                                        category=RuntimeWarning)
            return upsampler
        
        def __init__(self):
            

            import torch
            from basicsr.utils.download_util import load_file_from_url
            from basicsr.utils.misc import get_device
            
            
            from basicsr.utils.registry import ARCH_REGISTRY

            
            self.w = 0.7
            self.has_aligned=False
            self.only_center_face=False
            self.draw_box = False
            
            self.bg_upsampler = "realesrgan"
            pretrain_model_url = {
                    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
                }
            self.device = get_device()
            

            if self.bg_upsampler == 'realesrgan':
                self.bg_upsampler = self.set_realesrgan()
            else:
                self.bg_upsampler = None
                # ------------------ set up face upsampler ------------------
            
                # ------------------ set up CodeFormer restorer -------------------
            self.net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                                                        connect_list=['32', '64', '128', '256']).to(self.device)

                # ckpt_path = 'weights/CodeFormer/codeformer.pth'
            ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'], 
                                                model_dir='weights/CodeFormer', progress=True, file_name=None)
            checkpoint = torch.load(ckpt_path)['params_ema']
            self.net.load_state_dict(checkpoint)
            self.net.eval()

    a=stableDiffusion()

stub = Stub("codeformer_image2image")
image=Image.from_dockerhub(
    "nvidia/cuda:11.8.0-devel-ubuntu22.04",
    setup_dockerfile_commands=[ "RUN apt-get update --fix-missing",
                                "RUN apt-get install -y python3-pip",
                                "RUN apt install python-is-python3",
                                "RUN pip3 install torch torchvision torchaudio",
                                "RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y", 
                                "RUN apt-get -y install git",
                                "RUN git clone https://github.com/sczhou/CodeFormer.git",
                                "WORKDIR CodeFormer",
                                "RUN pip install -r requirements.txt",
                                "RUN python3 basicsr/setup.py develop",
                                "RUN python3 scripts/download_pretrained_models.py facelib",
                                "RUN python3 scripts/download_pretrained_models.py CodeFormer",
                               ]
).run_function(
    download_models,
    gpu="a10g"
)

stub.image = image

@stub.cls(gpu="a10g", container_idle_timeout=600)
class stableDiffusion:
    def set_realesrgan(self):
        import torch
        from basicsr.utils.misc import gpu_is_available
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.utils.realesrgan_utils import RealESRGANer
        use_half = False
        if torch.cuda.is_available(): # set False in CPU/MPS mode
            no_half_gpu_list = ['1650', '1660'] # set False for GPUs that don't support f16
            if not True in [gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list]:
                    use_half = True
        model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=2,
                )
        upsampler = RealESRGANer(
                    scale=2,
                    model_path="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
                    model=model,
                    tile=400,
                    tile_pad=40,
                    pre_pad=0,
                    half=use_half)

        if not gpu_is_available():  # CPU
            import warnings
            warnings.warn('Running on CPU now! Make sure your PyTorch version matches your CUDA.'
                                    'The unoptimized RealESRGAN is slow on CPU. '
                                    'If you want to disable it, please remove `--bg_upsampler` and `--face_upsample` in command.',
                                    category=RuntimeWarning)
        return upsampler
    
    def __enter__(self):
        

        import torch
        from basicsr.utils.download_util import load_file_from_url
        from basicsr.utils.misc import get_device
        
        
        from basicsr.utils.registry import ARCH_REGISTRY

        
        self.w = 0.7
        self.has_aligned=False
        self.only_center_face=False
        self.draw_box = False
        
        self.bg_upsampler = "realesrgan"
        pretrain_model_url = {
                'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
            }
        self.device = get_device()
        

        if self.bg_upsampler == 'realesrgan':
            self.bg_upsampler = self.set_realesrgan()
        else:
            self.bg_upsampler = None
            # ------------------ set up face upsampler ------------------
        
            # ------------------ set up CodeFormer restorer -------------------
        self.net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                                                    connect_list=['32', '64', '128', '256']).to(self.device)

            # ckpt_path = 'weights/CodeFormer/codeformer.pth'
        ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'], 
                                            model_dir='weights/CodeFormer', progress=True, file_name=None)
        checkpoint = torch.load(ckpt_path)['params_ema']
        self.net.load_state_dict(checkpoint)
        self.net.eval()
            # ------------------ set up FaceRestoreHelper -------------------
            # large det_model: 'YOLOv5l', 'retinaface_resnet50'
            # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'

        

    @method()
    def run_inference(self, image, upscale, face_upsample):
        print(image.size)
        from PIL import Image
        from basicsr.utils import imwrite, img2tensor, tensor2img
        from torchvision.transforms.functional import normalize
        from facelib.utils.face_restoration_helper import FaceRestoreHelper
        import torch
        from facelib.utils.misc import is_gray
        import cv2
        import numpy
        image=numpy.array(image)
        print(face_upsample)
        if face_upsample:
            if self.bg_upsampler is not None:
                face_upsampler = self.bg_upsampler
            else:
                face_upsampler = self.set_realesrgan()
        else:
            face_upsampler = None
        detection_model = "retinaface_resnet50"
        self.face_helper = FaceRestoreHelper(
                upscale,
                face_size=512,
                crop_ratio=(1, 1),
                det_model = detection_model,
                save_ext='png',
                use_parse=True,
                device=self.device)
        self.face_helper.clean_all()
           
        if self.has_aligned: 
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
            self.face_helper.is_gray = is_gray(image, threshold=10)
            if self.face_helper.is_gray:
                print('Grayscale input: True')
            self.face_helper.cropped_faces = [image]
        else:
            self.face_helper.read_image(image)
                # get face landmarks for each face
            num_det_faces = self.face_helper.get_face_landmarks_5(
                    only_center_face=self.only_center_face, resize=640, eye_dist_threshold=5)
            print(f'\tdetect {num_det_faces} faces')
                # align and warp each face
            self.face_helper.align_warp_face()
        for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)
            # try:
            with torch.no_grad():
                output = self.net(cropped_face_t, w=self.w, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            del output
            torch.cuda.empty_cache()
            # except Exception as error:
                # print(f'\tFailed inference for CodeFormer: {error}')
                # restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))
            restored_face = restored_face.astype('uint8')
            self.face_helper.add_restored_face(restored_face, cropped_face)
        if not self.has_aligned:
                    # upsample the background
            if self.bg_upsampler is not None:
                        # Now only support RealESRGAN for upsampling background
                bg_img = self.bg_upsampler.enhance(image, outscale=upscale)[0]
            else:
                bg_img = None
            self.face_helper.get_inverse_affine(None)
                    # paste each restored face to the input image
            if face_upsample and face_upsampler is not None: 
                restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=self.draw_box, face_upsampler=face_upsampler)
            else:
                restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=self.draw_box)
        # restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
        restored_img = Image.fromarray(restored_img)
        print(restored_img.size)
        return {"images":[restored_img],  "Has_NSFW_Content":[False]}
