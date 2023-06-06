from pathlib import Path

from modal import Image, Secret, Stub, web_endpoint

from fastapi import Security, Depends, HTTPException, UploadFile, status, Request
from typing import  Optional 
import io
from fastapi.responses import StreamingResponse
from starlette.status import HTTP_403_FORBIDDEN
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

auth_scheme = HTTPBearer()

stub = Stub("codeformer")
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
)

stub.image = image


@stub.function(gpu="a10g",secret=Secret.from_name("API_UPSCALING_KEY"))
@web_endpoint(label="codeformer", method="POST")
def image_upscale(file : UploadFile,
    w : Optional[float]= 0.7,
    upscale: Optional[int] = 2,
    has_aligned: Optional[bool] = False,
    only_center_face: Optional[bool] = False,
    draw_box: Optional[bool] = False,
    detection_model:  Optional[str] = "retinaface_resnet50",
    bg_upsampler:  Optional[str] = "realesrgan",
    face_upsample: Optional[bool] = True,
    api_key: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    import os
    import cv2
    import argparse
    import glob
    import torch
    from torchvision.transforms.functional import normalize
    from basicsr.utils import imwrite, img2tensor, tensor2img
    from basicsr.utils.download_util import load_file_from_url
    from basicsr.utils.misc import gpu_is_available, get_device
    from facelib.utils.face_restoration_helper import FaceRestoreHelper
    from facelib.utils.misc import is_gray
    from typing import List, Optional, Union
    from basicsr.utils.registry import ARCH_REGISTRY
    from PIL import Image
    import numpy

    if api_key.credentials != os.environ["API_UPSCALING_KEY"]:
        raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    else:   
        list_img=[".jpg",".png"]
        if any([x in file.filename for x in list_img]):
            

            
            pretrain_model_url = {
                'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
            }
            device = get_device()

            def set_realesrgan():
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
                    half=use_half
                )

                if not gpu_is_available():  # CPU
                    import warnings
                    warnings.warn('Running on CPU now! Make sure your PyTorch version matches your CUDA.'
                                    'The unoptimized RealESRGAN is slow on CPU. '
                                    'If you want to disable it, please remove `--bg_upsampler` and `--face_upsample` in command.',
                                    category=RuntimeWarning)
                return upsampler
            
            content=file.file.read()
            image = Image.open(io.BytesIO(content))
            image = numpy.array(image) 
            # Convert RGB to BGR 
            image = image[:, :, ::-1].copy() 

            if bg_upsampler == 'realesrgan':
                bg_upsampler = set_realesrgan()
            else:
                bg_upsampler = None

            # ------------------ set up face upsampler ------------------
            if face_upsample:
                if bg_upsampler is not None:
                    face_upsampler = bg_upsampler
                else:
                    face_upsampler = set_realesrgan()
            else:
                face_upsampler = None

            # ------------------ set up CodeFormer restorer -------------------
            net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                                                    connect_list=['32', '64', '128', '256']).to(device)
            
            # ckpt_path = 'weights/CodeFormer/codeformer.pth'
            ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'], 
                                            model_dir='weights/CodeFormer', progress=True, file_name=None)
            checkpoint = torch.load(ckpt_path)['params_ema']
            net.load_state_dict(checkpoint)
            net.eval()

            # ------------------ set up FaceRestoreHelper -------------------
            # large det_model: 'YOLOv5l', 'retinaface_resnet50'
            # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
            if not has_aligned: 
                print(f'Face detection model: {detection_model}')
            if bg_upsampler is not None: 
                print(f'Background upsampling: True, Face upsampling: {face_upsample}')
            else:
                print(f'Background upsampling: False, Face upsampling: {face_upsample}')

            face_helper = FaceRestoreHelper(
                upscale,
                face_size=512,
                crop_ratio=(1, 1),
                det_model = detection_model,
                save_ext='png',
                use_parse=True,
                device=device)

            face_helper.clean_all()

            if has_aligned: 
                image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
                face_helper.is_gray = is_gray(image, threshold=10)
                if face_helper.is_gray:
                    print('Grayscale input: True')
                face_helper.cropped_faces = [image]
            else:
                face_helper.read_image(image)
                # get face landmarks for each face
                num_det_faces = face_helper.get_face_landmarks_5(
                    only_center_face=only_center_face, resize=640, eye_dist_threshold=5)
                print(f'\tdetect {num_det_faces} faces')
                # align and warp each face
                face_helper.align_warp_face()

            for idx, cropped_face in enumerate(face_helper.cropped_faces):
                cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
                normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

                try:
                    with torch.no_grad():
                        output = net(cropped_face_t, w=w, adain=True)[0]
                        restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                    del output
                    torch.cuda.empty_cache()
                except Exception as error:
                    print(f'\tFailed inference for CodeFormer: {error}')
                    restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

                restored_face = restored_face.astype('uint8')
                face_helper.add_restored_face(restored_face, cropped_face)

            if not has_aligned:
                    # upsample the background
                if bg_upsampler is not None:
                        # Now only support RealESRGAN for upsampling background
                    bg_img = bg_upsampler.enhance(image, outscale=upscale)[0]
                else:
                    bg_img = None
                face_helper.get_inverse_affine(None)
                    # paste each restored face to the input image
                if face_upsample and face_upsampler is not None: 
                    restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=draw_box, face_upsampler=face_upsampler)
                else:
                    restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=draw_box)
            restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
            restored_img = Image.fromarray(restored_img)
            filtered_image = io.BytesIO()
            restored_img.save(filtered_image, "JPEG")
            filtered_image.seek(0)
            return StreamingResponse(filtered_image, media_type="image/jpeg")
        else:
            return "invalid file"
