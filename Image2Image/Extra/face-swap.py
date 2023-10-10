from modal import Image, Stub, method

def download_models():
    import sys
    sys.path.insert(0, '/roop')
    from roop.processors.frame.core import get_frame_processors_modules
    import sys, shutil, threading
    import roop.globals
    # from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
    import cv2
    import insightface
    import onnxruntime
    from gfpgan.utils import GFPGANer
    from roop.utilities import resolve_relative_path, conditional_download
    roop.globals.source_path = "1.png"
    roop.globals.target_path = "2.png"
    roop.globals.output_path = "3.png"
    roop.globals.headless = roop.globals.source_path is not None and roop.globals.target_path is not None and roop.globals.output_path is not None
    roop.globals.frame_processors = ['face_swapper', 'face_enhancer']
    roop.globals.many_faces = False
    roop.globals.reference_face_position = 0
    roop.globals.reference_frame_number = 0
    roop.globals.similar_face_distance = 0.85
    roop.globals.temp_frame_format = 'png'
    roop.globals.temp_frame_quality = 0
    THREAD_LOCK = threading.Lock()

    execution_provider=['cuda']
    def encode_execution_providers(execution_providers):
        return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


    def decode_execution_providers(execution_providers):
        return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
                    if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]

    roop.globals.execution_providers = decode_execution_providers(execution_provider)

    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/thebiglaskowski/inswapper_128.onnx/resolve/main/inswapper_128.onnx'])

    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        try:
            if not frame_processor.pre_check():
                return
        except:
            pass
        
    FACE_ANALYSER=None

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=roop.globals.execution_providers)
            FACE_ANALYSER.prepare(ctx_id=0)
    FACE_SWAPPER=None
  

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            execution_providers = decode_execution_providers(execution_provider)
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=execution_providers)

    FACE_ENHANCER=None
    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = resolve_relative_path('../models/GFPGANv1.4.pth')
                # todo: set models path -> https://github.com/TencentARC/GFPGAN/issues/399
            FACE_ENHANCER = GFPGANer(model_path=model_path, upscale=1, device='cuda')
        
        
stub = Stub("Face_swap"+"_image2image")
image = (
    Image.debian_slim(python_version="3.10")
    .run_commands([
        
        "apt-get update && apt-get install ffmpeg libsm6 libxext6 tk -y",
        "apt-get -y install git",
        "git clone https://github.com/s0md3v/roop.git",
        "pip install -r roop/requirements.txt",
        "pip install ffmpeg-python imageio==2.19.3 imageio-ffmpeg==0.4.7",
        "pip install diffusers transformers accelerate opencv-python Pillow xformers"
])
    ).run_function(
            download_models,
            gpu="a10g"
        )

stub.image = image

@stub.cls(gpu="a10g", container_idle_timeout=600, memory=10240)
class stableDiffusion:  
    def __enter__(self):
        import time
        st= time.time()
        import sys
        sys.path.insert(0, '/roop')
        from roop.processors.frame.core import get_frame_processors_modules
        import sys, threading
        import roop.globals
        import insightface
        import onnxruntime
        from gfpgan.utils import GFPGANer
        from roop.utilities import resolve_relative_path
        
        roop.globals.frame_processors = ['face_swapper', 'face_enhancer']
        roop.globals.many_faces = True
        roop.globals.reference_face_position = 0
        roop.globals.reference_frame_number = 0
        roop.globals.similar_face_distance = 0.85
        roop.globals.temp_frame_format = 'png'
        roop.globals.temp_frame_quality = 0
        THREAD_LOCK = threading.Lock()

        execution_provider=['cuda']
        def encode_execution_providers(execution_providers):
            return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


        def decode_execution_providers(execution_providers):
            return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
                        if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]

        roop.globals.execution_providers = decode_execution_providers(execution_provider)

        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            if not frame_processor.pre_check():
                return
            
        self.FACE_ANALYSER=None

        with THREAD_LOCK:
            if self.FACE_ANALYSER is None:
                self.FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=roop.globals.execution_providers)
                self.FACE_ANALYSER.prepare(ctx_id=0)
        self.FACE_SWAPPER=None
    

        with THREAD_LOCK:
            if self.FACE_SWAPPER is None:
                execution_providers = decode_execution_providers(execution_provider)
                model_path = resolve_relative_path('../models/inswapper_128.onnx')
                self.FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=execution_providers)

        self.FACE_ENHANCER=None
        with THREAD_LOCK:
            if self.FACE_ENHANCER is None:
                model_path = resolve_relative_path('../models/GFPGANv1.4.pth')
                    # todo: set models path -> https://github.com/TencentARC/GFPGAN/issues/399
                self.FACE_ENHANCER = GFPGANer(model_path=model_path, upscale=1, device='cuda')

        self.container_execution_time=time.time()-st

    def generate_image_urls(self, image_data):
        import io, base64, requests
        url = "https://qolaba-server-development-2303.up.railway.app/api/v1/uploadToCloudinary/image"
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
    def run_inference(self, img, bg_img=None):
        import sys
        sys.path.insert(0, '/roop')
        import sys, threading
        import roop.globals
        from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
        import cv2, torch
        import numpy as np
        from PIL import Image
        import time
        st = time.time()
        import requests, io
        try:
            response = requests.get(bg_img)
            s_img = Image.open(io.BytesIO(response.content))
        except:
            s_img=None
        if(s_img==None):
            raise Exception("Not able to fetch the Image using URL" , "Provide Proper Image URL")
        else:
            try:
                s_img=np.array(s_img)
                if(s_img.shape[1]>=s_img.shape[0]):
                    height=1024
                    width=((int(s_img.shape[0]*1024/s_img.shape[1]))//64)*64
                else:
                    width=1024
                    height=((int(s_img.shape[1]*1024/s_img.shape[0]))//64)*64
                s_img=cv2.resize(s_img, (height, width))
                source_face = get_one_face(s_img)
                temp_frame = np.array(img)
                if(temp_frame.shape[1]>=temp_frame.shape[0]):
                    height=1024
                    width=((int(temp_frame.shape[0]*1024/temp_frame.shape[1]))//64)*64
                else:
                    width=1024
                    height=((int(temp_frame.shape[1]*1024/temp_frame.shape[0]))//64)*64
                temp_frame=cv2.resize(temp_frame, (height, width))
                reference_face = get_one_face(temp_frame, roop.globals.reference_face_position)

                if roop.globals.many_faces:
                    roop.globals.many_faces = get_many_faces(temp_frame)
                    if roop.globals.many_faces:
                        for target_face in roop.globals.many_faces:
                            temp_frame = self.FACE_SWAPPER.get(temp_frame, target_face, source_face, paste_back=True)
                else:
                    target_face = find_similar_face(temp_frame, reference_face)
                    if target_face:
                        temp_frame = self.FACE_SWAPPER.get(temp_frame, target_face, source_face, paste_back=True)
                start_x, start_y, end_x, end_y = map(int, target_face['bbox'])
                padding_x = int((end_x - start_x) * 0.5)
                padding_y = int((end_y - start_y) * 0.5)
                start_x = max(0, start_x - padding_x)
                start_y = max(0, start_y - padding_y)
                end_x = max(0, end_x + padding_x)
                end_y = max(0, end_y + padding_y)
                temp_face = temp_frame[start_y:end_y, start_x:end_x]
                if temp_face.size:
                    with threading.Semaphore():
                        _, _, temp_face = self.FACE_ENHANCER.enhance(
                                temp_face,
                                paste_back=True
                            )
                    temp_frame[start_y:end_y, start_x:end_x] = temp_face
                torch.cuda.empty_cache()
                image=[Image.fromarray(temp_frame)] 
                image_data = {"images" :  image, "Has_NSFW_Content" : [False]*1}
                image_urls =self.generate_image_urls(image_data)
                
                self.runtime=time.time()-st
                return {"result":image_urls,  
                        "Has_NSFW_Content":[False]*1, 
                        "time": {"startup_time" : self.container_execution_time, "runtime":self.runtime}}
            except:
                raise Exception("Not able to detect face or Input image is invalid" , "Provide Proper Image")
            