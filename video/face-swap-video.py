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
        
        
stub = Stub("Face_swap"+"_video")
image = (
    Image.debian_slim(python_version="3.10")
    .run_commands([
        
        "apt-get update && apt-get install ffmpeg libsm6 libxext6 tk -y",
        "apt-get -y install git",
        "git clone https://github.com/s0md3v/roop.git",
        "pip install -r roop/requirements.txt",
        "pip install ffmpeg-python imageio==2.19.3 imageio-ffmpeg==0.4.7",
])
    ).run_function(
            download_models,
            gpu="a10g"
        )

stub.image = image

@stub.cls(gpu="a10g", container_idle_timeout=600, memory=10240, timeout=900)
class stableDiffusion:  
    def __enter__(self):
        import sys
        sys.path.insert(0, '/roop')
        from roop.processors.frame.core import get_frame_processors_modules
        import sys, shutil, threading
        import roop.globals
        import cv2
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

    @method()
    def run_inference(self, video_url, bg_img=None):
        import sys
        sys.path.insert(0, '/roop')
        import sys, threading
        import roop.globals
        from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
        import cv2
        import numpy as np
        from PIL import Image
        import requests, io, imageio, os, time, base64

        def write_video(file_path, frames, fps):

            frames = [frame for frame in frames if frame.size == frames[0].size]

            writer = imageio.get_writer(file_path, fps=int(fps), macro_block_size=None)

            for frame in frames:
                np_frame = np.array(frame)
                writer.append_data(np_frame)

            writer.close()


        try:
            response = requests.get(bg_img)
            s_img = Image.open(io.BytesIO(response.content))
            
        except:
            s_img=None
        if(s_img==None):
            raise ValueError("Source Image not found")
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

                vcap = cv2.VideoCapture(video_url)
                fps = vcap.get(cv2.CAP_PROP_FPS)
                length = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))

                if(length>400):
                    raise Exception("Video is too large for face swapping")
                
                all_frames=[]

                while(True):
                    ret, frame = vcap.read()
                    if frame is not None:
                        frame =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        temp_frame = np.array(frame)
                        if(temp_frame.shape[1]>=temp_frame.shape[0]):
                            height=1024
                            width=((int(temp_frame.shape[0]*1024/temp_frame.shape[1]))//64)*64
                        else:
                            width=1024
                            height=((int(temp_frame.shape[1]*1024/temp_frame.shape[0]))//64)*64

                        temp_frame=cv2.resize(temp_frame, (height, width))
                        try:
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
                        except:
                            pass
                        all_frames.append(temp_frame)
                    else:
                        print ("Frame is None")
                        break

                vcap.release()
                cv2.destroyAllWindows()


                if(len(all_frames)==0):
                    raise Exception("Failed in face swapping")
                
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
            except:
                raise ValueError("Not abled to detect face")
            