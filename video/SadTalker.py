from modal import Image, Stub, method

model_schema={
    "memory":10240,
    "container_idle_timeout":600,
    "name":"sad_talker_video",
    "gpu":"a100"
}

stub = Stub(model_schema["name"])
image = (
    Image.debian_slim(python_version="3.10")
    .run_commands([
        "apt-get update && apt-get install ffmpeg libsm6 libxext6 git -y",
        "git clone https://github.com/Winfredy/SadTalker.git",
        "pip install -r SadTalker/requirements.txt",
        "pip install python-ffmpeg realesrgan gfpgan pydub",
        "apt-get update && apt-get install wget -y",
        "bash SadTalker/scripts/download_models.sh",
        "mv gfpgan SadTalker/",
        "mv checkpoints SadTalker/"
        ])
    )

stub.image = image

@stub.cls(gpu=model_schema["gpu"], memory=model_schema["memory"], container_idle_timeout=model_schema["container_idle_timeout"], timeout=9000)
class stableDiffusion:    
    
    @method()
    def run_inference(self,file_url, audio_url):
        import os, sys

        os.chdir("../SadTalker")
        sys.path.insert(0, '/SadTalker')

        from time import strftime
        from time import time
        from src.utils.preprocess import CropAndExtract
        from src.test_audio2coeff import Audio2Coeff  
        from src.facerender.animate import AnimateFromCoeff
        from src.generate_batch import get_data
        from src.generate_facerender_batch import get_facerender_data
        from src.utils.init_path import init_path
        import requests, tempfile, shutil, base64
        from pydub import AudioSegment

        

        facerender_batch_size = 10
        sadtalker_paths = init_path("./checkpoints", 'src/config', "256", False, "full")

        preprocess_model = CropAndExtract(sadtalker_paths, "cuda")
        audio_to_coeff = Audio2Coeff(sadtalker_paths, "cuda")
        animate_from_coeff = AnimateFromCoeff(sadtalker_paths, "cuda")

        save_dir = os.path.join("results", strftime("%Y_%m_%d_%H.%M.%S"))

        temp_file_audio= tempfile.NamedTemporaryFile(delete=True, suffix='.mp3')

        response = requests.get(audio_url)

        if response.status_code == 200:
            with open(temp_file_audio.name, 'wb') as f:
                f.write(response.content)
        temp_file_img = tempfile.NamedTemporaryFile(delete=True, suffix='.jpg')
        file_url.save(temp_file_img, format='JPEG')

        def get_duration_pydub(file_path):
            audio_file = AudioSegment.from_file(file_path)
            duration = audio_file.duration_seconds
            return duration
        if(get_duration_pydub(temp_file_audio.name)>60):
            raise Exception(str("Audio Length should be less than 60s"))

        audio_path=temp_file_audio.name
        pic_path=temp_file_img.name

        first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
        os.makedirs(first_frame_dir, exist_ok=True)
        first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(pic_path, first_frame_dir, "full", source_image_flag=True)
        ref_eyeblink_coeff_path=None
        ref_pose_coeff_path=None
        batch = get_data(first_coeff_path, audio_path, "cuda", ref_eyeblink_coeff_path, still=True)
        coeff_path = audio_to_coeff.generate(batch, save_dir, 0, ref_pose_coeff_path)
        data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, 
                                    facerender_batch_size, None, None, None,
                                    expression_scale=1, still_mode=True, preprocess="full")
        video_path = animate_from_coeff.generate(data, save_dir, pic_path, crop_info, \
                                    enhancer=None, background_enhancer=None, preprocess="full")
            
        with open(video_path, "rb") as f:
            url="https://qolaba-server-development-2303.up.railway.app/api/v1/uploadToCloudinary/audiovideo"
            byte=f.read()
            myobj = {"image":"data:audio/mpeg;base64,"+(base64.b64encode(byte).decode("utf8"))}
                
            rps = requests.post(url, json=myobj, headers={'Content-Type': 'application/json'})
            video_url=rps.json()["data"]["secure_url"]
            

        shutil.rmtree("results")
        try:
            temp_file_audio.close()
            temp_file_img.close()
        except:
            pass
        return {"video_url":video_url}