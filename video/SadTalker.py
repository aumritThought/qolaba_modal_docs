from modal import Image, Stub, method

model_schema={
    "memory":10240,
    "container_idle_timeout":600,
    "name":"sad_talker_video",
    "gpu":"a10g"
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
    def run_inference(self,file_url, audio_url, preprocess,still_mode,  use_enhancer, size, pose_style, exp_scale, use_blink):
        import requests, tempfile, shutil, base64, os, sys
        from pydub import AudioSegment
        os.chdir("../SadTalker")
        sys.path.insert(0, '/SadTalker')
        from src.gradio_demo import SadTalker  
        temp_file_audio= tempfile.NamedTemporaryFile(delete=True, suffix='.mp3')
        response = requests.get(audio_url)
        if response.status_code == 200:
            with open(temp_file_audio.name, 'wb') as f:
                f.write(response.content)


        temp_file_img = tempfile.NamedTemporaryFile(delete=True, suffix='.png')

        file_url.save(temp_file_img, format='PNG')
            
        def get_duration_pydub(file_path):
            audio_file = AudioSegment.from_file(file_path)
            duration = audio_file.duration_seconds
            return duration
        if(get_duration_pydub(temp_file_audio.name)>31):
            raise Exception(str("Audio Length should be less than 10s"))
        st=SadTalker()
        video_path=st.test(source_image=temp_file_img.name, driven_audio=temp_file_audio.name, preprocess=preprocess, 
        still_mode=still_mode,  use_enhancer=use_enhancer, size=size, 
        pose_style = pose_style, exp_scale=exp_scale, use_blink=use_blink)
            

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