from modal import Image, Stub, method


def download_models():
    import whisper

    model = whisper.load_model("large-v2")

stub = Stub("whisper"+"_audio")
image = (
    Image.debian_slim(python_version="3.10")
    .run_commands([
        "apt-get update && apt-get install ffmpeg libsm6 libxext6 git -y",
        "pip install ffmpeg-python pydub",
        "pip install git+https://github.com/openai/whisper.git "
                   ])
    ).run_function(
            download_models,
            gpu="t4"
        )

stub.image = image

@stub.cls(gpu="t4", container_idle_timeout=300, memory=10240)
class stableDiffusion:  
    def __enter__(self):
        import time
        st= time.time()
        import whisper
        self.model = whisper.load_model("large-v2")
        self.container_execution_time=time.time()-st



    @method()
    def run_inference(self, file_url, language):
        import numpy as np
        import ffmpeg
        import time, requests,tempfile
        from pydub import AudioSegment
        st=time.time()

        response = requests.get(file_url)
        temp_file_audio= tempfile.NamedTemporaryFile(delete=True, suffix='.mp3')

        if response.status_code == 200:
            with open(temp_file_audio.name, 'wb') as f:
                f.write(response.content)


        def get_duration_pydub(file_path):
            audio_file = AudioSegment.from_file(file_path)
            duration = audio_file.duration_seconds
            return duration
        if(get_duration_pydub(temp_file_audio.name)>600):
                raise Exception("Audio Length should be less than 600s", "Trim the audio or Provide file less than 600s")

        result = self.model.transcribe(temp_file_audio.name, language=language)
        self.runtime=time.time()-st
        return {"result":result["text"],  
                "Has_NSFW_Content":[False]*1, 
                "time": {"startup_time" : self.container_execution_time, "runtime":self.runtime}}
