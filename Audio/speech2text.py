from modal import Image, Stub, method


def download_models():
    import whisper

    model = whisper.load_model("large-v2")

stub = Stub("whisper"+"_audio")
image = (
    Image.debian_slim(python_version="3.10")
    .run_commands([
        "apt-get update && apt-get install ffmpeg libsm6 libxext6 git -y",
        "pip install ffmpeg-python",
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
        import whisper

        self.model = whisper.load_model("large-v2")



    @method()
    def run_inference(self, file_url, language):
        import numpy as np
        import ffmpeg


        out, _ = (ffmpeg
            .input(file_url)
            .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(capture_stdout=True)
        )
        out = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
        result = self.model.transcribe(out, language=language)
        return {"text":result["text"]}
