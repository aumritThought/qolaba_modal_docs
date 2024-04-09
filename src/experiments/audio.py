from modal import Image, Stub, method


def download_models():
    from audiocraft.models import AudioGen

    model = AudioGen.get_pretrained("facebook/audiogen-medium")
    model.set_generation_params(duration=8)


stub = Stub("AudioCraft_audiogen" + "_audio")
image = (
    Image.debian_slim(python_version="3.11").run_commands(
        [
            "apt-get update && apt-get install ffmpeg libsm6 libxext6 git -y",
            "python3 -m pip install -U git+https://github.com/facebookresearch/audiocraft",
        ]
    )
).run_function(download_models, gpu="a10g")

stub.image = image


@stub.cls(gpu="a10g", container_idle_timeout=600, memory=10240)
class stableDiffusion:
    def __enter__(self):
        import time

        st = time.time()
        from audiocraft.models import AudioGen

        self.model = AudioGen.get_pretrained("facebook/audiogen-medium")
        self.container_execution_time = time.time() - st

    @method()
    def run_inference(self, prompt, duration):
        import time, os, requests, base64
        from audiocraft.data.audio import audio_write

        st = time.time()
        self.model.set_generation_params(duration=duration)

        wav = self.model.generate([prompt])
        for idx, one_wav in enumerate(wav):
            audio_write("audio", one_wav.cpu(), self.model.sample_rate)
        audio_file = open("audio.wav", "rb")
        audio_bytes = audio_file.read()
        url = "https://qolaba-server-development-2303.up.railway.app/api/v1/uploadToCloudinary/audiovideo"
        myobj = {
            "image": "data:audio/mpeg;base64,"
            + (base64.b64encode(audio_bytes).decode("utf8"))
        }
        rps = requests.post(
            url, json=myobj, headers={"Content-Type": "application/json"}
        )
        audio_url = rps.json()["data"]["secure_url"]
        os.remove("audio.wav")
        self.runtime = time.time() - st

        return {
            "result": audio_url,
            "Has_NSFW_Content": [False] * 1,
            "time": {
                "startup_time": self.container_execution_time,
                "runtime": self.runtime,
            },
        }
