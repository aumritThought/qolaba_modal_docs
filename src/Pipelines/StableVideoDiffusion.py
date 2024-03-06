from modal import Stub, method, Volume, Secret
from src.data_models.Configuration import stub_dictionary
from src.data_models.ModalAppSchemas import StubNames, StableVideoDiffusion, InitParameters
from src.utils.Globals import get_base_image, prepare_response, get_image_from_url, upload_cloudinary_video, SafetyChecker
from src.utils.Constants import VOLUME_NAME, VOLUME_PATH, SECRET_NAME
from diffusers import StableVideoDiffusionPipeline
import torch, time, secrets, string, os
from diffusers.utils import export_to_video


stub_name = StubNames().stable_video_diffusion

stub = Stub(stub_name)

def download_base_sdxl():
    pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")

vol = Volume.persisted(VOLUME_NAME)

image = get_base_image().run_function(download_base_sdxl)

stub.image = image


@stub.cls(gpu = stub_dictionary[stub_name].gpu, 
          container_idle_timeout = stub_dictionary[stub_name].container_idle_timeout,
          memory = stub_dictionary[stub_name].memory,
          volumes = {VOLUME_PATH: vol},
          secrets = [Secret.from_name(SECRET_NAME)])
class stableDiffusion:
    def __init__(self, init_parameters : dict) -> None:
        st = time.time()
        init_parameters : InitParameters = InitParameters(**init_parameters)

        self.pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")

        self.pipe.to("cuda")
        self.safety_checker = SafetyChecker()

        self.container_execution_time = time.time() - st

    @method()
    def run_inference(self, parameters : dict) -> dict:
        st = time.time()
        parameters : StableVideoDiffusion = StableVideoDiffusion(**parameters)
        parameters.image = get_image_from_url(parameters.image, resize = True)
        has_nsfw_concept = self.safety_checker.check_nsfw_content(parameters.image)
        
        if(has_nsfw_concept[0] == True):
            raise Exception("Provided image contains NSFW content")

        frames = self.pipe(parameters.image, decode_chunk_size=8).frames[0]
        torch.cuda.empty_cache()

        random_string = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(10))

        vid_path = export_to_video(frames, f"{random_string}.avi", fps=parameters.fps)



        image_urls = upload_cloudinary_video(vid_path)

        try:
            os.remove(vid_path)
        except:
            pass

        has_nsfw_content = [False]
        self.runtime = time.time() - st

        return prepare_response([image_urls], has_nsfw_content, self.container_execution_time, self.runtime)