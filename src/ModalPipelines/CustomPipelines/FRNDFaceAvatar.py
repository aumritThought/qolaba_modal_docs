from modal import Stub, method, Volume, Secret
from src.data_models.Configuration import stub_dictionary
from src.data_models.ModalAppSchemas import StubNames, FRNDFaceAvatarParameters
from src.utils.Globals import get_base_image, SafetyChecker, generate_image_urls, prepare_response, get_image_from_url, get_refiner
from src.utils.Constants import VOLUME_NAME, VOLUME_PATH, SECRET_NAME, sdxl_model_list, gender_word, OUTPUT_IMAGE_EXTENSION
import torch, time, os, sys
from diffusers import StableDiffusionXLPipeline
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import numpy as np
from transparent_background import Remover
import random



stub_name = StubNames().frnd_face_consistent

stub = Stub(stub_name)

vol = Volume.persisted(VOLUME_NAME)

def download_face_model():
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    Remover()

image = get_base_image().run_commands(
    "git clone https://github.com/tencent-ailab/IP-Adapter.git",
    "wget https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl.bin"
).run_function(download_face_model, secrets= [Secret.from_name(SECRET_NAME)])

stub.image = image



@stub.cls(gpu = stub_dictionary[stub_name].gpu, 
          container_idle_timeout = stub_dictionary[stub_name].container_idle_timeout,
          memory = stub_dictionary[stub_name].memory,
          volumes = {VOLUME_PATH: vol},
          secrets = [Secret.from_name(SECRET_NAME)],
          concurrency_limit=5)
class stableDiffusion:
    def __init__(self, init_parameters : dict) -> None:
        st = time.time()
        os.chdir("../IP-Adapter")
        sys.path.insert(0, "../IP-Adapter")

        from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDXL #type: ignore

        
        pipe = StableDiffusionXLPipeline.from_single_file(
            sdxl_model_list.get("Colorful"), torch_dtype=torch.float16, use_safetensors=True, variant="fp16", load_safety_checker = False
        )
        pipe.to("cuda")

        self.refiner = get_refiner(pipe)

        # pipe.enable_xformers_memory_efficient_attention()
        # self.refiner.enable_xformers_memory_efficient_attention()

        self.app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        ip_ckpt = "../ip-adapter-faceid_sdxl.bin"
        device = "cuda"

        self.ip_model = IPAdapterFaceIDXL(pipe, ip_ckpt, device)

        self.remover = Remover()
        
        self.safety_checker = SafetyChecker()
        self.container_execution_time = time.time() - st
    
    def hex_to_rgb(self, hex_color : str):
        if hex_color.startswith('#'):
            hex_color = hex_color[1:]
        
        if len(hex_color) == 6: 
            r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        elif len(hex_color) == 3: 
            r, g, b = int(hex_color[0]*2, 16), int(hex_color[1]*2, 16), int(hex_color[2]*2, 16)
        else:
            raise ValueError("Invalid hex color format")
        
        return (r, g, b)

    @method()
    def run_inference(self, parameters : dict) -> dict:
        st = time.time()

        parameters : FRNDFaceAvatarParameters = FRNDFaceAvatarParameters(**parameters)

        extra_negative_prompt="NSFW, nudity, no clothes, pornographic content, vagaina, nude breast, disfigured, kitsch, ugly, oversaturated, greain, low-res, Deformed, blurry, bad anatomy, poorly drawn face, mutation, mutated, extra limb, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, disgusting, poorly drawn, childish, mutilated, mangled, old, surreal, calligraphy, sign, writing, watermark, text, body out of frame, extra legs, extra arms, extra feet, out of frame, poorly drawn feet, cross-eye"

        parameters.negative_prompt = parameters.negative_prompt + extra_negative_prompt

        parameters.prompt = parameters.prompt.replace(gender_word, parameters.gender)

        parameters.file_url = get_image_from_url(parameters.file_url)

        face_img=np.array(parameters.file_url)

        faces = self.app.get(face_img)

        if(len(faces) == 0):
            raise Exception("Please provide proper image, Not able to detect the faces.")

        faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        face_image = face_align.norm_crop(face_img, landmark=faces[0].kps, image_size=224)

        images = []

        seed = random.randint(1, 10000000)

        for i in range(0, parameters.batch):
            image = self.ip_model.generate(
                prompt = parameters.prompt,
                negative_prompt = parameters.negative_prompt,
                face_image=face_image,
                faceid_embeds=faceid_embeds,
                width=parameters.width, 
                height=parameters.height,
                num_inference_steps = parameters.num_inference_steps,
                denoising_end = 0.8,
                guidance_scale = parameters.guidance_scale,
                output_type="latent",
                s_scale = 0.6,
                scale = 0.6,
                num_samples = 1, 
                seed = seed
            )
            torch.cuda.empty_cache()

            image = self.refiner(
                prompt = parameters.prompt,
                num_inference_steps = parameters.num_inference_steps,
                guidance_scale = parameters.guidance_scale,
                denoising_start=0.8,
                image=image[0],
            ).images[0]

            if (parameters.remove_background == True):
                if(parameters.bg_color):
                    r,g,b = self.hex_to_rgb(parameters.bg_color)
                    image = self.remover.process(image, type=str([r, g, b]))
                else:
                    image = self.remover.process(image, type="rgba")

            torch.cuda.empty_cache()

            images.append(image)


        images, has_nsfw_content = generate_image_urls(images, self.safety_checker, check_NSFW = False, quality = 60)
        
        self.runtime = time.time() - st

        return prepare_response(images, has_nsfw_content, self.container_execution_time, self.runtime, OUTPUT_IMAGE_EXTENSION)