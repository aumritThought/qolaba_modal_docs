from modal import Stub, method, Volume, Secret
import torch.version
from src.data_models.Configuration import stub_dictionary
from src.data_models.ModalAppSchemas import StubNames, OOTDiffusionParameters
from src.utils.Globals import get_base_image, SafetyChecker, generate_image_urls, prepare_response, get_image_from_url
from src.utils.Constants import VOLUME_NAME, VOLUME_PATH, SECRET_NAME, OUTPUT_IMAGE_EXTENSION
import torch, time, os, random, sys
from PIL import Image
from diffusers import UniPCMultistepScheduler
from diffusers import AutoencoderKL
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from transformers import CLIPTextModel, CLIPTokenizer
from pathlib import Path
import onnxruntime as ort
import torch



class OOTDiffusion:

    def __init__(self, gpu_id, VAE_PATH, UNET_PATH, MODEL_PATH, VIT_PATH):
        import torch


        from ootd.pipelines_ootd.pipeline_ootd import OotdPipeline
        from ootd.pipelines_ootd.unet_garm_2d_condition import UNetGarm2DConditionModel
        from ootd.pipelines_ootd.unet_vton_2d_condition import UNetVton2DConditionModel

        self.gpu_id = 'cuda:' + str(gpu_id)

        vae = AutoencoderKL.from_pretrained(
            VAE_PATH,
            subfolder="vae",
            torch_dtype=torch.float16,
        )

        unet_garm = UNetGarm2DConditionModel.from_pretrained(
            UNET_PATH,
            subfolder="unet_garm",
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        unet_vton = UNetVton2DConditionModel.from_pretrained(
            UNET_PATH,
            subfolder="unet_vton",
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

        self.pipe = OotdPipeline.from_pretrained(
            MODEL_PATH,
            unet_garm=unet_garm,
            unet_vton=unet_vton,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.gpu_id)

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        
        self.auto_processor = AutoProcessor.from_pretrained(VIT_PATH)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(VIT_PATH).to(self.gpu_id)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            MODEL_PATH,
            subfolder="tokenizer",
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            MODEL_PATH,
            subfolder="text_encoder",
        ).to(self.gpu_id)


    def tokenize_captions(self, captions, max_length):
        inputs = self.tokenizer(
            captions, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids


    def __call__(self,
                model_type='hd',
                category='upperbody',
                image_garm=None,
                image_vton=None,
                mask=None,
                image_ori=None,
                num_samples=1,
                num_steps=20,
                image_scale=1.0,
    ):
        random.seed(time.time())
        seed = random.randint(0, 2147483647)
        generator = torch.manual_seed(seed)

        with torch.no_grad():
            prompt_image = self.auto_processor(images=image_garm, return_tensors="pt").to(self.gpu_id)
            prompt_image = self.image_encoder(prompt_image.data['pixel_values']).image_embeds
            prompt_image = prompt_image.unsqueeze(1)
            if model_type == 'hd':
                prompt_embeds = self.text_encoder(self.tokenize_captions([""], 2).to(self.gpu_id))[0]
                prompt_embeds[:, 1:] = prompt_image[:]
            elif model_type == 'dc':
                prompt_embeds = self.text_encoder(self.tokenize_captions([category], 3).to(self.gpu_id))[0]
                prompt_embeds = torch.cat([prompt_embeds, prompt_image], dim=1)
            else:
                raise ValueError("model_type must be \'hd\' or \'dc\'!")

            images = self.pipe(prompt_embeds=prompt_embeds,
                        image_garm=image_garm,
                        image_vton=image_vton, 
                        mask=mask,
                        image_ori=image_ori,
                        num_inference_steps=num_steps,
                        image_guidance_scale=image_scale,
                        num_images_per_prompt=num_samples,
                        generator=generator,
            ).images

        return images




stub_name = StubNames().oot_diffusion

stub = Stub(stub_name)

vol = Volume.persisted(VOLUME_NAME)

def download_weights():
    os.chdir("../OOTDiffusion/")
    sys.path.insert(0, "../OOTDiffusion/")

    from preprocess.openpose.run_openpose import OpenPose
    from preprocess.humanparsing.run_parsing import Parsing
    from ootd.inference_ootd_hd import OOTDiffusionHD

    OpenPose(0)
    Parsing(0)

    OOTDiffusion(0)

image = get_base_image().run_commands(
    ["git clone https://github.com/levihsu/OOTDiffusion",
    "pip uninstall xformers -y",
    "pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 xformers",
    "pip install -r OOTDiffusion/requirements.txt",
    "pip install basicsr",
    "curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash",
    "apt-get install git-lfs",
    "git lfs install",
    "git clone https://huggingface.co/levihsu/OOTDiffusion OOTDiffusion-model-data",
    "rm -r OOTDiffusion/checkpoints",
    "mv OOTDiffusion-model-data/checkpoints OOTDiffusion/",
    "git clone https://huggingface.co/openai/clip-vit-large-patch14 OOTDiffusion/checkpoints/clip-vit-large-patch14"
],
).run_function(download_weights, secrets= [Secret.from_name(SECRET_NAME)], gpu="a10g")

stub.image = image



@stub.cls(gpu = stub_dictionary[stub_name].gpu, 
          container_idle_timeout = stub_dictionary[stub_name].container_idle_timeout,
          memory = stub_dictionary[stub_name].memory,
          volumes = {VOLUME_PATH: vol},
          secrets = [Secret.from_name(SECRET_NAME)],
          concurrency_limit=stub_dictionary[stub_name].num_containers)
class stableDiffusion:
    def __init__(self, init_parameters : dict) -> None:
        st = time.time()

        os.chdir("../OOTDiffusion/")
        sys.path.insert(0, "../OOTDiffusion/")

        from preprocess.openpose.run_openpose import OpenPose
        from preprocess.humanparsing.run_parsing import Parsing
        from ootd.inference_ootd_hd import OOTDiffusionHD

        self.openpose_model = OpenPose(0)
        self.parsing_model = Parsing(0)

       
        
        self.safety_checker = SafetyChecker()
        self.container_execution_time = time.time() - st

    @method()
    def run_inference(self, parameters : dict) -> dict:
        st = time.time()

        os.chdir("../OOTDiffusion")
        sys.path.insert(0, "../OOTDiffusion")


        from run.utils_ootd import get_mask_location

        parameters : OOTDiffusionParameters = OOTDiffusionParameters(**parameters)

        parameters.file_url = get_image_from_url(parameters.file_url).resize((768, 1024))

        try:
            parameters.bg_img = get_image_from_url(parameters.bg_img).resize((768, 1024))
            parameters.bg_img = parameters.bg_img.resize(parameters.file_url.size)
        except Exception as e:
            raise Exception("Not able to get cloth image")

        category_dict = ['upperbody', 'lowerbody', 'dress']
        category_dict_utils = ['upper_body', 'lower_body', 'dresses']

        if parameters.category == 0:
            model_type = "hd"
            VIT_PATH = "openai/clip-vit-large-patch14"
            VAE_PATH = "checkpoints/ootd"
            UNET_PATH = "checkpoints/ootd/ootd_hd/checkpoint-36000"
            MODEL_PATH = "checkpoints/ootd"
        else:
            model_type = "dc"
            VIT_PATH = "openai/clip-vit-large-patch14"
            VAE_PATH = "checkpoints/ootd"
            UNET_PATH = "checkpoints/ootd/ootd_dc/checkpoint-36000"
            MODEL_PATH = "checkpoints/ootd"
        
        keypoints = self.openpose_model(parameters.file_url.resize((384, 512)))

        model_parse, _ = self.parsing_model(parameters.file_url.resize((384, 512)))

        mask, mask_gray = get_mask_location(model_type, category_dict_utils[parameters.category], model_parse, keypoints)
        mask = mask.resize((768, 1024), Image.NEAREST)
        mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
            
        masked_vton_img = Image.composite(mask_gray, parameters.file_url, mask)

        self.model = OOTDiffusion(0, VAE_PATH, UNET_PATH, MODEL_PATH, VIT_PATH)

        images = self.model(
                model_type=model_type,
                category=category_dict[parameters.category],
                image_garm=parameters.bg_img,
                image_vton=masked_vton_img,
                mask=mask,
                image_ori=parameters.file_url,
                num_samples=parameters.batch,
                num_steps=parameters.num_inference_steps,
                image_scale=parameters.scale,
            )

        for i in range(0, len(images)):
            images[i] = images[i].resize(parameters.file_url.size)
        
        images, has_nsfw_content = generate_image_urls(images, self.safety_checker)

        self.runtime = time.time() - st

        return prepare_response(images, has_nsfw_content, self.container_execution_time, self.runtime, OUTPUT_IMAGE_EXTENSION)


