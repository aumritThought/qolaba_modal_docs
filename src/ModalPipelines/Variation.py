import os
import sys
import time

import torch
from diffusers import StableDiffusionXLPipeline
from modal import App, Secret, Volume, method

from src.data_models.Configuration import stub_dictionary
from src.data_models.ModalAppSchemas import (
    InitParameters,
    StubNames,
    VariationParameters,
)
from src.utils.Constants import (
    OUTPUT_IMAGE_EXTENSION,
    SECRET_NAME,
    VOLUME_NAME,
    VOLUME_PATH,
    extra_negative_prompt,
    sdxl_model_list,
)
from src.utils.Globals import (
    SafetyChecker,
    generate_image_urls,
    get_base_image,
    get_image_from_url,
    get_refiner,
    prepare_response,
)

stub_name = StubNames().image_variation

app = App(stub_name)

vol = Volume.from_name(VOLUME_NAME)

image = get_base_image().run_commands(
    "git clone https://github.com/tencent-ailab/IP-Adapter.git",
    "git clone https://huggingface.co/h94/IP-Adapter IP-Adapter/IP-Adapter",
)

app.image = image


@app.cls(
    gpu=stub_dictionary[stub_name].gpu,
    container_idle_timeout=stub_dictionary[stub_name].container_idle_timeout,
    memory=stub_dictionary[stub_name].memory,
    volumes={VOLUME_PATH: vol},
    secrets=[Secret.from_name(SECRET_NAME)],
    concurrency_limit=stub_dictionary[stub_name].num_containers,
)
class stableDiffusion:
    def __init__(self, init_parameters: dict) -> None:
        st = time.time()
        os.chdir("../IP-Adapter")
        sys.path.insert(0, "../IP-Adapter")

        from ip_adapter import IPAdapterPlusXL

        init_parameters: InitParameters = InitParameters(**init_parameters)

        pipe = StableDiffusionXLPipeline.from_single_file(
            sdxl_model_list.get(init_parameters.model),
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        pipe.to("cuda")
        if init_parameters.lora_model:
            pipe.load_lora_weights(init_parameters.lora_model)

        self.refiner = get_refiner(pipe)

        # pipe.enable_xformers_memory_efficient_attention()
        # self.refiner.enable_xformers_memory_efficient_attention()

        device = "cuda"
        image_encoder_path = "IP-Adapter/models/image_encoder"
        ip_ckpt = "IP-Adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"

        self.ip_model = IPAdapterPlusXL(
            pipe, image_encoder_path, ip_ckpt, device, num_tokens=16
        )

        self.safety_checker = SafetyChecker()
        self.container_execution_time = time.time() - st

    @method()
    def run_inference(self, parameters: dict) -> dict:
        st = time.time()

        parameters: VariationParameters = VariationParameters(**parameters)

        parameters.file_url = get_image_from_url(parameters.file_url)

        parameters.negative_prompt = parameters.negative_prompt + extra_negative_prompt

        images = []
        if parameters.prompt == None or parameters.prompt == "":
            parameters.prompt = "award-winning, professional, highly detailed"

        for i in range(0, parameters.batch):
            image = self.ip_model.generate(
                prompt=parameters.prompt,
                negative_prompt=parameters.negative_prompt,
                num_inference_steps=parameters.num_inference_steps,
                denoising_end=0.8,
                guidance_scale=parameters.guidance_scale,
                output_type="latent",
                height=parameters.file_url.size[1],
                width=parameters.file_url.size[0],
                pil_image=parameters.file_url,
                scale=parameters.strength,
                num_samples=1,
            )
            torch.cuda.empty_cache()

            image = self.refiner(
                prompt=parameters.prompt,
                num_inference_steps=parameters.num_inference_steps,
                guidance_scale=parameters.guidance_scale,
                denoising_start=0.8,
                image=image[0],
            ).images[0]

            torch.cuda.empty_cache()
            images.append(image)

        images, has_nsfw_content = generate_image_urls(images, self.safety_checker)

        self.runtime = time.time() - st

        return prepare_response(
            images,
            has_nsfw_content,
            self.container_execution_time,
            self.runtime,
            OUTPUT_IMAGE_EXTENSION,
        )
