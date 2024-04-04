import io, base64
from src.data_models.ModalAppSchemas import SDXLAPITextToImageParameters, SDXLAPIImageToImageParameters, SDXLAPIInpainting
from src.utils.Globals import timing_decorator, make_request, upload_data_gcp, get_image_from_url, prepare_response, invert_bw_image_color
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import OUTPUT_IMAGE_EXTENSION, extra_negative_prompt
from PIL.Image import Image as Imagetype


class SDXLText2Image(IService):
    def __init__(self) -> None:
        super().__init__()
        self.api_host = self.stability_api
        self.api_key = self.stability_api_key
        self.engine_id = self.stability_engine_id
        self.url = f"{self.api_host}/v1/generation/{self.engine_id}/text-to-image"

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : SDXLAPITextToImageParameters = SDXLAPITextToImageParameters(**parameters)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        json_data = {
            "text_prompts": [
                                {
                                "text": parameters.prompt,
                                "weight": 1
                                },
                                {
                                "text": f"{parameters.negative_prompt}, {extra_negative_prompt}",
                                "weight": -1
                                }
                            ],
            "cfg_scale": parameters.guidance_scale,
            "clip_guidance_preset": "FAST_BLUE",
            "height": parameters.height,
            "width": parameters.width,
            "samples": parameters.batch,
            "steps": parameters.num_inference_steps,
            "style_preset": parameters.style_preset,
        }
        response = make_request(
            self.url, "POST", json=json_data, headers=headers
        )
        
        Has_NSFW_Content = [False] * parameters.batch

        data = response.json()
        image_urls = []
        for image in data["artifacts"]:
            image_urls.append(
                upload_data_gcp(base64.b64decode(image["base64"]), OUTPUT_IMAGE_EXTENSION)
            )
        return prepare_response(image_urls, Has_NSFW_Content, 0, 0)


class SDXLImage2Image(IService):
    def __init__(self) -> None:
        super().__init__()
        self.api_host = self.stability_api
        self.api_key = self.stability_api_key
        self.engine_id = self.stability_engine_id
        self.url = f"{self.api_host}/v1/generation/{self.engine_id}/image-to-image"

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : SDXLAPIImageToImageParameters = SDXLAPIImageToImageParameters(**parameters)
        image = get_image_from_url(
            parameters.file_url
        )

        image = image.resize((parameters.width, parameters.height)).convert("RGB")

        filtered_image = io.BytesIO()
        image.save(filtered_image, "JPEG")

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        files = {"init_image": filtered_image.getvalue()}

        json_data = {
            "image_strength": 1 - parameters.strength,
            "init_image_mode": "IMAGE_STRENGTH",
            "text_prompts[0][text]": parameters.prompt,
            "text_prompts[0][weight]": 1,
            "text_prompts[1][text]": f"{parameters.negative_prompt}, {extra_negative_prompt}",
            "text_prompts[1][weight]": -1,
            "cfg_scale": parameters.guidance_scale,
            "samples": parameters.batch,
            # "steps": 30,
            "style_preset": parameters.style_preset,
        }

        response = make_request(
            self.url, "POST", json_data=json_data, headers=headers, files=files
        )

        Has_NSFW_Content = [False] * parameters.batch

        data = response.json()
        image_urls = []
        for image in data["artifacts"]:
            image_urls.append(
                upload_data_gcp(base64.b64decode(image["base64"]), OUTPUT_IMAGE_EXTENSION)
            )

        return prepare_response(image_urls, Has_NSFW_Content, 0, 0)

class SDXLInpainting(IService):
    def __init__(self) -> None:
        super().__init__()
        self.api_key = self.stability_api_key
        self.url = self.stability_inpaint_url

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : SDXLAPIInpainting = SDXLAPIInpainting(**parameters)
        image = get_image_from_url(
            parameters.file_url
        )

        filtered_image = io.BytesIO()
        image.save(filtered_image, "JPEG")

        mask : Imagetype = get_image_from_url(parameters.mask_url)

        mask = invert_bw_image_color(mask)

        mask = mask.resize((image.size))

        mask_filtered_image = io.BytesIO()

        mask.save(mask_filtered_image, "JPEG")

        headers = {
            "accept": "image/*",
            "Authorization": f"Bearer {self.api_key}",
        }
        files={
                "image": filtered_image.getvalue(),
                "mask": mask_filtered_image.getvalue()
        }

        json_data = {
            "prompt": parameters.prompt,
            "output_format": "jpeg",
            "negative_prompt" : parameters.negative_prompt + extra_negative_prompt,
        }

        response = make_request(
            self.url, "POST", json_data=json_data, headers=headers, files=files
        )

        Has_NSFW_Content = [False] * parameters.batch

        image_urls = []

        image_urls.append(
                upload_data_gcp(response.content, OUTPUT_IMAGE_EXTENSION)
            )

        return prepare_response(image_urls, Has_NSFW_Content, 0, 0)

class SDXLReplaceBackground(IService):
    def __init__(self) -> None:
        super().__init__()
        self.api_key = self.stability_api_key
        self.url = self.stability_inpaint_url
        self.remover = self.background_remover

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : SDXLAPIInpainting = SDXLAPIInpainting(**parameters)
        image = get_image_from_url(
            parameters.file_url
        )

        mask = self.remover.process(image, type="rgba")

        mask = mask.getchannel('A')

        filtered_image = io.BytesIO()
        image.save(filtered_image, "JPEG")

        mask = invert_bw_image_color(mask)

        mask = mask.resize((image.size))

        mask_filtered_image = io.BytesIO()

        mask.save(mask_filtered_image, "JPEG")

        headers = {
            "accept": "image/*",
            "Authorization": f"Bearer {self.api_key}",
        }
        files={
                "image": filtered_image.getvalue(),
                "mask": mask_filtered_image.getvalue()
        }

        json_data = {
            "prompt": parameters.prompt,
            "output_format": "jpeg",
            "negative_prompt" : parameters.negative_prompt + extra_negative_prompt,
        }

        response = make_request(
            self.url, "POST", json_data=json_data, headers=headers, files=files
        )

        Has_NSFW_Content = [False] * parameters.batch

        image_urls = []

        image_urls.append(
                upload_data_gcp(response.content, OUTPUT_IMAGE_EXTENSION)
            )

        return prepare_response(image_urls, Has_NSFW_Content, 0, 0)