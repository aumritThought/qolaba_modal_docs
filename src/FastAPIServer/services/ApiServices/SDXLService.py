import io, base64
from src.data_models.ModalAppSchemas import SDXLAPITextToImageParameters, SDXLAPIImageToImageParameters
from src.utils.Globals import timing_decorator, make_request, upload_cloudinary_image, get_image_from_url, prepare_response
from src.FastAPIServer.services.IService import IService

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
            "text_prompts": [{"text": parameters.prompt}],
            "cfg_scale": parameters.guidance_scale,
            "clip_guidance_preset": "FAST_BLUE",
            "height": parameters.height,
            "width": parameters.width,
            "samples": parameters.batch,
            "steps": parameters.num_inference_steps,
            "style_preset": parameters.style_preset,
            "seed": parameters.seed
        }
        response = make_request(
            self.url, "POST", json_data=json_data, headers=headers
        )

        Has_NSFW_Content = [False] * parameters.batch

        data = response.json()
        image_urls = []
        for image in data["artifacts"]:
            image_urls.append(
                upload_cloudinary_image(
                    base64.b64decode(image["base64"])
                )
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
            parameters.file_url, resize=False
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
            "cfg_scale": parameters.guidance_scale,
            "samples": parameters.batch,
            "steps": parameters.num_inference_steps,
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
                upload_cloudinary_image(
                    base64.b64decode(image["base64"])
                )
            )

        return prepare_response(image_urls, Has_NSFW_Content, 0, 0)

