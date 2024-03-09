import io
from data_models.Schemas import Image2ImageParameters
from src.utils.Globals import timing_decorator, upload_to_cloudinary, make_request, generate_image_from_url, invert_bw_image_color
from src.services.ApiServices.IService import IService


class ClipdropUncropImage2image(IService):
    def __init__(self) -> None:
        super().__init__()
        self.url = self.clipdrop_uncrop_url
        self.api_key = self.clipdrop_api_key

    @timing_decorator
    def remote(self, parameters: Image2ImageParameters) -> dict:
        img = generate_image_from_url(parameters.file_url)

        img = img.resize((parameters.width, parameters.height))
        filtered_image = io.BytesIO()
        img.save(filtered_image, "JPEG")

        files = {
            "image_file": ("image.jpg", filtered_image.getvalue(), "image/jpeg"),
        }
        data = {
            "extend_left": parameters.left,
            "extend_right": parameters.right,
            "extend_up": parameters.top,
            "extend_down": parameters.bottom,
        }

        headers = {"x-api-key": self.api_key}

        response = make_request(
            self.url, "POST", json_data=data, headers=headers, files=files
        )

        image_bytes = io.BytesIO(response.content)

        image_urls = upload_to_cloudinary(image_bytes)

        return {"result": [image_urls], "Has_NSFW_Content": [False]}


class ClipdropCleanupImage2image(IService):
    def __init__(self) -> None:
        super().__init__()
        self.url = self.clipdrop_cleanup_url
        self.api_key = self.clipdrop_api_key

    @timing_decorator
    def remote(self, parameters: Image2ImageParameters) -> dict:

        img = generate_image_from_url(parameters.file_url, resize=True)

        filtered_image = io.BytesIO()

        img.save(filtered_image, "JPEG")

        mask = generate_image_from_url(parameters.mask_url, resize=True)

        mask = invert_bw_image_color(mask)

        mask = mask.resize((img.size))

        mask_filtered_image = io.BytesIO()

        mask.save(mask_filtered_image, "JPEG")

        files = {
            "image_file": ("image.jpg", filtered_image.getvalue(), "image/jpeg"),
            "mask_file": ("mask.png", mask_filtered_image.getvalue(), "image/png"),
        }

        headers = {"x-api-key": self.api_key}

        response = make_request(self.url, "POST", headers=headers, files=files)

        image_bytes = io.BytesIO(response.content)

        image_urls = upload_to_cloudinary(image_bytes)

        return {"result": [image_urls], "Has_NSFW_Content": [False]}


class ClipdropReplaceBackgroundImage2Image(IService):
    def __init__(self) -> None:
        super().__init__()
        self.url = self.clipdrop_replace_background_url
        self.api_key = self.clipdrop_api_key

    @timing_decorator
    def remote(self, parameters: Image2ImageParameters) -> dict:
        img = generate_image_from_url( parameters.file_url, resize=True)

        filtered_image = io.BytesIO()
        img.save(filtered_image, "JPEG")

        files = {
            "image_file": ("image.jpg", filtered_image.getvalue(), "image/jpeg"),
        }
        json_data = {"prompt": parameters.prompt}

        headers = {"x-api-key": self.api_key}

        response = make_request(
            self.url, "POST", files=files, json_data=json_data, headers=headers
        )

        image_bytes = io.BytesIO(response.content)

        image_urls = upload_to_cloudinary(image_bytes)

        return {"result": [image_urls], "Has_NSFW_Content": [False]}


class ClipdropRemoveTextImage2Image(IService):
    def __init__(self) -> None:
        super().__init__()
        self.url = self.clipdrop_remove_text_url
        self.api_key = self.clipdrop_api_key

    @timing_decorator
    def remote(self, parameters: Image2ImageParameters) -> dict:

        img = generate_image_from_url(
            parameters.file_url, resize=True
        )

        filtered_image = io.BytesIO()
        img.save(filtered_image, "JPEG")

        files = {"image_file": ("image.jpg", filtered_image.getvalue(), "image/jpeg")}
        headers = {"x-api-key": self.api_key}

        response = make_request(
            self.url, "POST", files=files, headers=headers
        )
        image_bytes = io.BytesIO(response.content)

        image_urls = upload_to_cloudinary(image_bytes)

        return {"result": [image_urls], "Has_NSFW_Content": [False]}
