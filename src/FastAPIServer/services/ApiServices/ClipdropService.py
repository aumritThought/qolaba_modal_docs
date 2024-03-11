import io, time
from src.data_models.ModalAppSchemas import ClipDropCleanUpParameters, ClipDropRemoveTextParameters, ClipDropReplaceBackgroundParameters, ClipDropUncropParameters
from src.utils.Globals import timing_decorator, upload_cloudinary_image, make_request, get_image_from_url, invert_bw_image_color, prepare_response
from src.FastAPIServer.services.IService import IService
from requests import Response
from PIL.Image import Image as Imagetype

class ClipdropUncropImage2image(IService):
    def __init__(self) -> None:
        super().__init__()
        self.url = self.clipdrop_uncrop_url
        self.api_key = self.clipdrop_api_key

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : ClipDropUncropParameters = ClipDropUncropParameters(**parameters)
        img : Imagetype = get_image_from_url(parameters.image, resize=False)

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

        response : Response = make_request(
            self.url, "POST", json_data=data, headers=headers, files=files
        )

        image_urls = upload_cloudinary_image(response.content)

        return prepare_response([image_urls], [False], 0, 0)


class ClipdropCleanupImage2image(IService):
    def __init__(self) -> None:
        super().__init__()
        self.url = self.clipdrop_cleanup_url
        self.api_key = self.clipdrop_api_key

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : ClipDropCleanUpParameters = ClipDropCleanUpParameters(**parameters)

        img : Imagetype = get_image_from_url(parameters.image, resize=True)

        filtered_image = io.BytesIO()

        img.save(filtered_image, "JPEG")

        mask : Imagetype = get_image_from_url(parameters.mask_image, resize=True)

        mask = invert_bw_image_color(mask)

        mask = mask.resize((img.size))

        mask_filtered_image = io.BytesIO()

        mask.save(mask_filtered_image, "JPEG")

        files = {
            "image_file": ("image.jpg", filtered_image.getvalue(), "image/jpeg"),
            "mask_file": ("mask.png", mask_filtered_image.getvalue(), "image/png"),
        }

        headers = {"x-api-key": self.api_key}

        response : Response = make_request(self.url, "POST", headers=headers, files=files)

        image_urls = upload_cloudinary_image(response.content)

        return prepare_response([image_urls], [False], 0, 0)


class ClipdropReplaceBackgroundImage2Image(IService):
    def __init__(self) -> None:
        super().__init__()
        self.url = self.clipdrop_replace_background_url
        self.api_key = self.clipdrop_api_key

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : ClipDropReplaceBackgroundParameters = ClipDropReplaceBackgroundParameters(**parameters)

        img = get_image_from_url( parameters.image, resize=True)

        filtered_image = io.BytesIO()
        img.save(filtered_image, "JPEG")

        files = {
            "image_file": ("image.jpg", filtered_image.getvalue(), "image/jpeg"),
        }
        json_data = {"prompt": parameters.prompt}

        headers = {"x-api-key": self.api_key}

        response : Response = make_request(
            self.url, "POST", files=files, json_data=json_data, headers=headers
        )

        image_urls = upload_cloudinary_image(response.content)

        return prepare_response([image_urls], [False], 0, 0)


class ClipdropRemoveTextImage2Image(IService):
    def __init__(self) -> None:
        super().__init__()
        self.url = self.clipdrop_remove_text_url
        self.api_key = self.clipdrop_api_key

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : ClipDropRemoveTextParameters = ClipDropRemoveTextParameters(**parameters)
        img = get_image_from_url(
            parameters.image, resize=True
        )

        filtered_image = io.BytesIO()
        img.save(filtered_image, "JPEG")

        files = {"image_file": ("image.jpg", filtered_image.getvalue(), "image/jpeg")}
        headers = {"x-api-key": self.api_key}

        response : Response = make_request(
            self.url, "POST", files=files, headers=headers
        )

        image_urls = upload_cloudinary_image(response.content)

        return prepare_response([image_urls], [False], 0, 0)
