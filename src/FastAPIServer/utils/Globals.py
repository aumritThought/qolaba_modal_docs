from modal.client import _Client
from modal.environments import ensure_env
from modal_proto import api_pb2
from modal_utils.async_utils import synchronizer
from fastapi import HTTPException, status
from dotenv import load_dotenv
import io, requests, os, re
from pillow_heif import register_heif_opener, register_avif_opener
from PIL import Image, ImageOps

# from src.utils.Globals import *
from src.utils.Constants import *
from data_models.Schemas import TaskResponse
import time
from fastapi.security import HTTPAuthorizationCredentials
from cachetools import TTLCache
from typing import Union
from data_models.Schemas import (
    Text2ImageParameters,
    Image2ImageParameters,
    VideoParameters,
    Text2TextParameters,
)
import cloudinary
import cloudinary.uploader
from requests import Response

load_dotenv()
register_heif_opener()
register_avif_opener()



app_cache = TTLCache(
    maxsize=10000, ttl=MAX_TIME_MODAL_APP_CACHE
)


@synchronizer.create_blocking
async def list_apps() -> dict[str, str]:
    client = await _Client.from_env()
    env = ensure_env(os.environ["environment"])

    res: api_pb2.AppListResponse = await client.stub.AppList(
        api_pb2.AppListRequest(environment_name=env)
    )
    list_apps = {}
    for i in res.apps:
        if i.state == 3:
            list_apps[i.app_id] = i.description

    list_apps.update(api_apps)

    return list_apps


def get_modal_apps() -> dict[str, str]:
    if ( "data" not in app_cache or time.time() - app_cache["data"][1] > MAX_TIME_MODAL_APP_CACHE ):  # Update every hour
        data = list_apps()
        app_cache["data"] = (data, time.time())
    return app_cache["data"][0]


def timing_decorator(func: callable) -> callable:
    def wrapper(*args, **kwargs):

        start_time = time.time()
        result = func(*args, **kwargs)

        execution_time = time.time() - start_time

        time_data = {"runtime": execution_time, "startup_time": 0}
        result["time"] = time_data

        return result

    return wrapper


def check_token(api_key: HTTPAuthorizationCredentials):
    if api_key.credentials != os.environ["API_KEY"]:

        response = requests.post(
            url=os.environ["QOLABA_B2B_API_URL"],
            data={"keysHashedValue": api_key.credentials},
        )

        response.raise_for_status()

        data = response.json()
        if data["response"] == False:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect bearer token",
            )

def get_clean_name(name : str) -> str:
    pattern = re.compile('[^a-zA-Z0-9]')
    cleaned_string = pattern.sub('', name)
    return cleaned_string.lower()


class ParametersModification:
    def __init__(
        self,
        paramters: Union[
            Text2ImageParameters,
            Image2ImageParameters,
            Text2TextParameters,
            VideoParameters,
        ],
    ) -> None:
        self.parameters = paramters
        self.app_list = get_modal_apps()
        self.params_range = combined_params_list
        self.filtered_parameters: dict = None

    def filter_params_app(self):

        model_list = self.params_range.keys()
        params_list = {}
        model_name = self.app_list[self.parameters.app_id]
        if model_name in model_list:
            for i in self.params_range[model_name]:
                params_list[i] = self.parameters.model_dump()[i]
            self.filtered_parameters = params_list
            return self
        else:
            raise Exception("Check App_Id", "Model is not registered")

    def modify_prompt(self):
        if (
            "text2image" in self.app_list[self.parameters.app_id].lower()
            or "image2image" in self.app_list[self.parameters.app_id].lower()
        ):
            if not (self.parameters.app_id in api_apps.keys()):
                if self.parameters.style_preset != "None":
                    self.parameters.prompt = f"{style_presets[self.parameters.style_preset]['prompt_prefix']}, {self.parameters.prompt}, {style_presets[self.parameters.style_preset]['prompt_postfix']}"  #  }
                    self.parameters.negative_prompt = f"{style_presets[self.parameters.style_preset]['Negative_Prompt']}, {self.parameters.negative_prompt}"
            else:
                if not (self.parameters.style_preset in SDXL_API_Style_presets):
                    self.parameters.style_preset = SDXL_DEFAULT_PRESET

            if self.parameters.default_negative_prompt == True:
                self.parameters.negative_prompt = (
                    self.parameters.negative_prompt + ", " + extra_negative_prompt
                )

        return self

    def get_params(
        self,
    ) -> tuple[
        dict,
        Union[
            Text2ImageParameters,
            Image2ImageParameters,
            Text2TextParameters,
            VideoParameters,
        ],
    ]:
        return self.filtered_parameters, self.parameters

    def get_app_specific_input_parameters(
        self,
    ) -> tuple[
        dict,
        Union[
            Text2ImageParameters,
            Image2ImageParameters,
            Text2TextParameters,
            VideoParameters,
        ],
    ]:
        return self.modify_prompt().filter_params_app().get_params()


class BuildTaskResponse:
    # class task_Response_build:
    def __init__(
        self,
        task_id=None,
        error=None,
        parameters=None,
        output=None,
        app_parameters=None,
        status=None,
        taskresponse=None,
    ) -> None:
        self.error = error
        self.paramters = parameters
        self.app_parameters = app_parameters
        self.output = output
        self.taskresponse = taskresponse

        self.task_response = TaskResponse()
        self.task_response.task_id = task_id
        self.task_response.status = status
        self.app_list = get_modal_apps()

    def set_error(self):
        if self.error != None:
            self.task_response.error = self.error[0]
            self.task_response.error_data = self.error[1]
        return self

    def set_input(self):
        if self.paramters != None:
            if self.app_parameters == None:
                self.app_parameters = {}
            self.app_parameters["ref_id"] = self.paramters.ref_id
            self.task_response.input = self.app_parameters
            # if("app_id" in self.paramters.keys()):
            if self.paramters.app_id in self.app_list.keys():
                self.task_response.app_id = {
                    self.app_list[self.paramters.app_id]: self.paramters.app_id
                }
        return self

    def set_output(self):
        if self.output != None:
            self.task_response.time_required = self.output["time"]
            self.task_response.output = {
                "result": self.output["result"],
                "Has_NSFW_Content": self.output["Has_NSFW_Content"],
            }
            if "messageId" in self.output.keys() and "ppuId" in self.output.keys():
                self.task_response.output.update(
                    {
                        "messageId": self.output["messageId"],
                        "ppuId": self.output["ppuId"],
                    }
                )
        return self

    def set_taskresponse(self):
        if self.taskresponse != None:
            self.task_response = self.task_response.model_validate(self.taskresponse)
            self.task_response.input["ref_id"] = self.paramters.ref_id
        return self

    def prepare_response(self):
        return (
            self.set_error()
            .set_input()
            .set_output()
            .set_taskresponse()
            .task_response.model_dump()
        )


def upload_to_cloudinary(byte_data: bytes):
    cloudinary.config(
        cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
        api_key=os.environ["CLOUDINARY_API_KEY"],
        api_secret=os.environ["CLOUDINARY_API_SECRET"],
    )

    try:

        response = cloudinary.uploader.upload(
            file=byte_data,
        )
        return response["secure_url"]

    except Exception as e:
        raise Exception(f"Error uploading image to Cloudinary: {e}")




def make_request( url: str, method: str, files: dict = None, json_data: dict = None, headers: dict = None) -> Response:

    method = method.upper()

    if method not in ["GET", "POST"]:
        raise Exception(
            "Invalid request method", "Please check method input given with URL"
        )

    response = None
    if method == "GET":
        response = requests.get(url, headers=headers)
    elif method == "POST":
        response = requests.post(url, json=json_data, headers=headers)

    response.raise_for_status()

    return response




def generate_image_from_url(url: str, resize: bool = False) -> Image:
    response = make_request(url, "GET")
    img = Image.open(io.BytesIO(response.content)).convert("RGB")
    if resize == True:
        img = resize_image(img)
    return img



def resize_image(img: Image) -> Image:
    img = img.resize((64 * round(img.size[0] / 64), 64 * round(img.size[1] / 64)))
    if ( img.size[0] > MAX_HEIGHT or img.size[0] < MIN_HEIGHT or img.size[1] > MAX_HEIGHT or img.size[1] < MIN_HEIGHT):

        if img.size[1] >= img.size[0]:
            height = MAX_HEIGHT
            width = ((int(img.size[0]* MAX_HEIGHT/ img.size[1]))// 64) * 64
        else:
            width = MAX_HEIGHT
            height = ((int(img.size[1]*MAX_HEIGHT/ img.size[0]))// 64) * 64

        img = img.resize((width, height))
    return img

def invert_bw_image_color(img: Image) -> Image:
    white_background = Image.new("RGB", img.size, (255, 255, 255))

    white_background.paste(img, mask=img.split()[3])

    img = ImageOps.invert(white_background)

    return img
