# from locust import HttpUser, task
# # import random
# # import urllib
# import modal

# # class HelloWorldUser(HttpUser):
# #     host="https://qolaba--stable-diffusion-dev.modal.run"
# #     @task(1)

# #     def hello_world(self):
# #         url="/"
# #         # b=urllib.parse.quote_plus(url)
# #         self.client.post(url)

# from PIL import Image
# import requests
# from io import BytesIO
# from data_models.Configuration import ImageInferenceInput
# # response = requests.get("https://res.cloudinary.com/qolaba/image/upload/v1695690455/kxug1tmiolt1dtsvv5br.jpg")
# # response.raise_for_status()  # Raise an HTTPError for bad responses
# # image_data = BytesIO(response.content)
# # image =  Image.open(image_data)
# # prompt = "cute dog"
# # negative = "blurry"
# # batch = 1
# # guidance_Scale = 7.5
# output = modal.Function.lookup("normal_controlnet_image2image", 
#                       "StableDiffusion.run_inference", 
#                       environment_name="dev")
# input = ImageInferenceInput(image_url="https://res.cloudinary.com/qolaba/image/upload/v1695690455/kxug1tmiolt1dtsvv5br.jpg", 
#                             prompt="cyberpunk",
#                             num_inference_steps=15)
# # input.image = image

# print(dict(input))
# print(output.remote(dict(input)))

# from modal import Cls

# import time

# init_parameters = {
#     "model" : "Vibrant"
# }

# parameters = {
#     "height": 512,
#     "width": 512,
#     "num_inference_steps": 30, 
#     "guidance_scale":  7.5,
#     "batch":  8,
#     "prompt": "cute dog",
#     "negative_prompt": "blurry",
#     "lora_scale" : 0.5
# }

# st = time.time()
# Model = Cls.lookup("SDXL_Text_To_Image", "stableDiffusion", environment_name = "dev")  # returns a class-like object

# m = Model(init_parameters)
# print(m.run_inference.remote(parameters))
# print(time.time() - st)
# from modal import Cls
# import time

# init_parameters = {
#     "model" : "Vibrant"
# }

# parameters = {
#     "height": 1024,
#     "width": 1024,
#     "num_inference_steps": 30, 
#     "guidance_scale":  7.5,
#     "batch":  1,
#     "prompt": "cute dog",
#     "negative_prompt": "blurry",
#     "lora_scale" : 0.5,
#     "image" : "https://res.cloudinary.com/qolaba/image/upload/v1709571878/file_fdyz5m.jpg",
#     "strength" : 1
# }

# st = time.time()
# Model = Cls.lookup("SDXL_Image_To_Image", "stableDiffusion", environment_name = "dev")  # returns a class-like object

# m = Model(init_parameters)
# print("model obtained")
# print(time.time() - st)

# print(time.time()-st)
# print(m.run_inference.remote(parameters))
# print(time.time() - st)


# from modal import Cls

# import time

# init_parameters = {
#     "model" : "Vibrant",
#     ""
# }

# parameters = {
#     "height": 512,
#     "width": 512,
#     "num_inference_steps": 30, 
#     "guidance_scale":  7.5,
#     "batch":  8,
#     "prompt": "cute dog",
#     "negative_prompt": "blurry",
#     "lora_scale" : 0.5
# }

# st = time.time()
# Model = Cls.lookup("SDXL_controlnet", "stableDiffusion", environment_name = "dev")  # returns a class-like object

# m = Model(init_parameters)
# print(m.run_inference.remote(parameters))
# print(time.time() - st)
# from modal import Cls
# import time

# init_parameters = {
#     "model" : "Vibrant",
#     "controlnet_model" : "depth"
# }

# parameters = {
#     "height": 1024,
#     "width": 1024,
#     "num_inference_steps": 30, 
#     "guidance_scale":  7.5,
#     "batch":  1,
#     "prompt": "a cute man sitting on chair",
#     "negative_prompt": "blurry",
#     "lora_scale" : 0.5,
#     "image" : "https://res.cloudinary.com/qolaba/image/upload/v1709597877/pm51vshvnskxv8a16emv.png",
#     "strength" : 1
# }

# st = time.time()
# Model = Cls.lookup("SDXL_controlnet", "stableDiffusion", environment_name = "dev")  # returns a class-like object

# m = Model(init_parameters)
# print("model obtained")
# print(time.time() - st)

# print(time.time()-st)
# print(m.run_inference.remote(parameters))
# print(time.time() - st)
# from RealESRGAN import RealESRGAN
# import torch

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model8 = RealESRGAN(device, scale=3)
# model8.load_weights('weights/RealESRGAN_x4.pth', download=True)



# from modal import Cls
# import time

# init_parameters = {
#     "model" : "rev-anim",
#     "controlnet_model" : "canny"
# }
# extra_negative_prompt = "disfigured, kitsch, ugly, oversaturated, greain, low-res, Deformed, blurry, bad anatomy, poorly drawn face, mutation, mutated, extra limb, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, disgusting, poorly drawn, childish, mutilated, mangled, old, surreal, calligraphy, sign, writing, watermark, text, body out of frame, extra legs, extra arms, extra feet, out of frame, poorly drawn feet, cross-eye"

# parameters = {
#     "file_url" : "https://storage.googleapis.com/qolaba-staging/20240331174647_501f3839-832c-4ed7-a2bc-db168b5905fe.jpg",
#     "strength" :0.9,
#     "height": 1024,
#     "width": 1024,
#     "num_inference_steps": 50,
#     "guidance_scale": 7.5,
#     "batch": 2,
#     "gender" : "female",
#     "remove_background" : False,
#     "prompt" : "small cute smiling dog face, purchasing vegetables",
#     "scale"  : 2,
#     "mask_url" : "https://storage.googleapis.com/qolaba-staging/1712165074578_image.png"
# }
# # image : str | Any
# #     bg_img : Optional[str] = None
# #     bg_color : Optional[bool] = False 
# #     r_color: int = Query(default=MIN_COLOR,ge=MIN_COLOR, le=MAX_COLOR)
# #     g_color: int = Query(default=MIN_COLOR,ge=MIN_COLOR, le=MAX_COLOR)
# #     b_color: int = Query(default=MIN_COLOR,ge=MIN_COLOR, le=MAX_COLOR)
# #     blur: Optional[bool] = False  
# #     strength : float = Query(ge = MIN_STRENGTH, le = MAX_STRENGTH)
# st = time.time()
# Model = Cls.lookup("Differential_Diffusion_inpainting", "stableDiffusion", environment_name = "dev")  # returns a class-like object
# print(time.time() - st)


# # Model = Model.with_options(
# #     gpu="a100"
# # )

# m = Model(init_parameters)

# print("model obtained")


# print(time.time()-st)
# print(m.run_inference.remote(parameters)["result"][0].save("abc.jpg"))
# print(m.run_inference.remote(parameters)["result"][1].save("abc1.jpg"))
# print(time.time() - st)

# # import requests

# # url = 'http://localhost:9000/upload_data_GCP'
# # headers = {
# #     'accept': 'application/json',
# #     'Authorization': 'Bearer 123456'
# # }
# # import time
# # st = time.time()
# # files = {
# #     'file': ( open('/home/dhruv/Modal/20240319191101_6174bf9c-fadc-464d-860b-66002132e3b6.jpg', 'rb'))
# # }
# # data = {
# #     "file_type" : "pdf"
# # }
# # print(time.time()-st)

# # response = requests.post(url, headers=headers, data=data, files=files)
# # print(time.time()-st)
# # # Ensure you close the file after the request is made

# # print(response.status_code)
# # print(response.json())




import requests
import json
import time
from itertools import product


headers = {
            'Authorization': 'Bearer 123456',
            }

api_styles = ["text-to-Image", 
              "Image-to-Image", 
              "ImageVariation", 
              "Controlnet", 
              "Face Consistent",
              "IllusionDiffusionModel"]


api_internal_parameters = {
    "text-to-Image": {
        "app_id": "ap-PZYd1Bb5QH57Rw4BF0dPA4",
        "height": 1536, 
        "width": 1536, 
        "num_inference_steps" : 40, 
        "guidance_scale":7.5,
        "batch" : 1,
        "prompt" : "cute dog",
        "negative_prompt" : "blurr",
        "inference_type" : "h100"
    }, 


    "Image-to-Image" : {"app_id": "ap-vcbg1l7bBuScbDuUOk3Shl",
                        "file_url": "https://res.cloudinary.com/qolaba/image/upload/v1698780425/gad9baejpidfpijkn14l.jpg", 
                        "strength" : 0.9, 
                        "guidance_scale": 7.5,
                        "batch" : 1,
                        "prompt" : "cute dog",
                        "negative_prompt" : "blurr",
                        "inference_type" : "h100"
                        }, 

    "ImageVariation" : {
        "app_id": "ap-kNvCpegQY2HiDHmk5X7X6F",
        "file_url": "https://res.cloudinary.com/qolaba/image/upload/v1695690455/kxug1tmiolt1dtsvv5br.jpg",
        "strength" : 1,
        "guidance_scale": 7.5,
        "batch": 1,
        "prompt": "cute dog",
        "negative_prompt": "blurr",
        "num_inference_steps": 30,
        "inference_type" : "h100"
    },

    "Controlnet": { 
        "app_id": "ap-1us0FK21Ach6eiWxo22is8",
        "file_url": "https://res.cloudinary.com/qolaba/image/upload/v1695690455/kxug1tmiolt1dtsvv5br.jpg",
        "strength" : 1,
        "guidance_scale": 7.5,
        "batch": 1,
        "prompt": "cute dog",
        "negative_prompt": "blurr",
        "num_inference_steps": 30,
        "inference_type" : "h100"
    },

    "Face Consistent": {"app_id": "ap-SwsTOsD7STxJ0obCkskUHw",
                        "file_url": "https://res.cloudinary.com/qolaba/image/upload/v1695690455/kxug1tmiolt1dtsvv5br.jpg",
                        "height" : 1024,
                        "width" : 1024,
                        "num_inference_steps": 30,
                        "guidance_scale": 7.5,
                        "batch": 1,
                        "prompt": "cute dog",
                        "negative_prompt" : " ",
                        "strength" : 1,
                        "inference_type" : "h100"
                        },

    "IllusionDiffusionModel": { 
        "app_id": "ap-lpocU0cB9szyuvta9lZ83B",
        "file_url":"https://res.cloudinary.com/qolaba/image/upload/v1695690455/kxug1tmiolt1dtsvv5br.jpg",
        "strength" : 0.5,
        "guidance_scale":  7.5,
        "batch":  1,
        "prompt": "cute dog",
        "negative_prompt": "blurr",
        "num_inference_steps" : 30,
        "inference_type" : "h100"
    }
}

# GPU types
gpu_types = ["h100"]

style_gpu_cost = {
    "h100": {
        "text-to-Image": 0.002290320, 
        "Image-to-Image": 0.002290320, 
        "ImageVariation": 0.002230350,
        "Controlnet": 0.002311700, 
        "Face Consistent": 0.002223680,
        "IllusionDiffusionModel": 0.002245000
    },
}

num_attempts = 4

api_data = {}

def make_api_request(style, parameters, gpu_type):

    parameters=json.dumps(parameters)
    # Make API request and return the response time and data
    attempts = 0
    api_endpoint = "https://modal-fastapi-sls-server-dev.up.railway.app/generate_content"
    while attempts < 10:
        try:
            response = requests.post(api_endpoint, headers=headers, data=parameters)
            response.raise_for_status()
            response_data = response.json()
            return response_data["time_required"]["runtime"], response_data
        
        except requests.exceptions.RequestException as e:
            print(e)
            print(f"API request failed for style {style}, GPU {gpu_type}: {e}")
            attempts += 1
            time.sleep(1)  # Wait for 1 second before retrying
    return None, None

def calculate_cost(style, parameters, gpu_type):
    total_time = 0
    response_data_list = []
    while(True):
        response_time, response_data = make_api_request(style, parameters, gpu_type)
        if response_time is not None:
            total_time += response_time
            response_data_list.append(response_data)
            if(len(response_data_list)!=1):
                total_time += response_time
            if len(response_data_list) == num_attempts:
                break 
        time.sleep(1)

    if total_time == 0:
        print(f"Unable to get 4 successful responses for style {style}, GPU {gpu_type}.")
        return None, None

    average_time = total_time / num_attempts
    cost = average_time * style_gpu_cost[gpu_type][style]
    return cost, response_data_list

for style in api_styles[4:5]:
    parameters_dict = api_internal_parameters.get(style, {})

    cost, response_data_list = calculate_cost(style, parameters_dict, "h100")
    if cost is not None:
        key = f"{style}_h100_{json.dumps(parameters_dict)}"
        api_data[key] = {"cost": cost, "response_data_list": response_data_list}
        with open(f"{style}_api_data.json", "w") as json_file:
            json.dump(api_data, json_file, indent=2)


print("API costs calculation and data storage completed.")




