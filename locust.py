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

# from PIL import Image
# import requests
# from io import BytesIO

# response = requests.get("https://res.cloudinary.com/qolaba/image/upload/v1695690455/kxug1tmiolt1dtsvv5br.jpg")
# response.raise_for_status()  # Raise an HTTPError for bad responses
# image_data = BytesIO(response.content)
# image =  Image.open(image_data)
# # image = model8.predict(image)
# # print(image.size)

# import torch
# from PIL import Image
# import numpy as np
# from RealESRGAN import RealESRGAN

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = RealESRGAN(device, scale=4)
# model.load_weights('/home/prakhar-pc/qolaba/Modal-Deployments/4x-UltraSharp.pth')


# sr_image = model.predict(image)

# print(sr_image.size)


from modal import Cls
import time

init_parameters = {
    "model" : "Colorful",
    "controlnet_model" : "canny"
}
extra_negative_prompt = "disfigured, kitsch, ugly, oversaturated, greain, low-res, Deformed, blurry, bad anatomy, poorly drawn face, mutation, mutated, extra limb, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, disgusting, poorly drawn, childish, mutilated, mangled, old, surreal, calligraphy, sign, writing, watermark, text, body out of frame, extra legs, extra arms, extra feet, out of frame, poorly drawn feet, cross-eye"

parameters = {
    "file_url" : "https://res.cloudinary.com/qolaba/image/upload/v1710487238/aoaxohrfxzagvgexqfqf.jpg",
    "strength" : 0.5,
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "batch": 1,
    "gender" : "female",
    "remove_background" : False
}
# image : str | Any
#     bg_img : Optional[str] = None
#     bg_color : Optional[bool] = False 
#     r_color: int = Query(default=MIN_COLOR,ge=MIN_COLOR, le=MAX_COLOR)
#     g_color: int = Query(default=MIN_COLOR,ge=MIN_COLOR, le=MAX_COLOR)
#     b_color: int = Query(default=MIN_COLOR,ge=MIN_COLOR, le=MAX_COLOR)
#     blur: Optional[bool] = False  
#     strength : float = Query(ge = MIN_STRENGTH, le = MAX_STRENGTH)
st = time.time()
Model = Cls.lookup("IPAdapter_FRND_face_consistent", "stableDiffusion", environment_name = "dev")  # returns a class-like object
print(time.time() - st)
m = Model(init_parameters)
print("model obtained")


print(time.time()-st)
print(m.run_inference.remote(parameters))
print(time.time() - st)
