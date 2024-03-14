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


# from modal import Cls
# import time

# init_parameters = {
#     "model" : "Colorful",
#     "controlnet_model" : "canny"
# }
# extra_negative_prompt = "disfigured, kitsch, ugly, oversaturated, greain, low-res, Deformed, blurry, bad anatomy, poorly drawn face, mutation, mutated, extra limb, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, disgusting, poorly drawn, childish, mutilated, mangled, old, surreal, calligraphy, sign, writing, watermark, text, body out of frame, extra legs, extra arms, extra feet, out of frame, poorly drawn feet, cross-eye"

# parameters = {
#     "height": 1024,
#     "width": 1024,
#     "num_inference_steps": 20, 
#     "guidance_scale":  8,
#     "batch":  1,
#     "prompt": "black plain background,  VECTOR CARTOON ILLUSTRATION, half-body shot portrait male,  5 o clock shadow, 3d bitmoji avatar render, pixar, high def textures 8k, highly detailed, 3d render, award winning, no background elements",
#     "negative_prompt": extra_negative_prompt,
#     "lora_scale" : 0.5,
#     "file_url" : "https://res.cloudinary.com/qolaba/image/upload/v1710263511/ldioa1ncn4pn8r1gtfho.png",
#     "controlnet_scale" : 0.8,
#     "bg_img" : "https://res.cloudinary.com/qolaba/image/upload/v1705612740/jbsewrunbccplqd19y4p.png",
#     "controlnet_scale" : 1.5,
#     "strength" : 0.5,
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
# Model = Cls.lookup("IPAdapter_face_consistent", "stableDiffusion", environment_name = "dev")  # returns a class-like object
# print(time.time() - st)
# m = Model(init_parameters)
# print("model obtained")


# print(time.time()-st)
# print(m.run_inference.remote(parameters))
# print(time.time() - st)
print("-----BEGIN PRIVATE KEY-----\\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQCvxrJNlnt2jnka\\nWUNHWxzOhkdbKndX5V1r8MM9ZtVLWT1OJ3oBqv9MYi1MQN46QDqvNZ4a/w0hQQk/\\nn3L1SV/kZD/imRYI5y/8b3lFXityB8RckIfSMnW6BKZCbiQONGyBnQRxsTfSoFz2\\nh64zMBcVJJBjHfrNwNqfcgKpBQLQHak8HiDOJtxLiP5D597bS7FgXUMe3r4vH/0e\\nAibUw65cf2jO+sBxQRkZ+B8r38Nt0FB87QHbn2hdO9e3Q92++MsEKKBqeY4jGq3E\\nb0G4NP5k21wreAJOJ/Y8WhcVX/iKzFxqHh275//AqBXPReVt2U/NHd3ENdPJZV3P\\nXfl22B+xAgMBAAECggEABlScq71De3gjMysGEzyxhMbIAjKSi1fjaug9cIQ4DWoc\\nkmh1pSKmZhbj7KrqPf+TGtKOkUhrLa3g5677dfX4q29zWrrlSjiLopbVWlArYSSC\\nYqGMwgnxx2YUYqXYFqq2F4nDGlZLOALGb6q1tW+nfdTRaOpwXmbne3DWGPDT4pDG\\n0BeN61ttCsQuLKGletlzNabNS3QJlcy3fsIQiwqifECZF9MsmT2a31aruVDcX85B\\nuZPAleTGKmPRh1182t9P6QyzTpAeR1T/0jZd/qMWX2SdTidpNNVJa0j/57XFgMxR\\n33LlyqiAcZd4xQS738w7jHg3dl0SGCCKajz/vwX0SQKBgQDzqxOlFLV6ziok8mDw\\n41pKihoCI8Kt9YXSqM2NwL3H7OWcueXs4bB7eQtvbaQeZvbxvpiPIJdd0GOQa7JE\\nAhXzvpsopRt7T2kxqc189e4nPwl41JRz8dfjbWxY0G/oBxd1fiSqYq5u/5NaSzui\\nAfiNauZjAt1IGsgwJ5MazQ/baQKBgQC4rAWBo/5PV6EGk4N0rYFrDlFcDzmK5hFj\\nDgCwcKlGoXKdQhF6GBYegls+CVw664xBYc95ptmq6jILH+AMXo50Xr6HzJaYL3Gj\\nhlhWA9rt9n3U684i+JsnzSJjuN48VZNcJfCvMMFOnnIjGg4ZJ32bNjYhkaIIW/4b\\nPttH3j4BCQKBgBwD7GNLiT4QXBoZX/nyOdxeGnVqhSSZGQTKca+9nFRTMWcenIfq\\nvu7DUQRDt93i+rt6rXGvTpfzsK7XIwzcrId0v8Qhj5JS5AZYvo6CfBo5Di69SkA2\\naxrz5sQjWupzfkf889w/Mk0Cx39XLQkIbvpbcxepKaXzplabBjfLoeoZAoGACcCj\\n5c3IT15cVaGSrqW/EO1HpKn1Kv2ta2LA7JB2kBFUoTNtAtqkyGWv9d2+rinkVUua\\nDl2eXyVRET9UsLKJqWGbhEZsqYrP4IfmwFwhwrFiwczWLQieAQMXTtbjfaIzTwl9\\n5XKYstMxSeNFXVS1kG3TfABZ40EgcUXnhKsa8SkCgYBNE9CaOaB8+Bds80Xc1n/w\\nKi5iUE3YKbLUJPy/DAjfF35EBTCJSyE2/+K+zbWe+vZqGMt+CdMJKDmxM8WWUFTJ\\nzu1SolQdrlIRoKN5JLZ6AW8VbfZ4F9AgYVyRezUTvXEU3c9Cqh+u9BDzw4T5Y4/5\\nj7n282BjtkgK2Ut03mvfhQ==\\n-----END PRIVATE KEY-----\\n".replace("\\n", "\n"))
