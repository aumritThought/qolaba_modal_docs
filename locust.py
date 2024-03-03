# from locust import HttpUser, task
# import random
# import urllib
import modal

# class HelloWorldUser(HttpUser):
#     host="https://qolaba--stable-diffusion-dev.modal.run"
#     @task(1)

#     def hello_world(self):
#         url="/"
#         # b=urllib.parse.quote_plus(url)
#         self.client.post(url)

from PIL import Image
import requests
from io import BytesIO
from data_models.Configuration import ImageInferenceInput
# response = requests.get("https://res.cloudinary.com/qolaba/image/upload/v1695690455/kxug1tmiolt1dtsvv5br.jpg")
# response.raise_for_status()  # Raise an HTTPError for bad responses
# image_data = BytesIO(response.content)
# image =  Image.open(image_data)
# prompt = "cute dog"
# negative = "blurry"
# batch = 1
# guidance_Scale = 7.5
output = modal.Function.lookup("normal_controlnet_image2image", 
                      "StableDiffusion.run_inference", 
                      environment_name="dev")
input = ImageInferenceInput(image_url="https://res.cloudinary.com/qolaba/image/upload/v1695690455/kxug1tmiolt1dtsvv5br.jpg", 
                            prompt="cyberpunk",
                            num_inference_steps=15)
# input.image = image

print(dict(input))
print(output.remote(dict(input)))