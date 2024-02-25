from src.data_models.ImageToImage import Inference, Time
from src.utils.Strings import QOLABA_SERVER_UPLOAD_CLOUDINARY_URL
from modal import Image, Stub


def image_to_image_inference(image_urls, nsfw, time, runtime):
    inference = Inference()
    inference.result = image_urls
    inference.has_nsfw_content = nsfw
#     inference.time = Time()
    inference.startup_time = time
    inference.runtime = runtime
    return dict(inference)

import cloudinary
import cloudinary.uploader
# from PIL import Image as im
from io import BytesIO
# Configure Cloudinary with your account details
cloudinary.config(
    cloud_name="qolaba",
    api_key="587549555399337",
    api_secret="mPdHTEnSF0DFHY8f7OiRlISkkmI"
)
def upload_cloudinary_image(image):
     
     try:
        with BytesIO() as buffer:
            image.save(buffer, format="JPEG")  
            image_bytes = buffer.getvalue()

        response = cloudinary.uploader.upload(
            file=image_bytes,
        )
        return response['secure_url']
     
     except Exception as e:
        print(f"Error uploading image to Cloudinary: {e}")

# Replace 'your_cloud_name', 'your_api_key', 'your_api_secret', and 'your_upload_preset' with your Cloudinary account details
# Replace 'path/to/your/image.jpg' with the path to the PIL Image you want to upload
# image_path = "path/to/your/image.jpg"
# pil_image = Image.open(image_path)
# upload_pil_image(pil_image)


def cloudinary():
        return
    
def generate_image_urls(image_data):
    import io, base64, requests

    url = QOLABA_SERVER_UPLOAD_CLOUDINARY_URL
    image_urls = []
    for im in range(0, len(image_data["images"])):
        filtered_image = io.BytesIO()
        if image_data["Has_NSFW_Content"][im]:
            pass
        else:
        #     image_data["images"][im].save(filtered_image, "JPEG")
            im_url = upload_cloudinary_image(image_data["images"][im])
        #     myobj = {
        #         "image": "data:image/png;base64,"
        #         + (base64.b64encode(filtered_image.getvalue()).decode("utf8"))
        #     }
        #     rps = requests.post(
        #         url, json=myobj, headers={"Content-Type": "application/json"}
        #     )
        #     im_url = rps.json()["data"]["secure_url"]
            image_urls.append(im_url)
    return image_urls


def create_stub(name: str, run_commands: list, run_function) -> Stub:
    stub = Stub(name)
    image = (
        Image.debian_slim(python_version="3.11").run_commands(run_commands)
    ).run_function(run_function, gpu="a10g")

    stub.image = image
    return stub
