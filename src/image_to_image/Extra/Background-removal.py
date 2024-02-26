from modal import Image, Stub, method


def download_models():
    import cv2

    from PIL import Image
    from transparent_background import Remover

    remover = Remover()


stub = Stub("Background_removal_image2image")
image = (
    Image.debian_slim(python_version="3.11").run_commands(
        [
            "apt-get update && apt-get install ffmpeg libsm6 libxext6  -y",
            "pip install torch torchvision opencv-python timm tqdm kornia gdown pyvirtualcam transparent-background",
        ]
    )
).run_function(download_models, gpu="a10g")

stub.image = image


@stub.cls(gpu="a10g", container_idle_timeout=200, memory=10240)
class stableDiffusion:
    def __enter__(self):
        import time

        st = time.time()
        import cv2

        from PIL import Image
        from transparent_background import Remover

        # Load model
        # self.remover = Remover()
        self.container_execution_time = time.time() - st

    def generate_image_urls(self, image_data):
        import io, base64, requests

        url = "https://qolaba-server-production-caff.up.railway.app/api/v1/uploadToCloudinary/image"
        image_urls = []
        for im in range(0, len(image_data["images"])):
            filtered_image = io.BytesIO()
            if image_data["Has_NSFW_Content"][im]:
                pass
            else:
                image_data["images"][im].save(filtered_image, "JPEG")
                myobj = {
                    "image": "data:image/png;base64,"
                    + (base64.b64encode(filtered_image.getvalue()).decode("utf8"))
                }
                rps = requests.post(
                    url, json=myobj, headers={"Content-Type": "application/json"}
                )
                im_url = rps.json()["data"]["secure_url"]
                image_urls.append(im_url)
        return image_urls

    @method()
    def run_inference(
        self,
        file_url,
        bg_img=None,
        bg_color=False,
        r_color=255,
        g_color=0,
        b_color=0,
        blur=False,
    ):
        from PIL import Image
        import requests, io, tempfile, torch
        import time

        st = time.time()
        file_url = file_url.convert("RGB")
        from transparent_background import Remover

        self.remover = Remover()
        try:
            response = requests.get(bg_img)
            bg_img = Image.open(io.BytesIO(response.content))
            bg_img = bg_img.resize(file_url.size)
            bg_img = bg_img.convert("RGB")
        except:
            bg_img = None

        if blur == True:
            image = self.remover.process(file_url, type="blur")

        elif not (bg_img == None):
            temp_file_img = tempfile.NamedTemporaryFile(delete=True, suffix=".jpg")
            bg_img.save(temp_file_img, format="JPEG")
            image = self.remover.process(
                file_url, type=temp_file_img.name
            )  # use another image as a background
            try:
                temp_file_img.close()
                bg_img = None
                file_url = None
            except:
                pass
        elif bg_color == True:
            image = self.remover.process(
                file_url, type=str([r_color, g_color, b_color])
            )
        else:
            image = self.remover.process(file_url, type="white")
        torch.cuda.empty_cache()

        has_nsfw_concept = [False] * 1
        print(image)
        image_data = {"images": [image], "Has_NSFW_Content": has_nsfw_concept}
        image_urls = self.generate_image_urls(image_data)
        self.runtime = time.time() - st

        return {
            "result": image_urls,
            "Has_NSFW_Content": has_nsfw_concept,
            "time": {
                "startup_time": self.container_execution_time,
                "runtime": self.runtime,
            },
        }
