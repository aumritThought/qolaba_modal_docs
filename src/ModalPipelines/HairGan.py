from modal import App, method, Volume, Secret
from src.data_models.Configuration import stub_dictionary
from src.data_models.ModalAppSchemas import StubNames, HairFastParameters
from src.utils.Globals import SafetyChecker, generate_image_urls, prepare_response, get_image_from_url, get_base_image
from src.utils.Constants import VOLUME_NAME, VOLUME_PATH, OUTPUT_IMAGE_EXTENSION, SECRET_NAME
import time, os, sys
from torchvision import transforms
from torch import Tensor
from PIL import Image

stub_name = StubNames().hair_fast

app = App(stub_name)

vol = Volume.from_name(VOLUME_NAME)


image = get_base_image().run_commands(
    # "wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip",
    "apt install unzip dpkg cmake clang ninja-build -y",
    # "unzip -o ninja-linux.zip -d /usr/local/bin/",
    # "update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force",
    "git clone https://github.com/AIRI-Institute/HairFastGAN",
    "cd HairFastGAN && git clone https://huggingface.co/AIRI-Institute/HairFastGAN",
    "cd HairFastGAN/HairFastGAN && git lfs pull && cd ..",
    "mv HairFastGAN/HairFastGAN/pretrained_models HairFastGAN/pretrained_models",
    "mv HairFastGAN/HairFastGAN/input HairFastGAN/input",
    """grep -v "torch==1.13.1\|torchmetrics==0.11.4\|torchvision==0.14.1" HairFastGAN/requirements.txt > HairFastGAN/temp_requirements.txt""",
    "cd HairFastGAN && pip install -r temp_requirements.txt"
)

app.image = image

@app.cls(gpu = stub_dictionary[stub_name].gpu, 
          container_idle_timeout = stub_dictionary[stub_name].container_idle_timeout,
          memory = stub_dictionary[stub_name].memory,
          volumes = {VOLUME_PATH: vol},
          secrets = [Secret.from_name(SECRET_NAME)],
          concurrency_limit=stub_dictionary[stub_name].num_containers,
          retries=1)
class stableDiffusion:
    def __init__(self, init_parameters : dict) -> None:
        st = time.time()
        os.chdir("../HairFastGAN")
        sys.path.insert(0, "../HairFastGAN")
        from hair_swap import HairFast, get_parser
        

        model_args = get_parser()
        self.hair_fast = HairFast(model_args.parse_args([]))
        
        self.safety_checker = SafetyChecker()
        self.container_execution_time = time.time() - st

    def resize(self, img : Image.Image):
        from utils.shape_predictor import align_face
        
        img = align_face(img, return_tensors=False)[0]

        return img

    @method()
    def run_inference(self, parameters : dict) -> dict:
        st = time.time()

        parameters : HairFastParameters = HairFastParameters(**parameters)
        
        face_path = get_image_from_url(parameters.face_url)
        shape_path = get_image_from_url(parameters.shape_url) 
        color_path = get_image_from_url(parameters.color_url)

        final_image : Tensor = self.hair_fast.swap(self.resize(face_path), self.resize(shape_path), self.resize(color_path))
        tensor = final_image.cpu().detach()
    
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        
        to_pil = transforms.ToPILImage()
        
        pil_image = to_pil(tensor)
        
        images, has_nsfw_content = generate_image_urls([pil_image], self.safety_checker, check_NSFW=False)

        self.runtime = time.time() - st

        return prepare_response(images, has_nsfw_content, self.container_execution_time, self.runtime, OUTPUT_IMAGE_EXTENSION)