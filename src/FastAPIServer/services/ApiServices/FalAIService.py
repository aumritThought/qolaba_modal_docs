from src.data_models.ModalAppSchemas import FluxText2ImageParameters, IdeoGramText2ImageParameters, FluxImage2ImageParameters, RecraftV3Text2ImageParameters, SDXLText2ImageParameters
from src.utils.Globals import timing_decorator, prepare_response, make_request
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import OUTPUT_IMAGE_EXTENSION, IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG
import concurrent.futures 
import base64, io, fal_client
from src.utils.Globals import timing_decorator, make_request, prepare_response, get_image_from_url, upload_data_gcp, invert_bw_image_color, simple_boundary_blur
from PIL.Image import Image as Imagetype
from transparent_background import Remover

class FalAIFluxProText2Image(IService):
    def __init__(self) -> None:
        super().__init__()

    def make_api_request(self, parameters : FluxText2ImageParameters) -> str:
        input = {
            "prompt": parameters.prompt,
            "image_size": {
                    "width": parameters.width,
                    "height": parameters.height
            },
            "safety_tolerance" : "2",
            "sync_mode": True,
            "enable_safety_checker": True,
        }
        result = fal_client.subscribe(
            "fal-ai/flux-pro/v1.1",
            arguments=input,
            with_logs=False,
        )  
        if(sum(result["has_nsfw_concepts"])==1):
            raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)

        header, encoded = str(result["images"][0]["url"]).split(",", 1)
    
        data = base64.b64decode(encoded)
        
        return data

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : FluxText2ImageParameters = FluxText2ImageParameters(**parameters)

        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)
            
            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)
    

class FalAIFluxDevText2Image(IService):
    def __init__(self) -> None:
        super().__init__()

    def make_api_request(self, parameters : FluxText2ImageParameters) -> str:
        input = {
            "prompt": parameters.prompt,
            "image_size": {
                    "width": parameters.width,
                    "height": parameters.height
            },
            "safety_tolerance" : "2",
            "sync_mode": True,
            "enable_safety_checker": True,
        }
        result = fal_client.subscribe(
            "fal-ai/flux/dev",
            arguments=input,
            with_logs=False,
        )  
        if(sum(result["has_nsfw_concepts"])==1):
            raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)

        header, encoded = str(result["images"][0]["url"]).split(",", 1)
    
        data = base64.b64decode(encoded)
        
        return data

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : FluxText2ImageParameters = FluxText2ImageParameters(**parameters)

        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)
            
            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)
    
class FalAIFluxschnellText2Image(IService):
    def __init__(self) -> None:
        super().__init__()

    def make_api_request(self, parameters : FluxText2ImageParameters) -> str:
        input = {
            "prompt": parameters.prompt,
            "image_size": {
                    "width": parameters.width,
                    "height": parameters.height
            },
            "num_inference_steps": 12,
            "num_images": 1,
            "enable_safety_checker": True
        }
        result = fal_client.subscribe(
            "fal-ai/flux/schnell",
            arguments=input,
            with_logs=False,
        )  
        if(sum(result["has_nsfw_concepts"])==1):
            raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)

        response = make_request(result["images"][0]["url"], "GET")

        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : FluxText2ImageParameters = FluxText2ImageParameters(**parameters)

        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)
            
            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)

class FalAIFluxDevImage2Image(IService):
    def __init__(self) -> None:
        super().__init__()

    def make_api_request(self, parameters : FluxImage2ImageParameters) -> str:
        input = {
            "image_url" : parameters.file_url,
            "prompt": parameters.prompt,
            "safety_tolerance" : "2",
            "sync_mode": True,
            "enable_safety_checker": True,
            "strength": parameters.strength,
        }
        result = fal_client.subscribe(
            "fal-ai/flux/dev/image-to-image",
            arguments=input,
            with_logs=False,
        )  
        if(sum(result["has_nsfw_concepts"])==1):
            raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)

        header, encoded = str(result["images"][0]["url"]).split(",", 1)
    
        data = base64.b64decode(encoded)
        
        return data

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : FluxImage2ImageParameters = FluxImage2ImageParameters(**parameters)

        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)
            
            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)
    

class FalAIRefactorV3Text2Image(IService):
    def __init__(self) -> None:
        super().__init__()

    def make_api_request(self, parameters : RecraftV3Text2ImageParameters) -> str:
        input = {
            "prompt": parameters.prompt,
            "image_size": {
                    "width": parameters.width,
                    "height": parameters.height
            },
            "style" : parameters.style
        }
        result = fal_client.subscribe(
            "fal-ai/recraft-v3",
            arguments=input,
            with_logs=False,
        )

        response = make_request(result["images"][0]["url"], "GET")

        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : RecraftV3Text2ImageParameters = RecraftV3Text2ImageParameters(**parameters)

        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)
            
            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)
    
class FalAISD35LargeText2Image(IService):
    def __init__(self) -> None:
        super().__init__()

    def make_api_request(self, parameters : SDXLText2ImageParameters) -> str:
        input = {
            "prompt": parameters.prompt,
            "negative_prompt": parameters.negative_prompt,
            "image_size": {
                "width": parameters.width,
                "height": parameters.height
            },
            "num_inference_steps": parameters.num_inference_steps,
            "guidance_scale": parameters.guidance_scale,
            "num_images": 1,
            "enable_safety_checker": True,
            "output_format": "jpeg",
            "sync_mode": True
        }
        result = fal_client.subscribe(
            "fal-ai/stable-diffusion-v35-large",
            arguments=input,
            with_logs=False,
        )  
        if(sum(result["has_nsfw_concepts"])==1):
            raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)

        header, encoded = str(result["images"][0]["url"]).split(",", 1)
    
        data = base64.b64decode(encoded)
        
        return data

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : SDXLText2ImageParameters = SDXLText2ImageParameters(**parameters)

        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)
            
            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)

class FalAISD35LargeTurboText2Image(IService):
    def __init__(self) -> None:
        super().__init__()

    def make_api_request(self, parameters : SDXLText2ImageParameters) -> str:
        input = {
            "prompt": parameters.prompt,
            "negative_prompt": parameters.negative_prompt,
            "image_size": {
                "width": parameters.width,
                "height": parameters.height
            },
            "num_inference_steps": 10,
            "guidance_scale": 3,
            "num_images": 1,
            "enable_safety_checker": True,
            "output_format": "jpeg",
            "sync_mode": True
        }
        result = fal_client.subscribe(
            "fal-ai/stable-diffusion-v35-large/turbo",
            arguments=input,
            with_logs=False,
        )  
        if(sum(result["has_nsfw_concepts"])==1):
            raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)

        header, encoded = str(result["images"][0]["url"]).split(",", 1)
    
        data = base64.b64decode(encoded)
        
        return data

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : SDXLText2ImageParameters = SDXLText2ImageParameters(**parameters)

        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)
            
            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)
    
class FalAISD35MediumText2Image(IService):
    def __init__(self) -> None:
        super().__init__()

    def make_api_request(self, parameters : SDXLText2ImageParameters) -> str:
        input = {
            "prompt": parameters.prompt,
            "negative_prompt": parameters.negative_prompt,
            "image_size": {
                "width": parameters.width,
                "height": parameters.height
            },
            "num_inference_steps": parameters.num_inference_steps,
            "guidance_scale": parameters.guidance_scale,
            "num_images": 1,
            "enable_safety_checker": True,
            "output_format": "jpeg",
            "sync_mode": True
        }
        result = fal_client.subscribe(
            "fal-ai/stable-diffusion-v35-medium",
            arguments=input,
            with_logs=False,
        )  
        if(sum(result["has_nsfw_concepts"])==1):
            raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)

        header, encoded = str(result["images"][0]["url"]).split(",", 1)
    
        data = base64.b64decode(encoded)
        
        return data

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : SDXLText2ImageParameters = SDXLText2ImageParameters(**parameters)

        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)
            
            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)
    

class FalAIFlux3Inpainting(IService):
    def __init__(self) -> None:
        super().__init__()

    def generate_image(self, parameters : IdeoGramText2ImageParameters) -> Imagetype:
        input = {
            "prompt": parameters.prompt,
            "num_images": 1,
            "safety_tolerance": "2",
            "output_format": "jpeg",
            "image_url": parameters.file_url,
            "mask_url": parameters.mask_url
        }
        result = fal_client.subscribe(
            "fal-ai/flux-pro/v1/fill",
            arguments=input,
            with_logs=False,
        )  
        if(sum(result["has_nsfw_concepts"])==1):
            raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)

        response = make_request(result["images"][0]["url"], "GET")

        return response.content
        
    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : IdeoGramText2ImageParameters = IdeoGramText2ImageParameters(**parameters)
        original_image = get_image_from_url(
            parameters.file_url, rs_image=False
        )

        parameters.mask_url = get_image_from_url(parameters.mask_url)

        parameters.mask_url = invert_bw_image_color(parameters.mask_url)

        parameters.mask_url = simple_boundary_blur(parameters.mask_url)

        parameters.mask_url = parameters.mask_url.resize((original_image.size))

        parameters.mask_url = upload_data_gcp(parameters.mask_url, "webp")

        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.generate_image, parameters)
                futures.append(future)
            
            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)
    

class FalAIFlux3ReplaceBackground(IService):
    def __init__(self, remover : Remover) -> None:
        super().__init__()
        self.remover = remover        

    def generate_image(self, parameters : IdeoGramText2ImageParameters) -> Imagetype:
        input = {
            "prompt": parameters.prompt,
            "num_images": 1,
            "safety_tolerance": "2",
            "output_format": "jpeg",
            "image_url": parameters.file_url,
            "mask_url": parameters.mask_url
        }
        result = fal_client.subscribe(
            "fal-ai/flux-pro/v1/fill",
            arguments=input,
            with_logs=False,
        )  
        if(sum(result["has_nsfw_concepts"])==1):
            raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)

        response = make_request(result["images"][0]["url"], "GET")

        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : IdeoGramText2ImageParameters = IdeoGramText2ImageParameters(**parameters)
        original_image = get_image_from_url(
            parameters.file_url, rs_image=False
        )
        parameters.mask_url = self.remover.process(original_image, type="rgba")

        parameters.mask_url = parameters.mask_url.getchannel('A')

        parameters.mask_url = invert_bw_image_color(parameters.mask_url)

        parameters.mask_url = simple_boundary_blur(parameters.mask_url)

        parameters.mask_url = parameters.mask_url.resize((original_image.size))

        parameters.mask_url = upload_data_gcp(parameters.mask_url, "webp")

        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.generate_image, parameters)
                futures.append(future)
            
            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)
