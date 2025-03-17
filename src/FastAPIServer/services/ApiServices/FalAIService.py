from src.data_models.ModalAppSchemas import FluxText2ImageParameters, UpscaleParameters, IdeoGramText2ImageParameters, OmnigenParameters, Kling15Video, MinimaxVideo, HunyuanVideo, FluxImage2ImageParameters, RecraftV3Text2ImageParameters, SDXLText2ImageParameters
from src.utils.Globals import timing_decorator, prepare_response, make_request
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import OUTPUT_IMAGE_EXTENSION, IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG, OUTPUT_VIDEO_EXTENSION
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
        # if(sum(result["has_nsfw_concepts"])==1):
        #     raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)

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



class FalAIKling15Video(IService):
    def __init__(self) -> None:
        super().__init__()

    def generate_video(self, parameters : Kling15Video) -> str:
        if(parameters.file_url == None):
            parameters.file_url = []
        if(len(parameters.file_url)==0):
            input = {
                "prompt": parameters.prompt,
                "duration": parameters.duration,
                "aspect_ratio": parameters.aspect_ratio
            }
            result = fal_client.subscribe(
                "fal-ai/kling-video/v1.5/pro/text-to-video",
                arguments=input,
                with_logs=False,
            ) 
        else:
            input = {
                "prompt": parameters.prompt,
                "duration": parameters.duration,
                "aspect_ratio": parameters.aspect_ratio,
                "image_url" : parameters.file_url[0].uri
            }
            result = fal_client.subscribe(
                "fal-ai/kling-video/v1.5/pro/image-to-video",
                arguments=input,
                with_logs=False,
            ) 

        response = make_request(result["video"]["url"], "GET")

        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : Kling15Video = Kling15Video(**parameters)
        

        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.generate_video, parameters)
                futures.append(future)
            
            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_VIDEO_EXTENSION)


class FalAIMiniMaxVideo(IService):
    def __init__(self) -> None:
        super().__init__()

    def generate_video(self, parameters : MinimaxVideo) -> str:
        if(parameters.file_url == None):
            parameters.file_url = []
        if(len(parameters.file_url) == 0):
            input = {
                "prompt": parameters.prompt,
                "prompt_optimizer": True
            }
            result = fal_client.subscribe(
                "fal-ai/minimax/video-01-live",
                arguments=input,
                with_logs=False,
            ) 
        else:
            input = {
                "prompt": parameters.prompt,
                "prompt_optimizer": True,
                "image_url": parameters.file_url[0].uri
            }
            result = fal_client.subscribe(
                "fal-ai/minimax/video-01-live/image-to-video",
                arguments=input,
                with_logs=False,
            ) 

        response = make_request(result["video"]["url"], "GET")

        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : MinimaxVideo = MinimaxVideo(**parameters)
        

        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.generate_video, parameters)
                futures.append(future)
            
            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_VIDEO_EXTENSION)
  

class FalAIhunyuanVideo(IService):
    def __init__(self) -> None:
        super().__init__()

    def generate_video(self, parameters : HunyuanVideo) -> str:
        input = {
            "prompt": parameters.prompt,
            "aspect_ratio": parameters.aspect_ratio,
        }
        result = fal_client.subscribe(
            "fal-ai/hunyuan-video",
            arguments=input,
            with_logs=False,
        ) 

        response = make_request(result["video"]["url"], "GET")

        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : HunyuanVideo = HunyuanVideo(**parameters)
        

        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.generate_video, parameters)
                futures.append(future)
            
            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_VIDEO_EXTENSION)
    

class FalAIFluxProRedux(IService):
    def __init__(self) -> None:
        super().__init__()

    def make_api_request(self, parameters : FluxImage2ImageParameters) -> str:
        input = {
            "image_url" : parameters.file_url,
            "num_inference_steps": parameters.num_inference_steps,
            "guidance_scale": 5,
            "num_images": 1,
            "safety_tolerance": "2",
            "output_format": "jpeg",
            "image_prompt_strength": parameters.strength,
            "prompt": parameters.prompt,
            "image_size": {
                "width": parameters.height,
                "height": parameters.width
            },
        }
        result = fal_client.subscribe(
            "fal-ai/flux-pro/v1.1/redux",
            arguments=input,
            with_logs=False,
        )  
        if(sum(result["has_nsfw_concepts"])==1):
            raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)

        response = make_request(result["images"][0]["url"], "GET")
        return response.content

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
    

class FalAIFluxProCanny(IService):
    def __init__(self) -> None:
        super().__init__()

    def make_api_request(self, parameters : FluxImage2ImageParameters) -> str:
        input = {
            "control_image_url" : parameters.file_url,
            "num_inference_steps": parameters.num_inference_steps,
            "guidance_scale": 5,
            "num_images": 1,
            "safety_tolerance": "2",
            "output_format": "jpeg",
            "prompt": parameters.prompt
        }

        result = fal_client.subscribe(
            "fal-ai/flux-pro/v1/canny",
            arguments=input,
            with_logs=False,
        )  
        if(sum(result["has_nsfw_concepts"])==1):
            raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)

        response = make_request(result["images"][0]["url"], "GET")
        return response.content

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
    

class FalAIFluxProDepth(IService):
    def __init__(self) -> None:
        super().__init__()

    def make_api_request(self, parameters : FluxImage2ImageParameters) -> str:
        input = {
            "control_image_url" : parameters.file_url,
            "num_inference_steps": parameters.num_inference_steps,
            "guidance_scale": 5,
            "num_images": 1,
            "safety_tolerance": "2",
            "output_format": "jpeg",
            "prompt": parameters.prompt
        }

        result = fal_client.subscribe(
            "fal-ai/flux-pro/v1/depth",
            arguments=input,
            with_logs=False,
        )  
        if(sum(result["has_nsfw_concepts"])==1):
            raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)

        response = make_request(result["images"][0]["url"], "GET")
        return response.content

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
    
class OmnigenV1(IService):
    def __init__(self) -> None:
        super().__init__()

    def make_api_request(self, parameters : OmnigenParameters) -> str:
        if(type(parameters.file_url) == str):
            parameters.file_url = [parameters.file_url]
        input = {
            "prompt":parameters.prompt,
            "image_size": {
                "width": parameters.width,
                "height": parameters.height
            },
            "num_inference_steps": parameters.num_inference_steps,
            "guidance_scale": 2.5,
            "img_guidance_scale": 1.6,
            "num_images": 1,
            "enable_safety_checker": True,
            "output_format": "jpeg",
            "input_image_urls": parameters.file_url
        }

        result = fal_client.subscribe(
            "fal-ai/omnigen-v1",
            arguments=input,
            with_logs=False,
        )  
        if(sum(result["has_nsfw_concepts"])==1):
            raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)

        response = make_request(result["images"][0]["url"], "GET")
        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : OmnigenParameters = OmnigenParameters(**parameters)

        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)
            
            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)
    

class FalAIFluxPulID(IService):
    def __init__(self) -> None:
        super().__init__()

    def make_api_request(self, parameters : FluxImage2ImageParameters) -> str:
        input = {
            "reference_image_url" : parameters.file_url,
            "image_size": {
                "width": parameters.width,
                "height": parameters.height
            },
            "num_inference_steps": 20,
            "guidance_scale": 5,
            "num_images": 1,
            "safety_tolerance": "2",
            "output_format": "jpeg",
            "prompt": parameters.prompt,
            "negative_prompt": parameters.negative_prompt,
            "enable_safety_checker": True,
            "id_weight": 1,
        }

        result = fal_client.subscribe(
            "fal-ai/flux-pulid",
            arguments=input,
            with_logs=False,
        )  
        if(sum(result["has_nsfw_concepts"])==1):
            raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)

        response = make_request(result["images"][0]["url"], "GET")
        return response.content

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
    
class FalAIUpscaler(IService):
    def __init__(self) -> None:
        super().__init__()

    def make_api_request(self, parameters : UpscaleParameters) -> str:
        input = {
            "model_type": "SDXL",
            "image_url": parameters.file_url,
            "scale": parameters.scale,
            "creativity": 0.5,
            "detail": 1,
            "shape_preservation": 0.25,
            "prompt_suffix": " high quality, highly detailed, high resolution, sharp",
            "negative_prompt": "blurry, low resolution, bad, ugly, low quality, pixelated, interpolated, compression artifacts, noisey, grainy",
            "seed": 42,
            "guidance_scale": 7.5,
            "num_inference_steps": 20,
            "enable_safety_checks": True,
            "additional_lora_scale": 1,
            "override_size_limits" : True
        }

        result = fal_client.subscribe(
            "fal-ai/creative-upscaler",
            arguments=input,
            with_logs=False,
        )  

        if(sum(result["has_nsfw_concepts"])==1):
            raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)

        response = make_request(result["images"][0]["url"], "GET")
        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : UpscaleParameters = UpscaleParameters(**parameters)
            
        results = self.make_api_request(parameters)

        Has_NSFW_Content = [False]

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)