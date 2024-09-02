from src.data_models.ModalAppSchemas import FluxText2ImageParameters, FluxImage2ImageParameters
from src.utils.Globals import timing_decorator, make_request, get_image_from_url, prepare_response, convert_to_aspect_ratio
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import OUTPUT_IMAGE_EXTENSION, FLUX_RATIO_LIST
import concurrent.futures 
import replicate


class FluxProText2Image(IService):
    def __init__(self) -> None:
        super().__init__()

    def make_api_request(self, parameters : FluxText2ImageParameters) -> str:
        
        input = {
            "prompt": parameters.prompt,
            "steps" : parameters.num_inference_steps,
            "guidance" : parameters.guidance_scale,
            "aspect_ratio": parameters.aspect_ratio,
            "interval": parameters.interval,
            "safety_tolerance" : parameters.safety_tolerance
        }

        response = replicate.run(
            "black-forest-labs/flux-pro",
            input=input
        )
        response = make_request(response, "GET")

        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : FluxText2ImageParameters = FluxText2ImageParameters(**parameters)

        aspect_ratio = convert_to_aspect_ratio(parameters.width, parameters.height)

        if(not (aspect_ratio in FLUX_RATIO_LIST)):
            raise Exception("Invalid Height and width dimension")
        parameters.aspect_ratio = aspect_ratio
        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)
            
            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)



class FluxDevText2Image(IService):
    def __init__(self) -> None:
        super().__init__()

    def make_api_request(self, parameters : FluxText2ImageParameters) -> str:
        
        input = {
            "prompt": parameters.prompt,
            "num_inference_steps" : parameters.num_inference_steps,
            "guidance" : parameters.guidance_scale,
            "aspect_ratio": parameters.aspect_ratio,
            "output_quality": parameters.output_quality
        }

        response = replicate.run(
            "black-forest-labs/flux-dev",
            input=input
        )
        response = make_request(response[0], "GET")

        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : FluxText2ImageParameters = FluxText2ImageParameters(**parameters)

        aspect_ratio = convert_to_aspect_ratio(parameters.width, parameters.height)

        if(not (aspect_ratio in FLUX_RATIO_LIST)):
            raise Exception("Invalid Height and width dimension")
        parameters.aspect_ratio = aspect_ratio
        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)
            
            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)



class FluxDevImage2Image(IService):
    def __init__(self) -> None:
        super().__init__()

    def make_api_request(self, parameters : FluxImage2ImageParameters) -> str:
        
        input = {
            "prompt": parameters.prompt,
            "num_inference_steps" : parameters.num_inference_steps,
            "guidance" : parameters.guidance_scale,
            # "aspect_ratio": parameters.aspect_ratio,
            "output_quality": parameters.output_quality,
            "image" : parameters.file_url,
            "prompt_strength" : parameters.strength
        }

        response = replicate.run(
            "black-forest-labs/flux-dev",
            input=input
        )
        response = make_request(response[0], "GET")

        return response.content

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        parameters : FluxImage2ImageParameters = FluxImage2ImageParameters(**parameters)

        # aspect_ratio = convert_to_aspect_ratio(parameters.width, parameters.height)

        # if(not (aspect_ratio in FLUX_RATIO_LIST)):
        #     raise Exception("Invalid Height and width dimension")
        # parameters.aspect_ratio = aspect_ratio
        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)
            
            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)
