from src.data_models.ModalAppSchemas import TaskResponse, TimeData, UpscaleParameters
from modal import Image as MIM
from modal import Secret
from PIL import Image
from PIL.Image import Image as Imagetype
from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
import torch, time, os, requests, re, io, datetime, uuid, math
from src.utils.Constants import BASE_IMAGE_COMMANDS, IMAGE_FETCH_ERROR, STAGING_API, IMAGE_FETCH_ERROR_MSG, IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG, PYTHON_VERSION, REQUIREMENT_FILE_PATH, MEAN_HEIGHT, SDXL_REFINER_MODEL_PATH, google_credentials_info, OUTPUT_IMAGE_EXTENSION, SECRET_NAME, content_type, MAX_UPLOAD_RETRY
from fastapi.security import HTTPAuthorizationCredentials
from requests import Response
from google.cloud import storage
from google.oauth2 import service_account
import numpy as np
from modal import Cls
from PIL import ImageFilter
from pillow_heif import register_heif_opener


register_heif_opener()

#Safety Checker Utils
class SafetyChecker:
    """
    NSFW content detection for generated images.
    
    This class provides a consistent safety checking mechanism across all Modal applications,
    centralizing the NSFW detection logic to avoid code duplication. It uses the StableDiffusion
    safety checker model to detect potentially inappropriate content in generated images.
    
    Attributes:
        safety_checker: StableDiffusionSafetyChecker model loaded from pretrained weights
        feature_extractor: CLIP image processor for preparing images for the safety model
    """
    def __init__(self) -> None:
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor()

    def check_nsfw_content(self, image : Imagetype) -> list[bool]:
        """
        Checks if an image contains NSFW content.
        
        Args:
            image (Imagetype): PIL Image to check for NSFW content
            
        Returns:
            list[bool]: List of boolean flags indicating NSFW detection results
        """
        image = image.convert("RGB")
        safety_checker_input = self.feature_extractor(np.array(image), return_tensors="pt").to("cuda")

        image, has_nsfw_concept = self.safety_checker(
            images= np.array(image), clip_input=safety_checker_input.pixel_values
        )

        return has_nsfw_concept
    
def download_safety_checker():
    """
    Pre-downloads safety checker model during Modal container initialization.
    
    This function ensures the safety checker model is downloaded during container
    setup rather than during the first request, reducing initial request latency.
    """
    SafetyChecker()

#Modal image Utils
def get_base_image() -> MIM:
    """
    Creates a standardized base Modal image for all deployments.
    
    This function provides a consistent base environment for all Modal applications
    with common dependencies, Python version, and system packages. Each Modal file
    can extend this base image with additional requirements specific to its needs.
    
    Returns:
        MIM: A configured Modal Image object with common dependencies installed
    """
    return MIM.debian_slim(python_version="3.11.8").run_commands(BASE_IMAGE_COMMANDS).pip_install_from_requirements(REQUIREMENT_FILE_PATH).run_function(download_safety_checker, gpu = "t4", secrets = [Secret.from_name(SECRET_NAME)])
    

#Modal app utils
def get_refiner(pipe) -> DiffusionPipeline:
    """
    Creates a reusable image refinement pipeline.
    
    This function builds a SDXL refinement pipeline that can be used across multiple
    Modal applications to enhance image quality. It reuses components from the input
    pipeline to optimize memory usage and avoid redundant model loading.
    
    Args:
        pipe: The base diffusion pipeline to share components with
        
    Returns:
        DiffusionPipeline: A configured refinement pipeline
    """
    return DiffusionPipeline.from_pretrained(
            SDXL_REFINER_MODEL_PATH,
            text_encoder_2 = pipe.text_encoder_2,
            vae = pipe.vae,
            torch_dtype = torch.float16,
            use_safetensors = True,
            variant = "fp16",
        ).to("cuda")


#Modal App Output relataed utils
def generate_image_urls(image_data, safety_checker : SafetyChecker, check_NSFW : bool = True, quality : int = 95) -> tuple[list[str], list[bool]]:
    """
    Processes generated images through safety checking and conversion.
    
    This function centralizes image post-processing for all Modal applications,
    applying NSFW content detection and converting images to bytes. It avoids
    duplicate safety checking code across multiple model implementations.
    
    Args:
        image_data: List of PIL Images to process
        safety_checker (SafetyChecker): The safety checker instance to use
        check_NSFW (bool, optional): Whether to enforce NSFW filtering. Defaults to True.
        quality (int, optional): JPEG quality for image conversion. Defaults to 95.
        
    Returns:
        tuple[list[str], list[bool]]: Tuple of (image_bytes_list, nsfw_detection_flags)
    """
    images = []
    has_nsfw_content = []
    for im in range(0, len(image_data)):
        nsfw_content = safety_checker.check_nsfw_content(image_data[im])
        if nsfw_content[0] and check_NSFW:
            has_nsfw_content.append(nsfw_content[0])
        else:
            # im_url = upload_data_gcp(, OUTPUT_IMAGE_EXTENSION)
            images.append(convert_image_to_bytes(image_data[im], quality))
            has_nsfw_content.append(nsfw_content[0])
    return images, has_nsfw_content

def convert_image_to_bytes(image: Image.Image, quality : int = 95) -> bytes:
    """
    Converts PIL Image objects to optimized byte arrays.
    
    This function handles format-specific conversion of images to bytes,
    using PNG for images with transparency and JPEG for RGB images.
    
    Args:
        image (Image.Image): PIL Image to convert to bytes
        quality (int, optional): JPEG quality setting. Defaults to 95.
        
    Returns:
        bytes: The image data as a byte array
    """
    # return img_bytes.getvalue()
    if image.mode == 'RGBA':
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            return buffer.getvalue()
    else:
        with io.BytesIO() as buffer:
            image.save(buffer, format="JPEG", quality = quality)
            return buffer.getvalue()

def prepare_response(result: list[str] | dict, Has_NSFW_content : list[bool], time_data : float, runtime : float, extension : str = None) -> dict:
    """
    Creates a standardized response format for Modal applications.
    
    This function constructs a consistent TaskResponse object to be returned by
    all Modal applications, including results, NSFW flags, and timing information.
    It centralizes response formatting to maintain API consistency.
    
    Args:
        result (list[str] | dict): The generated output content
        Has_NSFW_content (list[bool]): NSFW detection flags for each result
        time_data (float): Startup/initialization time in seconds
        runtime (float): Execution time in seconds
        extension (str, optional): File extension for result data. Defaults to None.
        
    Returns:
        dict: A standardized task response dictionary
    """
    task_response = TaskResponse(
        result = result,
        Has_NSFW_Content = Has_NSFW_content,
        time = TimeData(startup_time = time_data, runtime = runtime),
        extension=extension        
    )
    return task_response.model_dump()


#Completing request 
def make_request(url: str, method: str, json_data: dict = None, headers: dict = None, files : dict = None, json : dict = None) -> Response:
    """
    Performs HTTP requests with standardized error handling.
    
    This function centralizes HTTP request logic across the application, providing
    consistent error handling especially for content moderation issues. It supports
    both GET and POST methods with appropriate parameter handling.
    
    Args:
        url (str): The URL to request
        method (str): HTTP method (GET or POST)
        json_data (dict, optional): Form data for POST requests. Defaults to None.
        headers (dict, optional): HTTP headers. Defaults to None.
        files (dict, optional): Files for multipart POST requests. Defaults to None.
        json (dict, optional): JSON data for POST requests. Defaults to None.
        
    Returns:
        Response: The HTTP response object
        
    Raises:
        Exception: If the request method is invalid or the response indicates an error
    """
    method = method.upper()

    if method not in ["GET", "POST"]:
        raise Exception(
            "Invalid request method, Please check method input given with URL"
        )

    response = None
    if method == "GET":
        response = requests.get(url, headers=headers)
    elif method == "POST":
        response = requests.post(url, data=json_data, headers=headers, files = files, json = json)

    if(response.status_code != 200):
        if("content_moderation" in response.text or "content moderation" in response.text):
            raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)
        
        raise Exception(str(response.text))

    return response

#Cloudinary
def upload_data_gcp(data : Imagetype | str, extension : str, upscale : bool = False) -> str:
    """
    Uploads data to Google Cloud Storage with retry logic.
    
    This function handles uploading images and other files to GCP storage with
    automatic retry on failure. It can optionally upscale images before upload
    using the Ultrasharp_Upscaler service.
    
    Args:
        data (Imagetype | str): Image or data to upload
        extension (str): File extension for the uploaded data
        upscale (bool, optional): Whether to upscale images before upload. Defaults to False.
        
    Returns:
        str: Public URL of the uploaded file
        
    Raises:
        Exception: If upload fails after maximum retries
    """
    for i in range(0, MAX_UPLOAD_RETRY):
        url = upload_to_gcp(data, extension, upscale)
        if(url != None and url != ""):
            break
    if(url == "" or url == None):
        raise Exception("Received an empty URL")
    return url
            
def upload_to_gcp(data : Imagetype | str, extension : str, upscale : bool = False) -> str:
    """
    Performs the actual upload to Google Cloud Storage.
    
    This function handles the GCP-specific upload logic, including file naming,
    optional image upscaling, and content type setting. It's used by upload_data_gcp
    which provides the retry mechanism.
    
    Args:
        data (Imagetype | str): Image or data to upload
        extension (str): File extension for the uploaded data
        upscale (bool, optional): Whether to upscale images before upload. Defaults to False.
        
    Returns:
        str: Public URL of the uploaded file
        
    Raises:
        Exception: If an error occurs during the upload process
    """
    try:
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        random_string = str(uuid.uuid4())

        destination_blob_name = f"{current_time}_{random_string}.{extension}"            
        
        if(upscale == True and extension == OUTPUT_IMAGE_EXTENSION):
            if(not isinstance(data, Imagetype)):
                image_buffer = io.BytesIO(data)
                data = Image.open(image_buffer).convert("RGB")

            Model = Cls.lookup("Ultrasharp_Upscaler", "stableDiffusion", environment_name = os.environ["environment"])
            Model = Model({})
        
            output = TaskResponse(
                **Model.run_inference.remote(
                    UpscaleParameters(
                        file_url=data,
                        scale=4,
                        check_nsfw=False
                    ).model_dump()  
                ))
            if(output.Has_NSFW_Content[0] == True):
                raise Exception(IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG)
            else:
                data = output.result[0]

        if(isinstance(data, Imagetype)):
            if data.mode == 'RGBA':
                with io.BytesIO() as buffer:
                    data.save(buffer, format="PNG")
                    data = buffer.getvalue()
            else:
                with io.BytesIO() as buffer:
                    data.save(buffer, format="JPEG")
                    data = buffer.getvalue()

        byte_data = io.BytesIO(data)

        credentials = service_account.Credentials.from_service_account_info(google_credentials_info)

        storage_client = storage.Client(credentials=credentials, project=google_credentials_info['project_id'])

        bucket = storage_client.bucket(os.environ["BUCKET_NAME"])

        blob = bucket.blob(destination_blob_name)

        blob.content_type = content_type[extension]

        blob.content_disposition = "inline"

        blob.upload_from_file(byte_data)

        url = blob.public_url
        url = url.replace(STAGING_API, os.environ["CDN_API"])
        return url

    except Exception as e:
        raise Exception(f"Error uploading to GCP: {e}")


#Image operations
def resize_image(img: Imagetype) -> Imagetype:
    """
    Resizes images to model-compatible dimensions.
    
    This function ensures images are resized to dimensions compatible with the
    model requirements (multiples of 64) and within size limits to prevent
    GPU memory errors during processing.
    
    Args:
        img (Imagetype): The PIL Image to resize
        
    Returns:
        Imagetype: The resized PIL Image
    """
    img = img.resize((64 * round(img.size[0] / 64), 64 * round(img.size[1] / 64)))
    if(img.size[0]*img.size[1] > MEAN_HEIGHT*MEAN_HEIGHT):
        # if ( img.size[0] > MAX_HEIGHT or img.size[0] < MIN_HEIGHT or img.size[1] > MAX_HEIGHT or img.size[1] < MIN_HEIGHT):
            if img.size[1] >= img.size[0]:
                height = MEAN_HEIGHT
                width = ((int(img.size[0]* MEAN_HEIGHT/ img.size[1]))// 64) * 64
            else:
                width = MEAN_HEIGHT
                height = ((int(img.size[1]*MEAN_HEIGHT/ img.size[0]))// 64) * 64

            img = img.resize((width, height))
    return img



def get_image_from_url(url: str, rs_image : bool = True) -> Imagetype:
    """
    Retrieves and optionally resizes images from URLs.
    
    This function downloads images from external URLs and converts them to PIL Image
    objects. It can optionally resize the images to be compatible with model requirements.
    
    Args:
        url (str): URL of the image to download
        rs_image (bool, optional): Whether to resize the image. Defaults to True.
        
    Returns:
        Imagetype: The downloaded PIL Image
        
    Raises:
        Exception: If the image cannot be fetched or processed
    """
    try:
        response : Response = make_request(url, method = "GET") 
        image_data = io.BytesIO(response.content)
        

        image = Image.open(image_data).convert("RGB")
        if(rs_image == True):
            image = resize_image(image)
        return image
    except Exception as error:
        print(f"Url of image : {url}")
        print(str(error))
        raise  Exception(IMAGE_FETCH_ERROR, IMAGE_FETCH_ERROR_MSG)

def invert_bw_image_color(img: Imagetype) -> Imagetype:
    """
    Inverts black and white image colors.
    
    This function converts black pixels to white and white pixels to black,
    which is useful for mask-based image operations where inverting the mask
    changes the affected areas.
    
    Args:
        img (Imagetype): The PIL Image to invert
        
    Returns:
        Imagetype: The inverted PIL Image
    """
    mask_array = np.array(img)

    inverted_mask_array = 255 - mask_array

    inverted_mask = Image.fromarray(inverted_mask_array)
    
    return inverted_mask

def simple_boundary_blur(pil_image : Imagetype, blur_radius=25) -> Imagetype:
    """
    Applies Gaussian blur to an image.
    
    This function provides a simple way to blur images, which is useful for
    creating soft masks or transitions between edited and original images.
    
    Args:
        pil_image (Imagetype): The PIL Image to blur
        blur_radius (int, optional): Radius of the Gaussian blur. Defaults to 25.
        
    Returns:
        Imagetype: The blurred PIL Image
    """
    return pil_image.filter(ImageFilter.GaussianBlur(blur_radius))


def get_seed_generator(seed: int) -> torch.Generator:
    """
    Creates a deterministic random generator with a specific seed.
    
    This function ensures consistent results across runs by initializing
    a PyTorch generator with a specific seed value.
    
    Args:
        seed (int): The random seed to use
        
    Returns:
        torch.Generator: Initialized random generator
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def compress_image(image : Imagetype, max_size : int = 100) -> bytes | str:
    """
    Compresses images to a target file size.
    
    This function iteratively reduces image quality until the resulting file
    size is below the target maximum size. It's useful for creating thumbnails
    or preview images that load quickly.
    
    Args:
        image (Imagetype): The PIL Image to compress
        max_size (int, optional): Target maximum size in KB. Defaults to 100KB.
        
    Returns:
        bytes | str: Compressed image data
    """
    quality = 50
    while True:
        with io.BytesIO() as buffer:
            image.save(buffer, format="WebP", quality=quality, optimize=True)
            temp_output : bytes = buffer.getvalue()
            size = buffer.getbuffer().nbytes
        
        if size/1024 <= max_size:
            return temp_output
        elif quality <= 20:  
            return temp_output
        else:
            quality -= 10
    
#API app utils
def timing_decorator(func: callable) -> callable:
    """
    Decorator to measure and record function execution time.
    
    This decorator wraps API service functions to automatically track and
    include timing information in the response. It helps monitor performance
    and provide transparency about processing times to API consumers.
    
    Args:
        func (callable): The function to time
        
    Returns:
        callable: Wrapped function that includes timing information in its response
    """
    def wrapper(*args, **kwargs):

        start_time = time.time()
        result = func(*args, **kwargs)

        runtime = time.time() - start_time

        result = TaskResponse(**result)
        result.time.startup_time = 0
        result.time.runtime = runtime

        return result.model_dump()

    return wrapper


def check_token(api_key: HTTPAuthorizationCredentials):
    """
    Validates API authentication tokens.
    
    This function checks if a provided API key matches the expected value
    stored in environment variables, providing a centralized authentication
    mechanism for all API endpoints.
    
    Args:
        api_key (HTTPAuthorizationCredentials): The credentials to validate
        
    Raises:
        Exception: If the API key is invalid
    """
    if api_key.credentials != os.environ["API_KEY"]:
        raise Exception("Invalid API Key")

def get_clean_name(name : str) -> str:
    """
    Sanitizes string values for safe usage in filenames or IDs.
    
    This function removes all non-alphanumeric characters from a string
    to create a clean, safe identifier that can be used in filenames,
    URLs, or as database keys.
    
    Args:
        name (str): The string to sanitize
        
    Returns:
        str: Sanitized string containing only alphanumeric characters
    """
    pattern = re.compile('[^a-zA-Z0-9]')
    cleaned_string = pattern.sub('', name)
    return cleaned_string.lower()


def convert_to_aspect_ratio(width : int, height : int) -> str:
    """
    Converts width and height to a simplified aspect ratio.
    
    This function calculates the greatest common divisor of width and height
    to express the aspect ratio in its simplest form (e.g., 16:9 instead of 1920:1080).
    
    Args:
        width (int): Image width
        height (int): Image height
        
    Returns:
        str: Simplified aspect ratio as a string (e.g., "16:9")
    """
    gcd = math.gcd(width, height)
        
    simplified_width = width // gcd
    simplified_height = height // gcd
        
    return f"{simplified_width}:{simplified_height}"


