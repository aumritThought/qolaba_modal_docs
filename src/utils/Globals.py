from src.data_models.ModalAppSchemas import TaskResponse, TimeData, UpscaleParameters, ChunkSummary, Subtopics
from modal import Image as MIM
from modal import Secret
from PIL import Image
from PIL.Image import Image as Imagetype
from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
import torch, time, os, requests, re, io, datetime, uuid, imageio, math, json, subprocess, tempfile
from src.utils.Constants import BASE_IMAGE_COMMANDS, IMAGE_FETCH_ERROR, STAGING_API, IMAGE_FETCH_ERROR_MSG, IMAGE_GENERATION_ERROR, NSFW_CONTENT_DETECT_ERROR_MSG, PYTHON_VERSION, REQUIREMENT_FILE_PATH, MEAN_HEIGHT, SDXL_REFINER_MODEL_PATH, google_credentials_info, OUTPUT_IMAGE_EXTENSION, SECRET_NAME, content_type, MAX_UPLOAD_RETRY
from fastapi.security import HTTPAuthorizationCredentials
from requests import Response
from google.cloud import storage
from google.oauth2 import service_account
import numpy as np
from modal import Cls
from PIL import ImageFilter
from pillow_heif import register_heif_opener
from typing import List, Dict, Tuple
from pydantic import BaseModel
from openai import OpenAI
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource
)


register_heif_opener()

#Safety Checker Utils
class SafetyChecker:
    def __init__(self) -> None:
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor()

    def check_nsfw_content(self, image : Imagetype) -> list[bool]:
        image = image.convert("RGB")
        safety_checker_input = self.feature_extractor(np.array(image), return_tensors="pt").to("cuda")

        image, has_nsfw_concept = self.safety_checker(
            images= np.array(image), clip_input=safety_checker_input.pixel_values
        )

        return has_nsfw_concept
    
def download_safety_checker():
    SafetyChecker()

#Modal image Utils
def get_base_image() -> MIM:
    return MIM.debian_slim(python_version="3.11.8").run_commands(BASE_IMAGE_COMMANDS).pip_install_from_requirements(REQUIREMENT_FILE_PATH).run_function(download_safety_checker, gpu = "t4", secrets = [Secret.from_name(SECRET_NAME)])
    

#Modal app utils
def get_refiner(pipe) -> DiffusionPipeline:
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
    task_response = TaskResponse(
        result = result,
        Has_NSFW_Content = Has_NSFW_content,
        time = TimeData(startup_time = time_data, runtime = runtime),
        extension=extension        
    )
    return task_response.model_dump()

def create_video(frames : list[Imagetype], vid_name : str, fps : int) -> None:
    writer = imageio.get_writer(vid_name, fps=fps)

    for img in frames:
        img_np = np.array(img)
        writer.append_data(img_np)

    writer.close()


#Completing request 
def make_request(url: str, method: str, json_data: dict = None, headers: dict = None, files : dict = None, json : dict = None) -> Response:

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
    for i in range(0, MAX_UPLOAD_RETRY):
        url = upload_to_gcp(data, extension, upscale)
        if(url != None and url != ""):
            break
    if(url == "" or url == None):
        raise Exception("Received an empty URL")
    return url
            
def upload_to_gcp(data : Imagetype | str, extension : str, upscale : bool = False) -> str:
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
    mask_array = np.array(img)

    inverted_mask_array = 255 - mask_array

    inverted_mask = Image.fromarray(inverted_mask_array)
    
    return inverted_mask

def simple_boundary_blur(pil_image : Imagetype, blur_radius=25) -> Imagetype:
    return pil_image.filter(ImageFilter.GaussianBlur(blur_radius))


def get_seed_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def compress_image(image : Imagetype, max_size : int = 100) -> bytes | str:
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
    if api_key.credentials != os.environ["API_KEY"]:
        raise Exception("Invalid API Key")

def get_clean_name(name : str) -> str:
    pattern = re.compile('[^a-zA-Z0-9]')
    cleaned_string = pattern.sub('', name)
    return cleaned_string.lower()


def convert_to_aspect_ratio(width : int, height : int) -> str:
    gcd = math.gcd(width, height)
        
    simplified_width = width // gcd
    simplified_height = height // gcd
        
    return f"{simplified_width}:{simplified_height}"




def process_video_url(video_url: str) -> List[Dict]:
    temp_dir = tempfile.mkdtemp()
    
    try:
        temp_video_path = os.path.join(temp_dir, "temp_video.mp4")
        temp_audio_path = os.path.join(temp_dir, "temp_audio.mp3")

        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        
        with open(temp_video_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        command = [
            "ffmpeg", 
            "-i", temp_video_path,  
            "-q:a", "0",      
            "-map", "a",       
            temp_audio_path, 
            "-y"          
        ]
            
        subprocess.run(command, check=True)

        deepgram = DeepgramClient(os.environ["DEEPGRAP_API_KEY"])

        with open(temp_audio_path, "rb") as file:
            buffer_data = file.read()

        payload: FileSource = {
                "buffer": buffer_data,
        }

        options = PrerecordedOptions(
            model="nova-2",
            detect_language=True,
        )

        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)

        data = response.to_dict()

        final_subtitles = []
        words = data['results']['channels'][0]['alternatives'][0]['words']

        current_subtitle = []
        current_start = None
        current_end = None

        for word in words:
            if current_start is None:
                current_start = word['start']

            current_subtitle.append(word['word'])
            current_end = word['end']

            # If the word ends with punctuation or subtitle is getting too long, create a new subtitle
            if word['word'][-1] in '.!?' or len(' '.join(current_subtitle)) > 40:
                timstamp = {
                    "start": current_start,
                    "end": current_end,
                    "subtitle": " ".join(current_subtitle)
                }
                current_start = None
                current_subtitle = []
                final_subtitles.append(timstamp)


    finally:
        pass
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        os.rmdir(temp_dir)

    return final_subtitles

def divide_transcript(transcript, chunk_duration=60):
    chunks = []
    current_start_time = 0
    current_chunk_text = []

    for entry in transcript:
        start_time = entry['start']
        end_time = entry['end']

        if start_time < current_start_time + chunk_duration:
            current_chunk_text.append(entry["subtitle"])
            prv_end = end_time
        else:
            chunks.append({
                "start" :  current_start_time,
                "subtitle" : " ".join(current_chunk_text),
                "end_time" : prv_end,
            })

            current_chunk_text = [entry["subtitle"]]
            current_start_time = start_time

    if current_chunk_text:
        chunks.append({
                "start" :  current_start_time,
                "subtitle" : " ".join(current_chunk_text),
                "end_time" : end_time,
            })

    return chunks



def extract_summary_and_topic(previous_summary: str, chunk: List[Dict], previous_topic : str) -> ChunkSummary:
    client = OpenAI(api_key = os.environ["OPENAI_API_KEY"])

    chunk_text = chunk["subtitle"]

    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are an assistant tasked with summarizing video transcript chunks and identifying the main topic. Use the provided previous summary, previous topic and the current chunk to generate an updated summary within 50 words and updated topic. YOU MUST make sure that new summary has balance of previous summary and current chunk. Otherwise, content appearred in start will fade away over time."},
            {"role": "user", "content": f"Here is the summary of the video so far: {previous_summary}\n\nHere is the previous topic of the video so far: {previous_topic}\n\nHere is the transcript of the current chunk: {chunk_text}\n\nPlease provide an updated summary that incorporates the new information from the current chunk. Also, identify the main topic of the video based on the updated summary."},
        ],
        response_format=ChunkSummary,
    )

    return response.choices[0].message.parsed

def process_transcript(chunks) -> Tuple[str, str]:

    overall_summary = "This video discusses various topics."
    overall_topic = "General Overview"

    for i, chunk in enumerate(chunks):
        result = extract_summary_and_topic(overall_summary, chunk, overall_topic)
        overall_summary = result.summary
        overall_topic = result.topic


    return overall_summary, overall_topic



def extract_subtopics_with_timestamps(transcript: List[Dict]) -> Subtopics:
    client = OpenAI(api_key = os.environ["OPENAI_API_KEY"])

    full_transcript_text = "\n\n".join([f"{str(entry)}" for entry in transcript])

    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": """
             Role: YouTube Chapter Generator

            Task: Generate accurate and well-placed chapters for YouTube videos based on the provided video transcript, topic, and length.

            Context:

            You are an AI-powered YouTube Chapter Generator designed to create optimal chapter placements for YouTube videos. Your goal is to analyze the provided video transcript, topic, and length to identify key points and transitions, and then generate a list of well-placed chapters with their corresponding timestamps.

            General Prompt:

            As a YouTube Chapter Generator, I need you to create chapters for my YouTube video.
             
            Please analyze the transcript carefully, considering the video topic and length, to identify the most appropriate placement for each chapter. Here are some guidelines to follow:

            -> Listen for natural pauses or transitions in the speaker's voice to identify when a new topic or section begins.
            -> Pay close attention to the content being discussed and try to identify when the speaker moves from one main point to another.
            -> If the speaker explicitly mentions a new topic or section, use that as a cue to place a timestamp.
            -> Consider the overall flow of the video and aim to place timestamps at intervals that make sense for the viewer, such as every 1-2 minutes for longer videos or at major topic shifts.
            -> If the video includes visual cues or text overlays indicating a new section, use those as a guide for placing timestamps.
            Please provide the chapters as a list, with each chapter title accompanied by its corresponding timestamp in the seconds only.
            -> YOU MUST make sure that you do not miss even single chapter so that viewers could enjoy the content based on their requirements. 
            -> While making chapters, YOU MUST make sure that chapters are not too small. Each chapter should be at least 100-120s long.
            Give your best.
             
            """},
            {"role": "user", "content": f"Here is the video transcript: {full_transcript_text}\n\nMake sure that entire video is covered"},
        ],
        response_format=Subtopics,
    )

    return response.choices[0].message.parsed
