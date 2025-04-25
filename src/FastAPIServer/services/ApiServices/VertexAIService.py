import concurrent.futures
import os
import google.auth
import google.auth.transport.requests
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from src.data_models.ModalAppSchemas import Veo2Parameters
from src.data_models.ModalAppSchemas import IdeoGramText2ImageParameters
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import (
    IMAGEGEN_ASPECT_RATIOS,
    IMAGEGEN_ERROR,
    IMAGEGEN_ERROR_MSG,
    OUTPUT_IMAGE_EXTENSION,
    google_credentials_info,
    OUTPUT_VIDEO_EXTENSION
)
from src.utils.Globals import (
    convert_to_aspect_ratio,
    prepare_response,
    timing_decorator,
)
from loguru import logger
from google.cloud import storage
import base64
import os
import time
import uuid
from urllib.parse import urlparse
import requests
from google.cloud import storage
from loguru import logger
from pydantic import ValidationError
import json


class ImageGenText2Image(IService):
    def __init__(self) -> None:
        """
        Initializes the Gemini service with appropriate API endpoints.

        Sets up the service by inheriting API credentials from the parent IService
        class and configuring the specific endpoint URLs needed for this operation.
        """
        super().__init__()
        credentials, project_id = google.auth.load_credentials_from_dict(
            google_credentials_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        request = google.auth.transport.requests.Request()
        credentials.refresh(request)
        vertexai.init(project="marine-potion-404413", credentials=credentials)
        self.generation_model = ImageGenerationModel.from_pretrained(
            "imagen-3.0-generate-001"
        )

    def make_api_request(self, parameters: IdeoGramText2ImageParameters) -> str:
        """
        Processes an individual image generation request through the SDXL API.

        This function constructs the appropriate API payload from the parameters,
        sends the request to the specified API endpoint, and processes the response
        including NSFW content detection and result formatting.

        Args:
            parameters: Configuration parameters specific to the operation type
            *args: Additional arguments specific to the operation type

        Returns:
            list: A list containing the generated image data and NSFW flag

        Raises:
            Exception: If the generated content is flagged as NSFW or an API error occurs
        """
        image = self.generation_model.generate_images(
            prompt=parameters.prompt,
            number_of_images=1,
            aspect_ratio=parameters.aspect_ratio,
            safety_filter_level="block_some",
        )
        if len(image.images) == 0:
            raise Exception(IMAGEGEN_ERROR, IMAGEGEN_ERROR_MSG)
        return image.images[0]._image_bytes

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        """
        Entry point for the service that handles batch processing of requests.

        This method validates the input parameters, prepares any required resources
        (like input images or masks), and creates multiple parallel generation tasks
        based on the batch size. The @timing_decorator tracks and adds execution
        time to the response.

        Args:
            parameters (dict): Request parameters for the specific operation

        Returns:
            dict: Standardized response containing generated images, NSFW flags,
                timing information, and file format

        Raises:
            Exception: If parameter validation fails or the API returns errors
        """
        parameters: IdeoGramText2ImageParameters = IdeoGramText2ImageParameters(
            **parameters
        )

        parameters.aspect_ratio = convert_to_aspect_ratio(
            parameters.width, parameters.height
        )
        if parameters.aspect_ratio not in IMAGEGEN_ASPECT_RATIOS:
            raise Exception("Invalid Height and width dimension")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(parameters.batch):
                future = executor.submit(self.make_api_request, parameters)
                futures.append(future)

            results = [future.result() for future in futures]

        Has_NSFW_Content = [False] * parameters.batch

        return prepare_response(results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION)
    

class VertexAIVeo(IService):
    def __init__(self) -> None:
        super().__init__()
        self.project_id = None
        self.credentials = None
        self.model_id = "veo-2.0-generate-001"
        # --- {{ CHANGE: Use specific location for Veo endpoint }} ---
        self.location = "us-central1" # Hardcode or prioritize 'us-central1' for Veo
        # --- End Change ---
        self.api_endpoint = f"https://{self.location}-aiplatform.googleapis.com"
        self.predict_url = None
        self.fetch_url = None
        self.output_bucket_name = os.getenv("BUCKET_NAME")
        if not self.output_bucket_name:
            # Raise error - service cannot function without a configured bucket
            raise ValueError("BUCKET_NAME environment variable is required for VertexAIVeo GCS output.")
        self.storage_client = None

        logger.info(f"Initializing VertexAIVeo Service for location: {self.location}...") # Log the used location
        try:
            scopes = ["https://www.googleapis.com/auth/cloud-platform"]
            # Use explicit credential loading from dict if preferred, or default()
            if google_credentials_info:
                self.credentials, self.project_id = google.auth.load_credentials_from_dict(
                    google_credentials_info, scopes=scopes
                )
            else:
                 self.credentials, self.project_id = google.auth.default(scopes=scopes)

            # Fallback logic for project_id
            if not self.project_id: self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            if not self.project_id: self.project_id = os.getenv("GCP_PROJECT_ID")
            if not self.project_id: raise ValueError("Could not determine Google Project ID.")

            self.storage_client = storage.Client(project=self.project_id, credentials=self.credentials)
            # --- {{ CHANGE: Use self.location in URLs }} ---
            self.predict_url = f"{self.api_endpoint}/v1/projects/{self.project_id}/locations/{self.location}/publishers/google/models/{self.model_id}:predictLongRunning"
            self.fetch_url = f"{self.api_endpoint}/v1/projects/{self.project_id}/locations/{self.location}/publishers/google/models/{self.model_id}:fetchPredictOperation"
            # --- End Change ---

            self._refresh_credentials()
            logger.info(f"VertexAIVeo Initialized: Project={self.project_id}, Location={self.location}, Output Bucket='{self.output_bucket_name}'")
        except Exception as e:
            logger.exception("VertexAIVeo initialization failed.")
            self.predict_url = None; self.fetch_url = None; self.storage_client = None
            raise RuntimeError("VertexAIVeo initialization failed") from e


    def _refresh_credentials(self):
        if hasattr(self.credentials, 'valid') and not self.credentials.valid and hasattr(self.credentials, 'refresh'):
            try:
                auth_req = google.auth.transport.requests.Request()
                self.credentials.refresh(auth_req)
            except Exception as e: logger.exception("VertexAIVeo: Failed to refresh credentials.")

    def _get_access_token(self) -> str:
        if not self.credentials: raise RuntimeError("VertexAIVeo: Credentials not initialized.")
        self._refresh_credentials()
        try:
            auth_req = google.auth.transport.requests.Request()
            self.credentials.before_request(auth_req, method="POST", url=self.predict_url, headers={})
            token = self.credentials.token
            if not token: raise ValueError("VertexAIVeo: Failed to get access token.")
            return token
        except Exception as e: raise ValueError("VertexAIVeo: Failed to get access token.") from e

    def _poll_operation(self, operation_name: str, timeout_seconds: int = 600, poll_interval: int = 20) -> dict:
        if not self.fetch_url: raise RuntimeError("VertexAIVeo not initialized.")
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            try:
                headers = {"Authorization": f"Bearer {self._get_access_token()}", "Content-Type": "application/json; charset=utf-8"}
                payload = {"operationName": operation_name}
                response = requests.post(self.fetch_url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                op_result = response.json()
                if op_result.get("done"):
                    if op_result.get("error"): raise Exception("Vertex AI Operation Error", f"{op_result['error'].get('message', 'Unknown API error')}")
                    return op_result.get("response", {})
                time.sleep(poll_interval)
            except requests.exceptions.RequestException as e: time.sleep(poll_interval * 2)
            except Exception as e: raise
        raise TimeoutError(f"Operation {operation_name} timed out after {timeout_seconds} seconds.")

    def _download_from_gcs(self, gcs_uri: str) -> bytes:
        if not self.storage_client: raise RuntimeError("GCS Storage client not initialized.")
        if not gcs_uri or not gcs_uri.startswith("gs://"): raise ValueError(f"Invalid GCS URI: {gcs_uri}")
        try:
            parsed_uri = urlparse(gcs_uri); bucket_name = parsed_uri.netloc; blob_name = parsed_uri.path.lstrip('/')
            if not bucket_name or not blob_name: raise ValueError(f"Could not parse GCS URI: {gcs_uri}")
            bucket = self.storage_client.bucket(bucket_name); blob = bucket.blob(blob_name)
            if not blob.exists(timeout=30): raise FileNotFoundError(f"GCS file not found: {gcs_uri}")
            video_bytes = blob.download_as_bytes(timeout=120)
            if os.getenv("DELETE_GCS_AFTER_DOWNLOAD", "false").lower() == "true":
                try: blob.delete(timeout=30)
                except Exception as del_ex: logger.warning(f"Failed to delete GCS file {gcs_uri}: {del_ex}")
            return video_bytes
        except Exception as e: raise Exception("GCS Download Error", f"Failed: {e}") from e

    def _get_bytes_from_url(self, url: str) -> bytes:
         response = requests.get(url, timeout=60); response.raise_for_status(); return response.content

    def make_api_request(self, parameters: Veo2Parameters) -> bytes:
        if not self.predict_url or not self.output_bucket_name: raise RuntimeError("VertexAIVeo not initialized or BUCKET_NAME missing.")
        timestamp = int(time.time()); request_uuid = str(uuid.uuid4()).split('-')[0]
        output_filename = f"veo_output_{timestamp}_{request_uuid}.mp4"
        gcs_output_uri = f"gs://{self.output_bucket_name}/veo_outputs/{output_filename}"
        headers = {"Authorization": f"Bearer {self._get_access_token()}", "Content-Type": "application/json; charset=utf-8"}
        instance = {"prompt": parameters.prompt}

        # --- {{ REVERTED api_parameters to match working code }} ---
        # Add durationSec and aspectRatio back into this dictionary
        api_parameters = {
            "storageUri": gcs_output_uri,
            "sampleCount": 1,
            "durationSec": int(parameters.duration.replace('s', '')), # RESTORED
            "aspectRatio": parameters.aspect_ratio                  # RESTORED
        }
        # --- End Reversion ---

        if parameters.file_url:
            try:
                image_bytes = self._get_bytes_from_url(parameters.file_url)
                if not image_bytes: raise ValueError("Downloaded image bytes are empty.")
                encoded_image = base64.b64encode(image_bytes).decode("utf-8"); mime_type = "image/jpeg"
                instance["image"] = {"bytesBase64Encoded": encoded_image, "mimeType": mime_type}
            except Exception as e: raise Exception("Image Processing Error", f"Failed: {e}") from e

        payload = {"instances": [instance], "parameters": api_parameters}; op_name = None
        try:
            payload_log_safe = json.dumps({'instances': [{'prompt': instance.get('prompt'), 'image_present': 'image' in instance}], 'parameters': api_parameters}, indent=2)
            logger.debug(f"VertexAIVeo Request Payload (Restored Params): {payload_log_safe}")

            response = requests.post(self.predict_url, headers=headers, json=payload, timeout=60); response.raise_for_status()
            op_info = response.json(); op_name = op_info.get("name")
            if not op_name: raise Exception("Vertex AI Error", "Failed to get operation name.")
        except Exception as e:
             error_details = str(e)
             if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
                 try: error_details = f"{e.response.status_code} - {e.response.text}"
                 except: pass
             logger.exception(f"VertexAIVeo submit failed: {error_details}")
             raise Exception("Vertex AI Submit Error", f"Failed: {error_details}") from e

        # --- Polling, GCS Download, Result Processing (Keep as is) ---
        try:
            final_response = self._poll_operation(op_name)
            videos_list = final_response.get("videos", []); gcs_uri = videos_list[0].get("gcsUri") if videos_list else None
            if not gcs_uri: raise Exception("Vertex AI Error", "Video GCS URI missing.")
            video_bytes = self._download_from_gcs(gcs_uri)
            if not video_bytes: raise ValueError("Downloaded video bytes are empty.")
            return video_bytes
        except Exception as e:
             error_prefix = "Vertex AI Processing Error"
             if isinstance(e, FileNotFoundError): error_prefix = "GCS File Not Found"
             elif isinstance(e, TimeoutError): error_prefix = "Operation Timeout"
             raise Exception(error_prefix, f"{e}") from e

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        try:
            params: Veo2Parameters = Veo2Parameters(**parameters)
            duration_int = int(params.duration.replace('s', ''))
            if not (5 <= duration_int <= 8): raise ValueError(f"Duration must be 5-8s, got {duration_int}s")
        except (ValidationError, ValueError) as e: raise Exception("Input Validation Error", str(e)) from e
        video_bytes = self.make_api_request(params)
        result_list = [video_bytes] if isinstance(video_bytes, bytes) else []
        return prepare_response(result=result_list, Has_NSFW_content=[False]*len(result_list), time_data=0, runtime=0, extension=OUTPUT_VIDEO_EXTENSION)


# --- Add VeoRouterService class ---
class VeoRouterService(IService):
    def __init__(self, vertex_veo_provider: callable, fal_veo_provider: callable):
        super().__init__()
        if not callable(vertex_veo_provider) or not callable(fal_veo_provider): raise TypeError("Providers must be callable")
        self._get_primary_service = vertex_veo_provider
        self._get_fallback_service = fal_veo_provider

    def remote(self, parameters: dict) -> dict:
        primary_service = self._get_primary_service() # Get instance via provider
        try:
            logger.info(f"VeoRouter: Attempting primary: {primary_service.__class__.__name__}")
            return primary_service.remote(parameters)
        except Exception as primary_error:
             # Check if it's our specific validation error format
            is_validation_error = (len(primary_error.args) >= 1 and
                                   primary_error.args[0] == "Input Validation Error")

            if is_validation_error:
                logger.error(f"VeoRouter: Primary {primary_service.__class__.__name__} validation failed: {primary_error.args[1]}. Re-raising.")
                raise primary_error # Re-raise the original validation error
            else:
                # --- {{ MODIFIED LOGGING }} ---
                # Log the actual error type and message from the primary service failure
                logger.warning(f"VeoRouter: Primary {primary_service.__class__.__name__} failed with {type(primary_error).__name__}: {primary_error}. Trying fallback.")
                # --- End Modified Logging ---
                fallback_service = self._get_fallback_service() # Get instance via provider
                try:
                    logger.info(f"VeoRouter: Attempting fallback: {fallback_service.__class__.__name__}")
                    return fallback_service.remote(parameters)
                except Exception as fallback_error:
                     logger.error(f"VeoRouter: Fallback {fallback_service.__class__.__name__} also failed: {type(fallback_error).__name__} - {fallback_error}")
                     raise fallback_error from primary_error
