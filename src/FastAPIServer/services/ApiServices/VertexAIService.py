import concurrent.futures
import os
from typing import List # Added List import
import google.auth
import google.auth.transport.requests
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from src.data_models.ModalAppSchemas import Veo2Parameters, IdeoGramText2ImageParameters, Lyria2MusicGenerationParameters
from src.FastAPIServer.services.IService import IService
from src.utils.Constants import (
    IMAGEGEN_ASPECT_RATIOS,
    IMAGEGEN_ERROR,
    IMAGEGEN_ERROR_MSG,
    OUTPUT_IMAGE_EXTENSION,
    google_credentials_info,
    OUTPUT_VIDEO_EXTENSION,
    VIDEO_GENERATION_ERROR,
    OUTPUT_AUDIO_EXTENSION,
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
        # Use google_credentials_info if available, otherwise default might fail if not configured
        if not google_credentials_info:
            logger.warning(
                "google_credentials_info not found in Constants, ImageGenText2Image might fail."
            )
            # Fallback or raise error depending on requirements
            # For now, proceed assuming it's defined elsewhere or default works
            self.credentials, self.project_id = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            if not self.project_id:
                self.project_id = os.getenv(
                    "GCP_PROJECT_ID", "marine-potion-404413"
                )  # Default project
        else:
            self.credentials, self.project_id = google.auth.load_credentials_from_dict(
                google_credentials_info,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )

        request = google.auth.transport.requests.Request()
        # Handle potential refresh errors
        try:
            self.credentials.refresh(request)
        except Exception as e:
            logger.error(
                f"Failed to refresh credentials during ImageGenText2Image init: {e}"
            )
            # Decide how to handle - raise error or continue?
            # raise RuntimeError("Credential refresh failed during init") from e

        # Use the determined project_id
        vertexai.init(project=self.project_id, credentials=self.credentials)
        # Ensure model name is correct
        self.generation_model = ImageGenerationModel.from_pretrained(
            "imagen-4.0-generate-preview-05-20"
        )
        logger.info(f"ImageGenText2Image Initialized with project: {self.project_id}")

    def make_api_request(
        self, parameters: IdeoGramText2ImageParameters
    ) -> bytes:  # Return type should be bytes
        """
        Processes an individual image generation request through the Vertex AI API.
        """
        logger.debug(
            f"Generating image with prompt: {parameters.prompt}, aspect_ratio: {parameters.aspect_ratio}"
        )
        try:
            image_response = self.generation_model.generate_images(
                prompt=parameters.prompt,
                number_of_images=1,  # Generate one image per call for batching in remote
                aspect_ratio=parameters.aspect_ratio,
                # Consider adding negative_prompt if the model supports it
                # negative_prompt=parameters.negative_prompt,
                safety_filter_level="block_some",  # Adjust safety level if needed
            )
            if not image_response.images:  # Check if the list is empty
                logger.error("Vertex AI generate_images returned no images.")
                # Use specific error constant if available
                raise Exception(
                    IMAGEGEN_ERROR
                    if "IMAGEGEN_ERROR" in globals()
                    else VIDEO_GENERATION_ERROR
                )
            # Return the bytes of the first image
            return image_response.images[0]._image_bytes
        except Exception as e:
            logger.error(
                f"Error during Vertex AI generate_images call: {e}", exc_info=True
            )
            # Re-raise with a generic internal error message
            raise Exception(
                IMAGEGEN_ERROR
                if "IMAGEGEN_ERROR" in globals()
                else VIDEO_GENERATION_ERROR
            ) from e

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        """
        Entry point for the image generation service using Vertex AI.
        Validates parameters, calculates aspect ratio, handles batching, and prepares response.
        Ensures ValueError for invalid aspect ratio propagates correctly.
        """
        validated_params: IdeoGramText2ImageParameters = None
        try:
            # Step 1: Validate parameters using Pydantic
            validated_params = IdeoGramText2ImageParameters(**parameters)

            # Step 2: Calculate and validate aspect ratio *within the same try block*
            validated_params.aspect_ratio = convert_to_aspect_ratio(
                validated_params.width, validated_params.height
            )
            if validated_params.aspect_ratio not in IMAGEGEN_ASPECT_RATIOS:
                # Raise ValueError directly if invalid
                raise ValueError(
                    "Invalid Height and width dimensions resulting in unsupported aspect ratio."
                )

        except (ValidationError, ValueError) as e:
            # Catch both Pydantic validation errors and our specific aspect ratio ValueError
            logger.error(
                f"Input validation failed for ImageGenText2Image: {e}", exc_info=False
            )
            # Re-raise as ValueError which the test expects
            raise ValueError(f"Invalid input parameters: {e}") from e

        # --- This part is only reached if ALL validation passes ---
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = [
                    executor.submit(self.make_api_request, validated_params)
                    for _ in range(validated_params.batch)
                ]
                results = [future.result() for future in futures]

            Has_NSFW_Content = [False] * validated_params.batch

            return prepare_response(
                results, Has_NSFW_Content, 0, 0, OUTPUT_IMAGE_EXTENSION
            )

        except Exception as e:
            # Catches errors only from make_api_request or thread execution
            logger.error(
                f"Error during ImageGenText2Image processing: {type(e).__name__} - {e}",
                exc_info=True,
            )
            raise Exception(
                IMAGEGEN_ERROR
                if "IMAGEGEN_ERROR" in globals()
                else VIDEO_GENERATION_ERROR
            ) from e


class VertexAIVeo(IService):
    def __init__(self) -> None:
        super().__init__()
        self.project_id = None
        self.credentials = None
        self.model_id = "veo-2.0-generate-001"
        self.location = "us-central1"
        self.api_endpoint = f"https://{self.location}-aiplatform.googleapis.com"
        self.predict_url = None
        self.fetch_url = None
        self.output_bucket_name = os.getenv("BUCKET_NAME")
        if not self.output_bucket_name:
            raise ValueError(
                "BUCKET_NAME environment variable is required for VertexAIVeo GCS output."
            )
        self.storage_client = None

        logger.info(
            f"Initializing VertexAIVeo Service for location: {self.location}..."
        )
        try:
            scopes = ["https://www.googleapis.com/auth/cloud-platform"]
            if google_credentials_info:
                self.credentials, self.project_id = (
                    google.auth.load_credentials_from_dict(
                        google_credentials_info, scopes=scopes
                    )
                )
            else:
                self.credentials, self.project_id = google.auth.default(scopes=scopes)

            if not self.project_id:
                self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            if not self.project_id:
                self.project_id = os.getenv("GCP_PROJECT_ID")
            if not self.project_id:
                raise ValueError("Could not determine Google Project ID.")

            self.storage_client = storage.Client(
                project=self.project_id, credentials=self.credentials
            )
            self.predict_url = f"{self.api_endpoint}/v1/projects/{self.project_id}/locations/{self.location}/publishers/google/models/{self.model_id}:predictLongRunning"
            self.fetch_url = f"{self.api_endpoint}/v1/projects/{self.project_id}/locations/{self.location}/publishers/google/models/{self.model_id}:fetchPredictOperation"  # This endpoint might be wrong

            self._refresh_credentials()
            logger.info(
                f"VertexAIVeo Initialized: Project={self.project_id}, Location={self.location}, Output Bucket='{self.output_bucket_name}'"
            )
        except Exception as e:
            logger.exception("VertexAIVeo initialization failed.")
            self.predict_url = None
            self.fetch_url = None
            self.storage_client = None
            raise Exception(VIDEO_GENERATION_ERROR) from e  # Updated

    def _refresh_credentials(self):
        if (
            hasattr(self.credentials, "valid")
            and not self.credentials.valid
            and hasattr(self.credentials, "refresh")
        ):
            try:
                auth_req = google.auth.transport.requests.Request()
                self.credentials.refresh(auth_req)
            except Exception as e:
                logger.exception("VertexAIVeo: Failed to refresh credentials.")

    def _get_access_token(self) -> str:
        if not self.credentials:
            raise RuntimeError("VertexAIVeo: Credentials not initialized.")
        self._refresh_credentials()
        try:
            auth_req = google.auth.transport.requests.Request()
            self.credentials.before_request(
                auth_req, method="POST", url=self.predict_url, headers={}
            )
            token = self.credentials.token
            if not token:
                raise ValueError("VertexAIVeo: Failed to get access token.")
            return token
        except Exception as e:
            raise ValueError("VertexAIVeo: Failed to get access token.") from e

    def _poll_operation(
        self, operation_name: str, timeout_seconds: int = 600, poll_interval: int = 20
    ) -> dict:
        # Note: The fetch_url used here might need adjustment based on Vertex AI API for polling operations
        if not self.fetch_url:
            raise RuntimeError("VertexAIVeo polling URL not initialized.")
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            try:
                headers = {
                    "Authorization": f"Bearer {self._get_access_token()}",
                    "Content-Type": "application/json; charset=utf-8",
                }
                # The payload might need adjustment depending on the actual polling endpoint structure
                payload = {"operationName": operation_name}
                response = requests.post(
                    self.fetch_url, headers=headers, json=payload, timeout=30
                )
                response.raise_for_status()
                op_result = response.json()
                if op_result.get("done"):
                    if op_result.get("error"):
                        raise Exception(VIDEO_GENERATION_ERROR)  # Updated
                    return op_result.get("response", {})
                time.sleep(poll_interval)
            except requests.exceptions.RequestException as e:
                time.sleep(poll_interval * 2)
            except Exception as e:
                raise Exception(VIDEO_GENERATION_ERROR) from e  # Updated
        raise TimeoutError(
            f"Operation {operation_name} timed out after {timeout_seconds} seconds."
        )  # Keep specific TimeoutError

    def _download_from_gcs(self, gcs_uri: str) -> bytes:
        if not self.storage_client:
            raise RuntimeError("GCS Storage client not initialized.")
        if not gcs_uri or not gcs_uri.startswith("gs://"):
            raise ValueError(f"Invalid GCS URI: {gcs_uri}")
        try:
            parsed_uri = urlparse(gcs_uri)
            bucket_name = parsed_uri.netloc
            blob_name = parsed_uri.path.lstrip("/")
            if not bucket_name or not blob_name:
                raise ValueError(f"Could not parse GCS URI: {gcs_uri}")
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            if not blob.exists(timeout=30):
                raise FileNotFoundError(
                    f"GCS file not found: {gcs_uri}"
                )  # Keep specific FileNotFoundError
            video_bytes = blob.download_as_bytes(timeout=120)
            if os.getenv("DELETE_GCS_AFTER_DOWNLOAD", "false").lower() == "true":
                try:
                    blob.delete(timeout=30)
                except Exception as del_ex:
                    logger.warning(f"Failed to delete GCS file {gcs_uri}: {del_ex}")
            return video_bytes
        except Exception as e:
            raise Exception(VIDEO_GENERATION_ERROR) from e  # Updated

    def _get_bytes_from_url(self, url: str) -> bytes:
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            return response.content
        except Exception as e:
            raise Exception(VIDEO_GENERATION_ERROR) from e  # Updated

    def make_api_request(self, parameters: Veo2Parameters) -> bytes:
        if not self.predict_url or not self.output_bucket_name:
            raise RuntimeError("VertexAIVeo not initialized or BUCKET_NAME missing.")
        timestamp = int(time.time())
        request_uuid = str(uuid.uuid4()).split("-")[0]
        output_filename = f"veo_output_{timestamp}_{request_uuid}.mp4"
        # This path might need adjustment based on how GCS output is configured in the payload
        gcs_output_uri = f"gs://{self.output_bucket_name}/veo_outputs/{output_filename}"
        headers = {
            "Authorization": f"Bearer {self._get_access_token()}",
            "Content-Type": "application/json; charset=utf-8",
        }
        instance = {"prompt": parameters.prompt}

        # API parameters might need restructuring depending on the exact payload VEO expects
        api_parameters = {
            "storageUri": gcs_output_uri,  # Check if this is the correct parameter name
            "sampleCount": 1,
            "durationSec": int(parameters.duration.replace("s", "")),
            "aspectRatio": parameters.aspect_ratio,
        }

        # Image handling needs to check parameters.file_url, which was removed in remote()
        # This block needs adjustment if image-to-video path is ever re-enabled here
        # if parameters.file_url:
        #     try:
        #         image_bytes = self._get_bytes_from_url(parameters.file_url)
        #         if not image_bytes: raise ValueError("Downloaded image bytes are empty.")
        #         encoded_image = base64.b64encode(image_bytes).decode("utf-8"); mime_type = "image/jpeg"
        #         instance["image"] = {"bytesBase64Encoded": encoded_image, "mimeType": mime_type}
        #     except Exception as e: raise Exception(VIDEO_GENERATION_ERROR) from e # Updated

        payload = {"instances": [instance], "parameters": api_parameters}
        op_name = None
        try:
            payload_log_safe = json.dumps(
                {
                    "instances": [
                        {
                            "prompt": instance.get("prompt"),
                            "image_present": "image" in instance,
                        }
                    ],
                    "parameters": api_parameters,
                },
                indent=2,
            )
            logger.debug(f"VertexAIVeo Request Payload: {payload_log_safe}")

            response = requests.post(
                self.predict_url, headers=headers, json=payload, timeout=60
            )
            response.raise_for_status()
            op_info = response.json()
            op_name = op_info.get("name")  # Verify correct key for operation name
            if not op_name:
                raise Exception(VIDEO_GENERATION_ERROR)  # Updated
        except Exception as e:
            error_details = str(e)
            if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
                try:
                    error_details = f"{e.response.status_code} - {e.response.text}"
                except:
                    pass
            logger.exception(f"VertexAIVeo submit failed: {error_details}")
            raise Exception(VIDEO_GENERATION_ERROR) from e  # Updated

        try:
            final_response = self._poll_operation(op_name)
            # Parsing the result needs verification based on the actual VEO API response structure
            videos_list = final_response.get("videos", [])
            gcs_uri = videos_list[0].get("gcsUri") if videos_list else None
            if not gcs_uri:
                raise Exception(VIDEO_GENERATION_ERROR)  # Updated
            video_bytes = self._download_from_gcs(gcs_uri)
            if not video_bytes:
                raise ValueError(
                    "Downloaded video bytes are empty."
                )  # Keep specific ValueError
            return video_bytes
        except Exception as e:
            error_prefix = "Vertex AI Processing Error"
            if isinstance(e, FileNotFoundError):
                error_prefix = "GCS File Not Found"  # Keep specific context if possible
            elif isinstance(e, TimeoutError):
                error_prefix = "Operation Timeout"  # Keep specific context
            # Raise generic error for propagation
            raise Exception(VIDEO_GENERATION_ERROR) from e  # Updated

    def _prepare_api_parameters(self, parameters: Veo2Parameters) -> dict:
        """Prepares the non-instance specific parameters for the API call."""
        # (Implementation as provided previously)
        try:
            duration_sec = int(parameters.duration.replace("s", ""))
            if not (5 <= duration_sec <= 8):
                raise ValueError("Duration must be between 5 and 8 seconds.")
        except (ValueError, AttributeError, TypeError):
            logger.error(
                f"Invalid duration format or value received: {parameters.duration}"
            )
            raise ValueError(
                f"Invalid duration: '{parameters.duration}'. Use 'Ns' (e.g., '5s') between 5 and 8."
            )

        valid_aspect_ratios = [
            "16:9",
            "9:16",
            "1:1",
            "4:5",
            "5:4",
        ]  # Example, use actual valid ratios
        if parameters.aspect_ratio not in valid_aspect_ratios:
            logger.error(f"Invalid aspect ratio received: {parameters.aspect_ratio}")
            raise ValueError(
                f"Invalid aspect ratio: '{parameters.aspect_ratio}'. Must be one of {valid_aspect_ratios}"
            )

        return {
            "durationSec": duration_sec,
            "aspectRatio": parameters.aspect_ratio,
            # Add other static parameters if needed
        }

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        params_for_validation = parameters.copy()
        if "file_url" in params_for_validation:
            del params_for_validation["file_url"]
        try:
            params: Veo2Parameters = Veo2Parameters(**params_for_validation)
            duration_int = int(params.duration.replace("s", ""))
            if not (5 <= duration_int <= 8):
                raise ValueError(
                    f"Duration must be 5-8s, got {duration_int}s"
                )  # Keep specific validation error
        except (ValidationError, ValueError) as e:
            raise ValueError(
                f"Invalid input parameters: {e}"
            ) from e  # Re-raise as specific ValueError for input issues
        # Catch any other unexpected error during validation/param processing
        except Exception as e:
            raise Exception(VIDEO_GENERATION_ERROR) from e  # Updated

        # --- Main processing ---
        try:
            video_bytes = self.make_api_request(
                params
            )  # This now contains the poll and download logic
            result_list = [video_bytes] if isinstance(video_bytes, bytes) else []
            # Ensure time_data and runtime are properly calculated or passed if needed
            return prepare_response(
                result=result_list,
                Has_NSFW_content=[False] * len(result_list),
                time_data=0,
                runtime=0,
                extension=OUTPUT_VIDEO_EXTENSION,
            )
        except (FileNotFoundError, TimeoutError, ValueError) as e:
            # Catch specific errors from make_api_request/helpers that we want to log differently maybe
            logger.error(
                f"Error during VertexAIVeo video generation: {type(e).__name__} - {e}",
                exc_info=True,
            )
            raise Exception(VIDEO_GENERATION_ERROR) from e  # Updated
        except Exception as e:
            # Catch all other exceptions during the main processing
            logger.error(
                f"Unexpected error during VertexAIVeo video generation: {e}",
                exc_info=True,
            )
            raise Exception(VIDEO_GENERATION_ERROR) from e  # Updated


class VeoRouterService(IService):
    def __init__(self, vertex_veo_provider: callable, fal_veo_provider: callable):
        super().__init__()
        if not callable(vertex_veo_provider) or not callable(fal_veo_provider):
            raise TypeError("Providers must be callable")
        self._get_primary_service = vertex_veo_provider
        self._get_fallback_service = fal_veo_provider

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        primary_service = None
        fallback_service = None
        primary_service_name = "UnknownPrimaryService"  # Default name
        fallback_service_name = "UnknownFallbackService"  # Default name
        try:
            primary_service = self._get_primary_service()
            # Get name safely after ensuring instance exists
            if primary_service:
                primary_service_name = primary_service.__class__.__name__
            logger.info(f"VeoRouter: Attempting primary: {primary_service_name}")
            return primary_service.remote(parameters)

        except ValueError as validation_error:
            # Use the captured primary_service_name
            logger.error(
                f"VeoRouter: Input validation failed in primary service {primary_service_name}: {validation_error}",
                exc_info=False,
            )
            raise validation_error  # Re-raise validation errors directly

        except Exception as primary_error:
            # Use the captured primary_service_name
            logger.warning(
                f"VeoRouter: Primary service {primary_service_name} failed: {type(primary_error).__name__} - {primary_error}. Attempting fallback."
            )

            # Check specifically if it was a validation error mistakenly caught here
            if isinstance(
                primary_error, ValueError
            ) and "Invalid input parameters" in str(primary_error):
                logger.error(
                    f"VeoRouter: Primary service {primary_service_name} failed due to validation error, not falling back: {primary_error}"
                )
                raise primary_error  # Re-raise the validation error

            # Proceed with fallback for other exceptions
            try:
                fallback_service = self._get_fallback_service()
                # Get name safely after ensuring instance exists
                if fallback_service:
                    fallback_service_name = fallback_service.__class__.__name__
                logger.info(f"VeoRouter: Attempting fallback: {fallback_service_name}")
                return fallback_service.remote(parameters)

            except ValueError as fallback_validation_error:
                # Use the captured fallback_service_name
                logger.error(
                    f"VeoRouter: Input validation failed in fallback service {fallback_service_name}: {fallback_validation_error}",
                    exc_info=False,
                )
                raise fallback_validation_error  # Re-raise fallback validation errors directly

            except Exception as fallback_error:
                # Use the captured fallback_service_name
                logger.error(
                    f"VeoRouter: Fallback service {fallback_service_name} also failed: {type(fallback_error).__name__} - {fallback_error}",
                    exc_info=True,
                )
                # Raise generic error after both primary and fallback failed internally
                raise Exception(VIDEO_GENERATION_ERROR) from fallback_error


class Lyria2MusicGeneration(IService):
    def __init__(self) -> None:
        super().__init__()
        self.project_id = None
        self.credentials = None
        self.location = "us-central1"  # As per notebook
        self.model_id = "lyria-002"  # As per notebook
        self.music_model_endpoint = None

        logger.info(
            f"Initializing Lyria2MusicGeneration Service for project and location: {self.location}..."
        )
        try:
            scopes = ["https://www.googleapis.com/auth/cloud-platform"]
            if google_credentials_info:
                self.credentials, self.project_id = (
                    google.auth.load_credentials_from_dict(
                        google_credentials_info, scopes=scopes
                    )
                )
            else:
                logger.warning(
                    "google_credentials_info not found in Constants, Lyria2MusicGeneration might fail to determine project ID automatically."
                )
                self.credentials, self.project_id = google.auth.default(scopes=scopes)
            
            if not self.project_id:
                self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            if not self.project_id:
                self.project_id = os.getenv("GCP_PROJECT_ID")
            if not self.project_id:
                # Fallback similar to ImageGenText2Image to avoid raising new ValueError
                self.project_id = os.getenv("GCP_PROJECT_ID", "marine-potion-404413") 
                logger.warning(f"Lyria2MusicGeneration falling back to default/env project_id: {self.project_id}")

            self.music_model_endpoint = f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.location}/publishers/google/models/{self.model_id}:predict"
            
            # Refresh credentials if they were loaded
            if self.credentials:
                auth_req = google.auth.transport.requests.Request()
                try:
                    self.credentials.refresh(auth_req)
                except Exception as e:
                    # Log error but don't necessarily fail init, token fetching will handle later
                    logger.error(f"Failed to refresh credentials during Lyria2MusicGeneration init: {e}")
            
            logger.info(
                f"Lyria2MusicGeneration Initialized: Project={self.project_id}, Endpoint='{self.music_model_endpoint}'"
            )
        except Exception as e:
            logger.exception("Lyria2MusicGeneration initialization failed.")
            self.music_model_endpoint = None # Ensure endpoint is None if init fails
            raise Exception(VIDEO_GENERATION_ERROR) from e

    def _refresh_credentials(self):
        if self.credentials and hasattr(self.credentials, "valid") and not self.credentials.valid and hasattr(self.credentials, "refresh"):
            try:
                auth_req = google.auth.transport.requests.Request()
                self.credentials.refresh(auth_req)
            except Exception as e:
                logger.exception("Lyria2MusicGeneration: Failed to refresh credentials during operation.")
                # Do not raise here, let _get_access_token handle failure to get token

    def _get_access_token(self) -> str:
        if not self.credentials:
            # This case should ideally be caught by __init__ failing if creds are essential
            logger.error("Lyria2MusicGeneration: Credentials not initialized before token fetch.")
            raise RuntimeError("Lyria2MusicGeneration: Credentials not initialized.")
        
        self._refresh_credentials() # Attempt refresh before getting token
        
        try:
            # Re-fetch token directly using credentials, similar to notebook's send_request_to_google_api
            # This ensures the token is fresh if self.credentials.token was stale or None
            if not self.credentials.token: # Or if self.credentials.expired:
                 auth_req = google.auth.transport.requests.Request()
                 self.credentials.refresh(auth_req)

            token = self.credentials.token
            if not token:
                logger.error("Lyria2MusicGeneration: Failed to get access token after refresh.")
                raise ValueError("Lyria2MusicGeneration: Failed to get access token.")
            return token
        except Exception as e:
            logger.exception("Lyria2MusicGeneration: Exception during get_access_token.")
            raise ValueError("Lyria2MusicGeneration: Failed to get access token.") from e

    def make_api_request(self, parameters: Lyria2MusicGenerationParameters) -> List[bytes]:
        if not self.music_model_endpoint:
            # This implies __init__ failed.
            logger.error("Lyria2MusicGeneration.make_api_request called when service not initialized.")
            raise Exception(VIDEO_GENERATION_ERROR) 

        headers = {
            "Authorization": f"Bearer {self._get_access_token()}",
            "Content-Type": "application/json; charset=utf-8",
        }

        instance_payload = {"prompt": parameters.prompt}
        if parameters.negative_prompt is not None: # Check for None explicitly
            instance_payload["negative_prompt"] = parameters.negative_prompt
        if parameters.sample_count is not None:
            instance_payload["sample_count"] = parameters.sample_count
        elif parameters.seed is not None: # elif, as they are mutually exclusive per schema
            instance_payload["seed"] = parameters.seed
        # If neither sample_count nor seed is provided, the API might have a default (e.g., sample_count=1)
        
        payload = {"instances": [instance_payload], "parameters": {}} # Parameters is an empty dict for Lyria

        try:
            logger.debug(f"Lyria2MusicGeneration Request Payload: {json.dumps(payload, indent=2)}")
            response = requests.post(
                self.music_model_endpoint, headers=headers, json=payload, timeout=180 # Increased timeout for music gen
            )
            response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
            response_json = response.json()
            
            predictions = response_json.get("predictions") # Do not default to [], check if key exists
            if predictions is None: # Check if 'predictions' key is missing or None
                logger.error("Lyria2 API response missing 'predictions' key.")
                raise Exception(VIDEO_GENERATION_ERROR) 
            if not isinstance(predictions, list) or not predictions: # Check if it's an empty list or not a list
                logger.error(f"Lyria2 API returned no predictions or invalid format: {predictions}")
                raise Exception(VIDEO_GENERATION_ERROR)

            audio_bytes_list = []
            for pred_item in predictions:
                if isinstance(pred_item, dict) and "bytesBase64Encoded" in pred_item:
                    audio_bytes_list.append(base64.b64decode(pred_item["bytesBase64Encoded"]))
                else:
                    logger.warning(f"Prediction item missing bytesBase64Encoded or not a dict: {pred_item}")
            
            if not audio_bytes_list: # If loop completed but list is still empty
                logger.error("No valid audio data found in predictions.")
                raise Exception(VIDEO_GENERATION_ERROR)

            return audio_bytes_list
        except requests.exceptions.RequestException as e: # Covers network errors, timeouts, HTTPError
            logger.exception(f"Lyria2MusicGeneration API request failed: {e}")
            raise Exception(VIDEO_GENERATION_ERROR) from e
        except (json.JSONDecodeError, KeyError, TypeError, base64.binascii.Error) as e: # Specific parsing/data errors
            logger.exception(f"Error processing Lyria2MusicGeneration response: {e}")
            raise Exception(VIDEO_GENERATION_ERROR) from e
        except Exception as e: # Catch-all for other unexpected errors
            logger.exception(f"Unexpected error in Lyria2MusicGeneration.make_api_request: {e}")
            raise Exception(VIDEO_GENERATION_ERROR) from e

    @timing_decorator
    def remote(self, parameters: dict) -> dict:
        try:
            # User's fix for nested parameters, assuming 'parameters' key exists
            # and Lyria2MusicGenerationParameters now includes a 'batch' field (e.g., batch: int = 1)
            actual_params_for_lyria = parameters.get('parameters', parameters) 
            validated_params = Lyria2MusicGenerationParameters(**actual_params_for_lyria)
        except ValidationError as e:
            logger.error(f"Input validation failed for Lyria2MusicGeneration: {e}", exc_info=False)
            raise ValueError(f"Invalid input parameters: {e}") from e
        
        try:
            all_audio_results_bytes_list = []
            
            # Assuming validated_params.batch exists and is an int (e.g., default=1 in Pydantic model)
            batch_count = validated_params.batch if hasattr(validated_params, 'batch') and isinstance(validated_params.batch, int) and validated_params.batch > 0 else 1

            if batch_count > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor: # Using 8 as in ImageGen
                    futures = [
                        executor.submit(self.make_api_request, validated_params)
                        for _ in range(batch_count)
                    ]
                    results_from_futures = [future.result() for future in futures]
                    
                    for single_call_results_list in results_from_futures:
                        all_audio_results_bytes_list.extend(single_call_results_list)
            else: # batch_count is 1
                all_audio_results_bytes_list = self.make_api_request(validated_params)
            
            has_nsfw_content = [False] * len(all_audio_results_bytes_list)

            return prepare_response(
                result=all_audio_results_bytes_list,
                Has_NSFW_content=has_nsfw_content,
                time_data=0,  # timing_decorator will populate this at a higher level if used on remote
                runtime=0,    # timing_decorator will populate this
                extension=OUTPUT_AUDIO_EXTENSION, # Ensured correct extension
            )
        except ValueError as ve: # Catch validation errors re-raised from this remote or make_api_request
            raise ve 
        except Exception as e:
            logger.error(
                f"Error during Lyria2MusicGeneration processing: {type(e).__name__} - {e}",
                exc_info=True, 
            )
            raise Exception(VIDEO_GENERATION_ERROR) from e
