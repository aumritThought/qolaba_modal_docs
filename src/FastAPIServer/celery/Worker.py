from celery import Celery
import time, io
from celery.signals import worker_init, worker_process_init
from src.utils.Constants import REDIS_URL, CELERY_RESULT_EXPIRATION_TIME, CELERY_MAX_RETRY, CELERY_SOFT_LIMIT, OUTPUT_IMAGE_EXTENSION
from src.data_models.ModalAppSchemas import APIInput, APITaskResponse, TaskResponse, TimeData
from celery.result import AsyncResult
from src.FastAPIServer.services.ServiceContainer import ServiceContainer, ServiceRegistry
from src.utils.Globals import upload_data_gcp, compress_image
from PIL import Image
import concurrent.futures
from PIL import Image
import io
from src.utils.Constants import INTERNAL_ERROR # Import INTERNAL_ERROR
from loguru import logger
from pydantic import ValidationError


celery = Celery(
    "task_scheduler",
    broker = REDIS_URL,
    backend = REDIS_URL,
)

celery.conf.update(
    result_expires = CELERY_RESULT_EXPIRATION_TIME
)

celery.conf.CELERY_TASK_SOFT_TIME_LIMIT = CELERY_RESULT_EXPIRATION_TIME

celery.conf.CELERY_TASK_TIME_LIMIT = CELERY_RESULT_EXPIRATION_TIME

celery.conf.worker_init = worker_init


def initialize_shared_object():
    """
    Initializes global service container and registry for the Celery worker process.
    
    This function creates a shared container object that's accessible across different
    Celery worker threads. Since Celery processes run on different threads, we need to
    initialize this container when the Celery process starts to ensure all tasks have
    access to the same service registry.
    
    The container provides access to all registered services that can be used by tasks.
    """
    global service_registry
    global container 

    container = ServiceContainer()
    service_registry = ServiceRegistry(container)
    service_registry.register_internal_services() 
    
    pass

def upload_image(index, image_data, extension, upscale=False):
    """
    Uploads an image to Google Cloud Platform storage.
    
    This function tracks the original index of the image in the batch, allowing for
    parallel processing while maintaining the correct order of results. When running
    concurrent uploads, the index ensures each URL can be correctly matched to its
    original image position.
    
    Args:
        index (int): Position index of the image in the original batch
        image_data (bytes): Raw image data to upload
        extension (str): File extension for the image (e.g., 'jpg', 'png')
        upscale (bool, optional): Whether to upscale the image. Defaults to False.
        
    Returns:
        tuple: (index, url) where index is the original position and url is the GCP storage URL
    """
    url = upload_data_gcp(image_data, extension, upscale)
    return index, url

def upload_low_res_image(index, image_data, extension):
    """
    Uploads a compressed low-resolution version of an image to GCP storage.
    
    Similar to upload_image, but compresses the image first to create a lower-resolution
    preview. This is useful for generating thumbnails or preview images that load faster
    while the full-resolution image is being processed or downloaded.
    
    Args:
        index (int): Position index of the image in the original batch
        image_data (bytes): Raw image data to compress and upload
        extension (str): File extension for the image
        
    Returns:
        tuple: (index, url) where index is the original position and url is the GCP storage URL
    """
    low_rs_image = Image.open(io.BytesIO(image_data))
    low_rs_image = compress_image(low_rs_image)
    url = upload_data_gcp(low_rs_image, extension)
    return index, url

def identify_copy_righted_materials(image_url : str, index : int) -> tuple[bool, int]:
    """
    Checks images for NSFW or copyrighted content.
    
    Uses the image checker service from the container to determine if an image contains
    inappropriate or copyrighted material. Images flagged by this function can be filtered
    out of the final results.
    
    Args:
        image_url (str): URL of the image to check
        index (int): Position index of the image in the batch
        
    Returns:
        tuple: (is_copyrighted, index) where is_copyrighted is a boolean and index is the original position
    """
    image_check_object = container.image_checker()
    return image_check_object.remote(image_url), index

@worker_process_init.connect
def on_worker_init(**kwargs):
    """
    Initializes shared objects when a worker process starts.
    
    This function is connected to Celery's worker_process_init signal and automatically
    runs when a new worker process is spawned. It ensures that each worker process has
    its own properly initialized container and service registry.
    """
    initialize_shared_object()


def get_service_list()-> dict:
    """
    Retrieves a list of available services from the service registry.
    
    Provides a way to discover what services are currently registered and available
    for use within the system.
    
    Returns:
        dict: API response containing the list of available services
    """
    task_response = APITaskResponse(output=service_registry.get_available_services())
    return task_response.model_dump()



@celery.task(name="AI_task", time_limit = CELERY_RESULT_EXPIRATION_TIME, max_retries = CELERY_MAX_RETRY, soft_time_limit = CELERY_SOFT_LIMIT)
def create_task(parameters: dict) -> dict:
    """
    Main task handler for processing AI requests.
    # ... (rest of docstring) ...
    """
    st = time.time()
    output_data = None # Initialize output_data
    task_status = "FAILED" # Default status to FAILED
    error_details = None
    error_type = INTERNAL_ERROR

    try:
        # --- Validate Input ---
        try:
            parameters_model : APIInput = APIInput(**parameters)
            logger.info(f"Validated task input for app_id: {parameters_model.app_id}")
        except ValidationError as e:
            logger.error(f"Input validation failed: {e}")
            raise Exception("Input Validation Error", str(e)) # Re-raise for Celery/handler

        app_provider = service_registry.get_service(parameters_model.app_id)
        app_instance = None

        # --- Execute Service ---
        logger.info(f"Executing service: {parameters_model.app_id}")
        if parameters_model.app_id in service_registry.api_services:
            # Handle specific dependencies like bg_remover
            if parameters_model.app_id == "falaiflux3replacebackground_api":
                 remover_instance = container.bg_remover()
                 app_instance = app_provider(remover=remover_instance) # Pass dependency
                 service_output = app_instance.remote(parameters_model.parameters)
            else:
                 app_instance = app_provider() # Instantiate the service
                 service_output = app_instance.remote(parameters_model.parameters)

            # --- Validate Service Output ---
            output_data = TaskResponse(**service_output)
            logger.info(f"API Service {parameters_model.app_id} executed successfully.")

        elif parameters_model.app_id in service_registry.modal_services:
            # Handle Modal service execution (assuming it returns TaskResponse compatible dict)
            # This part seems less likely to be the source based on the error, but good to check
            app_modal = app_provider.with_options(gpu=parameters_model.inference_type)
            app_modal_instance = app_modal(parameters_model.init_parameters)
            service_output = app_modal_instance.run_inference.remote(parameters_model.parameters)

            # --- Validate Service Output ---
            output_data = TaskResponse(**service_output)
            logger.info(f"Modal Service {parameters_model.app_id} executed successfully.")

        else:
            logger.error(f"App ID not found: {parameters_model.app_id}")
            raise Exception("Service Not Found", f"Given APP Id '{parameters_model.app_id}' is not available")

        # If we reach here without error, process results
        task_status = "SUCCESS" # Set status to SUCCESS only if everything above passed

        # --- Process Results (Uploads, etc.) ---
        if output_data and output_data.extension is not None:
            logger.info(f"Processing results with extension: {output_data.extension}")
            urls = [None] * len(output_data.result)
            low_res_urls = [None] * len(output_data.result)
            has_copyright_content = [False] * len(output_data.result)

            # Use ThreadPoolExecutor (ensure imports are present)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                full_res_futures = [executor.submit(upload_image, i, img, output_data.extension, parameters_model.upscale)
                                    for i, img in enumerate(output_data.result)]

                for future in concurrent.futures.as_completed(full_res_futures):
                    index, url = future.result()
                    urls[index] = url

                low_res_futures = []
                copyrighted_content_futures = []
                if output_data.extension == OUTPUT_IMAGE_EXTENSION:
                    if parameters_model.check_copyright_content: # Only check if requested
                        # Pass the correct container instance to identify_copy_righted_materials
                        # Ensure container is accessible (it should be if initialized correctly)
                        copyrighted_content_futures = [executor.submit(identify_copy_righted_materials, urls[index], index)
                                                      for index in range(len(urls))]
                    # Create low-res images only for image outputs
                    low_res_futures = [executor.submit(upload_low_res_image, i, img, output_data.extension)
                                       for i, img in enumerate(output_data.result)]

                # Process results
                for future in concurrent.futures.as_completed(low_res_futures):
                    index, url = future.result()
                    low_res_urls[index] = url

                if parameters_model.check_copyright_content:
                     for future in concurrent.futures.as_completed(copyrighted_content_futures):
                         is_copyrighted, index = future.result()
                         has_copyright_content[index] = is_copyrighted

            # Filter results based on copyright check
            final_urls = [url for url, is_copyrighted in zip(urls, has_copyright_content) if not is_copyrighted]
            final_low_res_urls = [url for url, is_copyrighted in zip(low_res_urls, has_copyright_content) if not is_copyrighted]

            # Update output_data only if it exists
            if output_data:
                 output_data.result = final_urls
                 output_data.low_res_urls = final_low_res_urls
                 # Optionally add copyright info if needed in the response schema
                 # output_data.Has_copyrighted_Content = [c for c, is_c in zip(has_copyright_content, has_copyright_content) if not is_c] # Filtered list

            if not final_urls: # Check if all results were filtered out
                logger.warning("All results filtered out due to copyright/NSFW checks.")
                # Decide how to handle this - maybe raise an error or return empty success?
                # For now, let it proceed, APITaskResponse output will be empty

        # --- Calculate Timing ---
        time_required = time.time() - st
        if output_data and output_data.time: # Check if output_data and time exist
            if output_data.time.runtime < time_required < (output_data.time.runtime + output_data.time.startup_time):
                 output_data.time.startup_time = 0 # Adjust startup time if needed
            # Ensure runtime calculation is robust
            output_data.time.runtime = max(0, time_required - output_data.time.startup_time)
        else:
            # Handle case where output_data or output_data.time is None
            logger.warning("Could not calculate detailed timing, output_data or output_data.time missing.")
            # Assign total time to runtime if time object doesn't exist
            if output_data and not output_data.time:
                 output_data.time = TimeData(startup_time=0, runtime=time_required)


    except Exception as e:
        # Catch any exception during the process
        logger.exception(f"Task execution failed for app_id {parameters.get('app_id', 'N/A')}: {e}")
        task_status = "FAILED"
        # Try to get specific error details if available
        if hasattr(e, 'args') and len(e.args) >= 2:
             error_type = str(e.args[0])
             error_details = str(e.args[1])
        else:
             error_type = type(e).__name__
             error_details = str(e)

    # --- Construct Final Response ---
    final_output_dict = {}
    time_required_dict = {}

    if task_status == "SUCCESS" and output_data:
        final_output_dict = {
            "result" : output_data.result,
            "Has_NSFW_Content" : output_data.Has_NSFW_Content,
            "low_res_urls" : output_data.low_res_urls
            # Add "Has_copyrighted_Content" if needed and present in output_data
            # "Has_copyrighted_Content" : getattr(output_data, 'Has_copyrighted_Content', [])
        }
        if output_data.time:
            time_required_dict = output_data.time.model_dump()

    task_response = APITaskResponse(
         status=task_status,
         input = parameters, # Log original input dict
         output = final_output_dict if task_status == "SUCCESS" else {},
         time_required = time_required_dict if task_status == "SUCCESS" else {},
         error = error_type if task_status == "FAILED" else None,
         error_data = error_details if task_status == "FAILED" else None
    )

    # Return the final validated model dump
    return task_response.model_dump()

def task_gen(parameters: APIInput) -> dict:
    """
    Determines whether to execute a task via Celery or directly.
    
    This function acts as a router that checks if a task should be processed
    asynchronously via Celery (returning a task ID for later status checking),
    or synchronously (returning the complete result immediately).
    
    The decision is based on the 'celery' parameter in the input. This provides
    flexibility in how tasks are executed based on client requirements.
    
    Args:
        parameters (APIInput): Task parameters including a 'celery' flag
        
    Returns:
        dict: Either a task ID and status (async) or complete task results (sync)
    """
    if parameters.celery == True:
        task: AsyncResult = create_task.delay(parameters.model_dump())

        task_response = APITaskResponse(
            task_id=task.id, input=parameters.model_dump(), status=task.status
        ).model_dump()

        return task_response
    else:
        return create_task(parameters.model_dump())
    
def get_task_status(id : str) -> AsyncResult:
    """
    Retrieves the current status of a Celery task.
    
    This function allows clients to check on the progress of asynchronous tasks
    that were previously submitted through task_gen with celery=True.
    
    Args:
        id (str): The Celery task ID to check
        
    Returns:
        AsyncResult: Celery task result object containing status information
    """
    task_update = celery.AsyncResult(id)
    return task_update

