from src.data_models.ImageToImage import Inference

def image_to_image_inference(image_urls, nsfw, time, runtime):
        inference = Inference()
        inference.result = image_urls
        inference.has_nsfw_content = nsfw
        inference.time.startup_time = time
        inference.time.runtime = runtime
        return inference