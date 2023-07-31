from modal import Image, Secret, Stub, web_endpoint, method

from fastapi import Depends, HTTPException, UploadFile, status, Request, Query
from typing import  Optional 
import io
from fastapi.responses import StreamingResponse
from starlette.status import HTTP_403_FORBIDDEN
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

stub = Stub("prompt_extend_text2text")
def download_models():
    from transformers import pipeline

    generator = pipeline("text-generation", model="daspartho/prompt-extend")

image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        ["torch==1.12.1+cu113", "torchvision==0.13.1+cu113", "torchaudio==0.12.1","transformers==4.21.1"],
        find_links="https://download.pytorch.org/whl/torch_stable.html"
    )
    .pip_install(["diffusers", "transformers", "accelerate"])
).run_function(
    download_models
)
stub.image = image


@stub.cls(gpu="t4",container_idle_timeout=1200)
class stableDiffusion():
    def __enter__(self):
        from transformers import pipeline

        self.generator = pipeline('text-generation', model='daspartho/prompt-extend',  device=0)

    
    @method()
    def run_inference(
        self, 
        prompt= "a beautiful painting",
        num_prompts_to_generate=5, 
        max_prompt_length=80,
        min_prompt_length=30):
        import itertools
    
        output_sequences=self.generator(prompt, max_length=max_prompt_length, num_return_sequences=num_prompts_to_generate, min_length=min_prompt_length)

        generated_prompts = []
        for generated_sequence in output_sequences:
            # precision is a virtue

            text = generated_sequence["generated_text"]
            text = (
                text.strip().replace("\n", " ").replace("/", ",")
            )  # / remove slash. It causes problems in namings
            # remove repeated adjacent words from `text`. For example: "lamma lamma is cool cool" -> "lamma is cool"
            text = " ".join([k for k, g in itertools.groupby(text.split())])
            generated_prompts.append(text)

        return generated_prompts
