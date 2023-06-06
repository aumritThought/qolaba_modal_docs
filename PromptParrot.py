from modal import Image, Secret, Stub, web_endpoint, method

from fastapi import Depends, HTTPException, UploadFile, status, Request, Query
from typing import  Optional 
import io
from fastapi.responses import StreamingResponse
from starlette.status import HTTP_403_FORBIDDEN
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

stub = Stub("promptparrot")

image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        ["torch==1.12.1+cu113", "torchvision==0.13.1+cu113", "torchaudio==0.12.1","transformers==4.21.1"],
        find_links="https://download.pytorch.org/whl/torch_stable.html"
    )
    .apt_install(["wget"])
    .run_commands([
                    "wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1loDPyYZlNAuq0gbhCQOK-XJ8CvTo_np4' -O config.json",
                    "wget --load-cookies /tmp/cookies.txt \"https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1iFFFdUBbGu9WJGNHvvoHEU2f1652DYB9' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id=1iFFFdUBbGu9WJGNHvvoHEU2f1652DYB9\" -O pytorch_model.bin && rm -rf /tmp/cookies.txt!mkdir model",
                    "mkdir model",
                    "mv config.json model/",
                    "mv pytorch_model.bin model/",
                   ])
)
stub.image = image


@stub.cls(gpu="t4",container_idle_timeout=1200)
class Predictor():
    def __enter__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        start_token = "<BOP>"
        pad_token = "<PAD>"
        end_token = "<EOP>"
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = AutoModelForCausalLM.from_pretrained("/model").to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "distilgpt2", cache_dir="./model", bos_token=start_token, eos_token=end_token, pad_token=pad_token
        )
    
    @method()
    def predict(
        self, 
        prompt= "a beautiful painting",
        num_prompts_to_generate=5, 
        max_prompt_length=50,
        min_prompt_length=30):
    
        temperature = 1.0
        top_k = 70
        top_p = 0.9
        start_token = "<BOP>"
        pad_token = "<PAD>"
        end_token = "<EOP>"

        import itertools

        if not prompt:
            raise UserWarning("manual_prompt must be at least 1 letter")

        encoded_prompt = self.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        encoded_prompt = encoded_prompt.to(self.model.device)

        output_sequences = self.model.generate(
            input_ids=encoded_prompt,
            max_length=max_prompt_length,
            min_length=min_prompt_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=num_prompts_to_generate,
            pad_token_id=self.tokenizer.pad_token_id,  # gets rid of warning
        )

        tokenized_start_token = self.tokenizer.encode(start_token)

        generated_prompts = []
        for generated_sequence in output_sequences:
            # precision is a virtue
            tokens = []
            for i, s in enumerate(generated_sequence):
                if s in tokenized_start_token and i != 0:
                    if len(tokens) >= min_prompt_length:
                        break
                tokens.append(s)

            text = self.tokenizer.decode(
                tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True
            )
            text = (
                text.strip().replace("\n", " ").replace("/", ",")
            )  # / remove slash. It causes problems in namings
            # remove repeated adjacent words from `text`. For example: "lamma lamma is cool cool" -> "lamma is cool"
            text = " ".join([k for k, g in itertools.groupby(text.split())])
            generated_prompts.append(text)

        return generated_prompts


auth_scheme = HTTPBearer()

@stub.function(gpu="a10g",secret=Secret.from_name("API_UPSCALING_KEY"))
@web_endpoint(label="promptparrot", method="POST")
def image_upscale(
        prompt: str = "a beautiful painting",
        num_prompts_to_generate: int= Query(default=5,ge=1, le=10), 
        max_prompt_length: int= Query(default=50,ge=40, le=100),
        min_prompt_length: int= Query(default=30,ge=10, le=40),
    api_key: HTTPAuthorizationCredentials = Depends(auth_scheme)):

    import os 

    if api_key.credentials != os.environ["API_UPSCALING_KEY"]:
        raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    else:
        prompts = Predictor().predict.call(prompt, num_prompts_to_generate, max_prompt_length, min_prompt_length)

        return {"generated-prompts":prompts}
        




