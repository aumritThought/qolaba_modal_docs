from modal import Image, Stub , method

stub = Stub("promptparrot_text2text")

def download_models():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    start_token = "<BOP>"
    pad_token = "<PAD>"
    end_token = "<EOP>"
    model = AutoModelForCausalLM.from_pretrained("/model").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
            "distilgpt2", cache_dir="./model", bos_token=start_token, eos_token=end_token, pad_token=pad_token
        )


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
).run_function(
        download_models,
        gpu="t4",
    )
stub.image = image


@stub.cls(gpu="t4",container_idle_timeout=100)
class stableDiffusion():
    def __enter__(self):
        import time
        st= time.time()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        start_token = "<BOP>"
        pad_token = "<PAD>"
        end_token = "<EOP>"
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = AutoModelForCausalLM.from_pretrained("/model").to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "distilgpt2", cache_dir="./model", bos_token=start_token, eos_token=end_token, pad_token=pad_token
        )
        self.container_execution_time=time.time()-st
    
    @method()
    def run_inference(
        self, 
        prompt= "a beautiful painting",
        batch=5, 
        max_prompt_length=50,
        min_prompt_length=30):
        import time
        st=time.time()
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
            num_return_sequences=batch,
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
            ) 
            text = " ".join([k for k, g in itertools.groupby(text.split())])
            generated_prompts.append(text)
        self.runtime=time.time()-st
        return {"result":generated_prompts,  
                "Has_NSFW_Content":  [False]*1, 
                "time": {"startup_time" : self.container_execution_time, "runtime":self.runtime}}

