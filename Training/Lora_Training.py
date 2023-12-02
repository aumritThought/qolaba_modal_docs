from modal import Image, Stub, method

model_schema={
    "memory":30240,
    "container_idle_timeout":200,
    "gpu":"a10g",
    "concurrency_limit" : 1
    }

model_schema["name"] = "Lora_Training"

def download_models():
    import os
    import sys
    sys.path.insert(0, "/kohya_ss")
    os.chdir("/kohya_ss")
    train_data_path="/kohya_ss/zip_Data"
    os.system(f'python3 "finetune/make_captions.py" --batch_size="1" --num_beams="1" --top_p="0.9" --max_length="75" --min_length="5" --beam_search  --caption_extension=".txt" {train_data_path} --caption_weights="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth"')    
    from diffusers import StableDiffusionXLPipeline
    import torch

    pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=True
        )
    pipe.to(device="cuda", dtype=torch.float16)

    img_folder="/workspace/kohya_ss/zip_Data/x4lyyB1P/img"
    log_folder="/workspace/kohya_ss/zip_Data/x4lyyB1P/log"
    model_folder="/workspace/kohya_ss/zip_Data/x4lyyB1P/model"
    train_file_name = "Lora_Model"
    train_cmd= f'''accelerate launch --num_cpu_threads_per_process=2 "./sdxl_train_network.py" --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" --train_data_dir="{img_folder}" --resolution="1024,1024"  --output_dir="{model_folder}"  --logging_dir="{log_folder}"  --text_encoder_lr=0.0004 --unet_lr=0.0004 --learning_rate="0.0004" --output_name="{train_file_name}"  --lr_scheduler_num_cycles="10"  --train_batch_size="1"  --max_train_steps="1000"  --network_dim=256 --network_alpha="1" --enable_bucket --min_bucket_reso=256 --max_bucket_reso=2048 --network_module=networks.lora --save_model_as=safetensors --no_half_vae  --lr_scheduler="constant" --save_every_n_epochs="1" --mixed_precision="bf16" --save_precision="bf16" --caption_extension=".txt" --optimizer_type="Adafactor" --cache_latents --cache_latents_to_disk --max_data_loader_n_workers="0" --bucket_reso_steps=64 --gradient_checkpointing --xformers --bucket_no_upscale --noise_offset=0.0 --optimizer_args scale_parameter=False relative_step=False warmup_init=False'''

    os.system(train_cmd)


stub = Stub(model_schema["name"])
image = (
    Image.debian_slim(python_version="3.11")
    .run_commands([
        "apt-get update && apt-get install ffmpeg libsm6 libxext6 git -y",
        "apt-get update && apt-get install wget -y",
        "git clone https://github.com/bmaltais/kohya_ss.git",
        "pip install invisible_watermark transformers accelerate safetensors xformers==0.0.22 omegaconf einops toml voluptuous timm fairscale",
        "pip install diffusers --upgrade",
        "mkdir kohya_ss/zip_Data",
        ])
    ).run_function(
            download_models,
            gpu="t4"
        )

stub.image = image

@stub.cls(gpu="a100", memory=model_schema["memory"], container_idle_timeout=200,  timeout=7200)
class stableDiffusion:
         
    def is_image_file(self, file_path):
        from PIL import Image
        try:
            with Image.open(file_path):
                return True
        except Exception as e:
            print(e)
            return False
            
    def process_image(self, image_path):
        from PIL import Image
        import random, os

        random_color = False 
        background_colors = [
            (255, 255, 255),
            (0, 0, 0),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]

        img = Image.open(image_path)
        img_dir, image_name = os.path.split(image_path)

        if img.mode in ("RGBA", "LA"):
            if random_color:
                background_color = random.choice(background_colors)
            else:
                background_color = (255, 255, 255)
            bg = Image.new("RGB", img.size, background_color)
            bg.paste(img, mask=img.split()[-1])

            if image_name.endswith(".webp"):
                bg = bg.convert("RGB")
                new_image_path = os.path.join(img_dir, image_name.replace(".webp", ".jpg"))
                bg.save(new_image_path, "JPEG")
                os.remove(image_path)
                print(f" Converted image: {image_name} to {os.path.basename(new_image_path)}")
            else:
                bg.save(image_path, "PNG")
                print(f" Converted image: {image_name}")
        else:
            if image_name.endswith(".webp"):
                new_image_path = os.path.join(img_dir, image_name.replace(".webp", ".jpg"))
                img.save(new_image_path, "JPEG")
                os.remove(image_path)
                print(f" Converted image: {image_name} to {os.path.basename(new_image_path)}")
            else:
                img.save(image_path, "PNG")

    def download_zip_extract_and_delete(self, url, extract_folder):
        import string, shutil, zipfile, requests, random, os

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            random_folder_name = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            random_folder_path = os.path.join(extract_folder, random_folder_name)
            os.makedirs(random_folder_path, exist_ok=True)
            zip_file_path = os.path.join(random_folder_path, 'downloaded_file.zip')
            with open(zip_file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=128):
                    file.write(chunk)
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(random_folder_path)
            os.remove(zip_file_path)
            non_image_found = False
            for extracted_file in os.listdir(random_folder_path):
                file_path = os.path.join(random_folder_path, extracted_file)
                print(file_path)
                if not self.is_image_file(file_path):
                    non_image_found = True
                    break
                else:
                    self.process_image(file_path)
            if non_image_found:
                shutil.rmtree(random_folder_path)
                raise Exception("Please provide zip file which only contains Images. There should not be inside another folder in zip.")
        else:
            raise Exception(f"Failed to download zip file. Status code: {response.status_code}", "Provide proper ZIP File which contains only Images")


        return random_folder_path

    
    def organize_files(self, parent_folder):
        import os, shutil
        # Define the names of the three folders
        img_folder = os.path.join(parent_folder, 'img')
        log_folder = os.path.join(parent_folder, 'log')
        model_folder = os.path.join(parent_folder, 'model')

        for folder in [img_folder, log_folder, model_folder]:
            os.makedirs(folder, exist_ok=True)

        for filename in os.listdir(parent_folder):
            file_path = os.path.join(parent_folder, filename)

            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.txt')):
                shutil.move(file_path, os.path.join(img_folder, filename))

        return img_folder, log_folder, model_folder

    def keep_only_alphabets(self, input_string):
        import re
        result = re.sub(r'[^a-zA-Z]', '', input_string)
        return result

    import os

    def add_prefix_to_content(self,folder_path, prefix):
        import os
        files = os.listdir(folder_path)
        for file_name in files:
            if file_name.endswith(".txt"):
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'r') as file:
                    content = file.read()

                new_content = prefix + content
                with open(file_path, 'w') as file:
                    print(new_content)
                    file.write(new_content)



    def create_and_move_folder(self, source_path, repetition, token):
        import os, shutil
        new_folder_name = f"{repetition}_{token}"

        new_folder_path = os.path.join(source_path, new_folder_name)
        os.makedirs(new_folder_path, exist_ok=True)

        allowed_extensions = {'.txt', '.jpg', '.jpeg', '.png'}

        for item in os.listdir(source_path):
            item_path = os.path.join(source_path, item)
            if os.path.isfile(item_path) and os.path.splitext(item)[1].lower() in allowed_extensions:
                shutil.move(item_path, os.path.join(new_folder_path, item))


    @method()
    def run_inference(self,file_url,max_steps, Image_repetition, token_string, category, epochs):
        import os, sys, time, string, secrets, torch, shutil
        from huggingface_hub import HfApi, login
        login("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")

        sys.path.insert(0, "../kohya_ss")
        os.chdir("../kohya_ss")
        st = time.time()

        print(os.getcwd())
        extract_folder = '/kohya_ss/zip_Data'
        train_data_path=self.download_zip_extract_and_delete(file_url, extract_folder)
        print(train_data_path)
        print(os.listdir(train_data_path))

        os.system(f'python3 "finetune/make_captions.py" --batch_size="1" --num_beams="1" --top_p="0.9" --max_length="75" --min_length="5" --beam_search  --caption_extension=".txt" {train_data_path} --caption_weights="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth"')  
        print(os.listdir(train_data_path))
        self.add_prefix_to_content(train_data_path, prefix=token_string + ", ")  

        img_folder,log_folder,model_folder = self.organize_files(train_data_path)
        print(img_folder, log_folder, model_folder)

        input_string = token_string
        train_file_name = "Lora_"+self.keep_only_alphabets(input_string)
        print(train_file_name)
        token_parameter=f"{token_string} {category}"
        self.create_and_move_folder(img_folder, Image_repetition, token_parameter)
        print(os.listdir(train_data_path))
        print(os.listdir(img_folder))

        train_cmd= f'''accelerate launch --num_cpu_threads_per_process=2 "./sdxl_train_network.py" --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" --train_data_dir="{img_folder}" --resolution="1024,1024"  --output_dir="{model_folder}"  --logging_dir="{log_folder}"  --text_encoder_lr=0.0004 --unet_lr=0.0004 --learning_rate="0.0004" --output_name="{train_file_name}"  --lr_scheduler_num_cycles="{epochs}"  --train_batch_size="1"  --max_train_steps="{max_steps}"  --network_dim=256 --network_alpha="1" --enable_bucket --min_bucket_reso=256 --max_bucket_reso=2048 --network_module=networks.lora --save_model_as=safetensors --no_half_vae  --lr_scheduler="constant" --save_every_n_epochs="1" --mixed_precision="bf16" --save_precision="bf16" --caption_extension=".txt" --optimizer_type="Adafactor" --cache_latents --cache_latents_to_disk --max_data_loader_n_workers="0" --bucket_reso_steps=64 --gradient_checkpointing --xformers --bucket_no_upscale --noise_offset=0.0 --optimizer_args scale_parameter=False relative_step=False warmup_init=False'''

        os.system(train_cmd)

        characters = string.ascii_letters + string.digits

        random_string = ''.join(secrets.choice(characters) for _ in range(8))

        path_or_fileobj = f"{model_folder}/{train_file_name}.safetensors"
        api = HfApi()
        api.upload_file(
            path_or_fileobj=path_or_fileobj,
            path_in_repo=f"{train_file_name}_{random_string}.safetensors",
            repo_id="Qolaba/Lora_Models",
            repo_type="model",
        )
        print(torch.cuda.max_memory_allocated())
        try:
        # Use shutil.rmtree() to delete the folder and its contents
            shutil.rmtree(train_data_path)
            print(f"Folder '{train_data_path}' and its contents deleted successfully.")
        except Exception as e:
            print(f"Error: {e}")

        self.runtime = time.time()-st
        return {"result":[f"{train_file_name}_{random_string}.safetensors", token_string],  
                "Has_NSFW_Content":[False], 
                "time": {"startup_time" : 0, "runtime":self.runtime}}