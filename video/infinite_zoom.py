from modal import Image, Stub, method


def download_models():
    from diffusers import StableDiffusionInpaintPipeline
    import torch
    pipe = StableDiffusionInpaintPipeline.from_ckpt(
         "https://huggingface.co/lenML/LOFI-V2_2-inpainting/blob/main/LOFI_V22.inpainting.safetensors",
        dtype=torch.float16
    )

stub = Stub("infinite_zoom_video")
image = (
    Image.debian_slim(python_version="3.10")
    .run_commands([
        "apt-get update && apt-get install ffmpeg libsm6 libxext6 git -y",
        "git clone https://github.com/v8hid/infinite-zoom-stable-diffusion.git",
        "pip install ffmpeg-python omegaconf diffusers transformers accelerate opencv-python Pillow xformers scipy ftfy imageio[ffmpeg]",
        ])
    ).run_function(
            download_models,
            gpu="t4"
        )

stub.image = image

@stub.cls(gpu="a10g", container_idle_timeout=600, memory=10240)
class stableDiffusion:  
    def __enter__(self):

        from diffusers import StableDiffusionInpaintPipeline, EulerAncestralDiscreteScheduler
        import torch

        self.pipe = StableDiffusionInpaintPipeline.from_single_file(
            "https://huggingface.co/lenML/LOFI-V2_2-inpainting/blob/main/LOFI_V22.inpainting.safetensors",
            dtype=torch.float16
        )
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config)
        self.pipe = self.pipe.to("cuda")

        self.pipe.safety_checker = None
        self.pipe.enable_attention_slicing()
        self.pipe.enable_xformers_memory_efficient_attention()
    
    

    

    @method()
    def run_inference(self,file_url, List_of_prompt,zoom,Num_of_sec,height,width,guidance_scale, negative_prompt, frame_numbers, fps):

        from PIL import Image
        import numpy as np
        import base64, time, requests, os
        import numpy as np
        import imageio
        from PIL import Image

        def shrink_and_paste_on_blank(current_image, mask_width):

            height = current_image.height
            width = current_image.width

            #shrink down by mask_width
            prev_image = current_image.resize((height-2*mask_width,width-2*mask_width))
            prev_image = prev_image.convert("RGBA")
            prev_image = np.array(prev_image)

            #create blank non-transparent image
            blank_image = np.array(current_image.convert("RGBA"))*0
            blank_image[:,:,3] = 1

            #paste shrinked onto blank
            blank_image[mask_width:height-mask_width,mask_width:width-mask_width,:] = prev_image
            prev_image = Image.fromarray(blank_image)

            return prev_image
    
        def write_video(file_path, frames, fps, reversed=True, start_frame_dupe_amount=15, last_frame_dupe_amount=30):
            
            """
            Writes frames to an mp4 video file
            :param file_path: Path to output video, must end with .mp4
            :param frames: List of PIL.Image objects
            :param fps: Desired frame rate
            :param reversed: if order of images to be reversed (default = True)
            """
            if reversed == True:
                frames = frames[::-1]

            # Get dimensions of the frames
            w, h = frames[0].size

            # Create an imageio video writer
            writer = imageio.get_writer(file_path, fps=fps)

            # Duplicate the start and end frames
            start_frames = [frames[0]] * start_frame_dupe_amount
            end_frames = [frames[-1]] * last_frame_dupe_amount

            # Write the duplicated frames to the video writer
            for frame in start_frames:
                # Convert PIL image to numpy array
                np_frame = np.array(frame)
                writer.append_data(np_frame)

            # Write the frames to the video writer
            for frame in frames:
                np_frame = np.array(frame)
                writer.append_data(np_frame)

            # Write the duplicated frames to the video writer
            for frame in  end_frames:
                np_frame = np.array(frame)
                writer.append_data(np_frame)

            # Close the video writer
            writer.close()

        prompts={}
        for i in range(0, len(List_of_prompt)):
            prompts[frame_numbers[i]]=List_of_prompt[i]
        

        height = height
        width = width

        current_image = Image.new(mode="RGBA", size=(height, width))
        mask_image = np.array(current_image)[:, :, 3]
        mask_image = Image.fromarray(255-mask_image).convert("RGB")
        current_image = current_image.convert("RGB")
        if (file_url):
            current_image = file_url.resize(
                (width, height), resample=Image.LANCZOS)
        else:
            init_images = self.pipe(prompt=prompts[min(k for k in prompts.keys() if k >= 0)],
                            negative_prompt=negative_prompt,
                            image=current_image,
                            guidance_scale=guidance_scale,
                            height=height,
                            width=width,
                            mask_image=mask_image,
                            num_inference_steps=30)[0]
            current_image = init_images[0]
        mask_width = 128
        num_interpol_frames = 30

        all_frames = []
        all_frames.append(current_image)

        for i in range(Num_of_sec):
            print('Outpaint step: ' + str(i+1) +
                ' / ' + str(Num_of_sec))

            prev_image_fix = current_image

            prev_image = shrink_and_paste_on_blank(current_image, mask_width)

            current_image = prev_image

            # create mask (black image with white mask_width width edges)
            mask_image = np.array(current_image)[:, :, 3]
            mask_image = Image.fromarray(255-mask_image).convert("RGB")

            # inpainting step
            current_image = current_image.convert("RGB")
            images = self.pipe(prompt=prompts[max(k for k in prompts.keys() if k <= i)],
                        negative_prompt=negative_prompt,
                        image=current_image,
                        guidance_scale=guidance_scale,
                        height=height,
                        width=width,
                        mask_image=mask_image,
                        num_inference_steps=30)[0]
            current_image = images[0]
            current_image.paste(prev_image, mask=prev_image)

            # interpolation steps bewteen 2 inpainted images (=sequential zoom and crop)
            for j in range(num_interpol_frames - 1):
                interpol_image = current_image
                interpol_width = round(
                    (1 - (1-2*mask_width/height)**(1-(j+1)/num_interpol_frames))*height/2
                )
                interpol_image = interpol_image.crop((interpol_width,
                                                    interpol_width,
                                                    width - interpol_width,
                                                    height - interpol_width))

                interpol_image = interpol_image.resize((height, width))

                # paste the higher resolution previous image in the middle to avoid drop in quality caused by zooming
                interpol_width2 = round(
                    (1 - (height-2*mask_width) / (height-2*interpol_width)) / 2*height
                )
                prev_image_fix_crop = shrink_and_paste_on_blank(
                    prev_image_fix, interpol_width2)
                interpol_image.paste(prev_image_fix_crop, mask=prev_image_fix_crop)

                all_frames.append(interpol_image)
            all_frames.append(current_image)
        video_file_name = "infinite_zoom_" + str(time.time())
        save_path = video_file_name + ".mp4"
        start_frame_dupe_amount = 15
        last_frame_dupe_amount = 15

        write_video(save_path, all_frames, fps, zoom,
                    start_frame_dupe_amount, last_frame_dupe_amount)

        with open(save_path, "rb") as f:
            url="https://qolaba-server-development-2303.up.railway.app/api/v1/uploadToCloudinary/audiovideo"
            byte=f.read()
            myobj = {"image":"data:audio/mpeg;base64,"+(base64.b64encode(byte).decode("utf8"))}
            
            rps = requests.post(url, json=myobj, headers={'Content-Type': 'application/json'})
            video_url=rps.json()["data"]["secure_url"]
        

        os.remove(save_path)

        return {"video_url":video_url}

        
