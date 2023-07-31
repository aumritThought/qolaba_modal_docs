from pathlib import Path

from modal import Image, Secret, Stub, web_endpoint, method, gpu

from fastapi import  Depends, HTTPException, status, Query, UploadFile
from typing import  Optional
from starlette.status import HTTP_403_FORBIDDEN
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

auth_scheme = HTTPBearer()

stub = Stub("Versatile-Diffusion_image2image")

def download_models():
    class vd_inference(object):
        
        def __init__(self, fp16=False, which='v1.0'):
            
            import os
            import sys
            sys.path.append("/Versatile-Diffusion")
            os.chdir('/Versatile-Diffusion')
            import torch
            from lib.cfg_helper import model_cfg_bank
            from lib.model_zoo import get_model
            from lib.model_zoo.ddim import DDIMSampler
            
            self.which = which

            if self.which == 'v1.0':
                cfgm = model_cfg_bank()('vd_four_flow_v1-0')
            else:
                assert False, 'Model type not supported'

            net = get_model()(cfgm)

            if fp16:
                if self.which == 'v1.0':
                    net.ctx['text'].fp16 = True
                    net.ctx['image'].fp16 = True
                net = net.half()
                self.dtype = torch.float16
            else:
                self.dtype = torch.float32
            if self.which == 'v1.0':
                if fp16:
                    sd = torch.load('pretrained/vd-four-flow-v1-0-fp16.pth', map_location='cpu')
                else:
                    sd = torch.load('pretrained/vd-four-flow-v1-0.pth', map_location='cpu')
            net.load_state_dict(sd, strict=False)

            self.use_cuda = torch.cuda.is_available()
            if self.use_cuda:
                net.to('cuda')
            self.net = net
            self.sampler = DDIMSampler(net)

    a=vd_inference()

def download_models1():
    from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
    from transformers import CLIPFeatureExtractor
    safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
    feature_extractor = CLIPFeatureExtractor()


image = (
    Image.debian_slim(python_version="3.8")
    .pip_install(
        ["torch==1.12.1+cu113", "torchvision==0.13.1+cu113", "torchaudio==0.12.1"],
        find_links="https://download.pytorch.org/whl/torch_stable.html")
    .pip_install(["tensorflow==2.2.0", "protobuf==3.11.3", "numpy==1.23.5", "matplotlib==3.4.2", "pyyaml==5.4.1", "easydict==1.9", "tensorboardx==2.1", "lpips==0.1.3", "fsspec==2022.7.1", "tqdm==4.60.0", "transformers==4.24.0", "torchmetrics==0.7.3", "einops==0.3.0", "omegaconf==2.1.1", "open_clip_torch==2.0.2", "webdataset==0.2.5", "huggingface-hub==0.11.1", "gradio==3.17.1"])
    .apt_install(["git","curl"])
    .run_commands([
                   "curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash",
                   "apt-get install git-lfs",
                   "git clone https://huggingface.co/shi-labs/versatile-diffusion",
                   "git clone https://github.com/qolaba/Versatile-Diffusion.git",
                   "mkdir Versatile-Diffusion/pretrained",
                   "mv versatile-diffusion/pretrained_pth/* Versatile-Diffusion/pretrained/",
                   ])
    .pip_install("diffusers")
    .pip_install("git+https://github.com/huggingface/transformers")

).run_function(
        download_models,
        gpu="a10g"
    ).run_function(
        download_models1,
    )


stub.image = image

@stub.cls(gpu="a100",container_idle_timeout=60, memory=10240)
class stableDiffusion(object):
    
    class adjust_rank(object):
        
        def __init__(self, max_drop_rank=[1, 5], q=20):
            import numpy as np
            import numpy.random as npr
            import torch
            self.max_semantic_drop_rank = max_drop_rank[0]
            self.max_style_drop_rank = max_drop_rank[1]
            self.q = q

            t0, y00 = np.exp((0  -0.5)*2), -self.max_semantic_drop_rank
            t1, y01 = np.exp((0.5-0.5)*2), 1
            self.t2y0_semf = lambda t: (np.exp((t-0.5)*2)-t0)/(t1-t0)*(y01-y00)+y00

 
            x0 = 0
            x1, y1 = self.max_semantic_drop_rank+1, 1
            self.x2y_semf = lambda x, y0: (x-x0)/(x1-x0)*(y1-y0)+y0
            

            t0, y00 = np.exp((1  -0.5)*2), -(q-self.max_style_drop_rank)
            t1, y01 = np.exp((0.5-0.5)*2), 1
            self.t2y0_styf = lambda t: (np.exp((t-0.5)*2)-t0)/(t1-t0)*(y01-y00)+y00

            x0 = q-1
            x1, y1 = self.max_style_drop_rank-1, 1
            self.x2y_styf = lambda x, y0: (x-x0)/(x1-x0)*(y1-y0)+y0

        def __call__(self, x, lvl):
            import torch 
            
            if lvl == 0.5:
                return x

            if x.dtype == torch.float16:
                fp16 = True
                x = x.float()
            else:
                fp16 = False
            std_save = x.std(axis=[-2, -1])

            u, s, v, x_mean, x_remain = stableDiffusion.decompose(x, q=self.q)

            if lvl < 0.5:
                assert lvl>=0
                for xi in range(0, self.max_semantic_drop_rank+1):
                    y0 = self.t2y0_semf(lvl)
                    print(self.x2y_semf)
                    yi = self.x2y_semf(xi, y0)
                    yi = 0 if yi<0 else yi
                    s[:, xi] *= yi

            elif lvl > 0.5:
                assert lvl <= 1
                for xi in range(self.max_style_drop_rank, self.q):
                    y0 = self.t2y0_styf(lvl)
                    yi = self.x2y_styf(xi, y0)
                    yi = 0 if yi<0 else yi
                    s[:, xi] *= yi
                x_remain = 0

            ss = torch.stack([torch.diag(si) for si in s])
            x_lowrank = torch.bmm(torch.bmm(u, ss), torch.permute(v, [0, 2, 1]))
            x_new = x_lowrank + x_mean + x_remain

            std_new = x_new.std(axis=[-2, -1])
            x_new = x_new / std_new * std_save

            if fp16:
                x_new = x_new.half()

            return x_new
    
    def __enter__(self, fp16=False, which='v1.0'):
        
        import os
        import sys
        sys.path.append("/Versatile-Diffusion")
        os.chdir('/Versatile-Diffusion')
        import torch
        from lib.cfg_helper import model_cfg_bank
        from lib.model_zoo import get_model
        from lib.model_zoo.ddim import DDIMSampler
        from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
        from transformers import CLIPFeatureExtractor
        
        self.which = which

        if self.which == 'v1.0':
            cfgm = model_cfg_bank()('vd_four_flow_v1-0')
        else:
            assert False, 'Model type not supported'

        net = get_model()(cfgm)

        if fp16:
            if self.which == 'v1.0':
                net.ctx['text'].fp16 = True
                net.ctx['image'].fp16 = True
            net = net.half()
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        if self.which == 'v1.0':
            if fp16:
                sd = torch.load('pretrained/vd-four-flow-v1-0-fp16.pth', map_location='cpu')
            else:
                sd = torch.load('pretrained/vd-four-flow-v1-0.pth', map_location='cpu')
        net.load_state_dict(sd, strict=False)

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            net.to('cuda')
        self.net = net
        self.sampler = DDIMSampler(net)

        self.output_dim = [512, 512]
        self.n_sample_image = 1
        self.ddim_steps = 50
        self.ddim_eta = 0.0
        self.image_latent_dim = 4
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker").to("cuda")
        self.feature_extractor = CLIPFeatureExtractor()
        if which == 'v1.0':
            self.adjust_rank_f = stableDiffusion.adjust_rank(max_drop_rank=[1, 5], q=20)
            self.scale_imgto = 7.5
            self.disentanglement_noglobal = True
    
    
    def decompose(x, q=20, niter=100):
        import torch
        
        x_mean = x.mean(-1, keepdim=True)
        x_input = x - x_mean
        u, s, v = torch.pca_lowrank(x_input, q=q, center=False, niter=niter)
        ss = torch.stack([torch.diag(si) for si in s])
        x_lowrank = torch.bmm(torch.bmm(u, ss), torch.permute(v, [0, 2, 1]))
        x_remain = x_input - x_lowrank
        return u, s, v, x_mean, x_remain
    
    @method()
    def run_inference(self, im, fid_lvl=0.1, fcs_lvl=0.1,  batch=1, num_inference_steps=50):
        import torchvision.transforms as tvtrans
        import torch 
        import numpy as np
        from PIL import Image

        n_samples = batch
        scale = self.scale_imgto
        sampler = self.sampler
        clr_adj='Simple'
        seed=1
        device = self.net.device
        self.ddim_steps=num_inference_steps
        w, h = [768,int(im.size[1]*768/im.size[0])]

        
        im = im.resize([w, 64 * round(h / 64)])
        
        if fid_lvl == 1:
            return {"images":[im]*n_samples,  "Has_NSFW_Content":[False]*n_samples}
            return 

        cx = tvtrans.ToTensor()(im)[None].to(device).to(self.dtype)
        c = self.net.ctx_encode(cx, which='image')
        if self.disentanglement_noglobal:
            c_glb = c[:, 0:1]
            c_loc = c[:, 1: ]
            c_loc = self.adjust_rank_f(c_loc, fcs_lvl)
            c = torch.cat([c_glb, c_loc], dim=1).repeat(n_samples, 1, 1)
        else:
            c = self.adjust_rank_f(c, fcs_lvl).repeat(n_samples, 1, 1)
        u = torch.zeros_like(c)

        shape = [n_samples, self.image_latent_dim, h//8, w//8]
        np.random.seed(seed)
        torch.manual_seed(seed + 100)
        if fid_lvl!=0:
            x0 = self.net.vae_encode(cx, which='image').repeat(n_samples, 1, 1, 1)
            step = int(self.ddim_steps * (1-fid_lvl))
            x, _ = sampler.sample(
                steps=self.ddim_steps,
                x_info={'type':'image', 'x0':x0, 'x0_forward_timesteps':step},
                c_info={'type':'image', 'conditioning':c, 'unconditional_conditioning':u, 
                        'unconditional_guidance_scale':scale},
                shape=shape,
                verbose=False,
                eta=self.ddim_eta)
        else:
            x, _ = sampler.sample(
                steps=self.ddim_steps,
                x_info={'type':'image',},
                c_info={'type':'image', 'conditioning':c, 'unconditional_conditioning':u, 
                        'unconditional_guidance_scale':scale},
                shape=shape,
                verbose=False,
                eta=self.ddim_eta)

        imout = self.net.vae_decode(x, which='image')

        if clr_adj == 'Simple':
            cx_mean = cx.view(3, -1).mean(-1)[:, None, None]
            cx_std  = cx.view(3, -1).std(-1)[:, None, None]
            imout_mean = [imouti.view(3, -1).mean(-1)[:, None, None] for imouti in imout]
            imout_std  = [imouti.view(3, -1).std(-1)[:, None, None] for imouti in imout]
            imout = [(ii-mi)/si*cx_std+cx_mean for ii, mi, si in zip(imout, imout_mean, imout_std)]
            imout = [torch.clamp(ii, 0, 1) for ii in imout]

        image = [tvtrans.ToPILImage()(i) for i in imout]
        safety_checker_input = self.feature_extractor(
                image, return_tensors="pt"
            ).to("cuda")
        image, has_nsfw_concept = self.safety_checker(
                        images=[np.array(i) for i in image], clip_input=safety_checker_input.pixel_values
                    )
        image=[ Image.fromarray(np.uint8(i)) for i in image] 
        return {"images":image,  "Has_NSFW_Content":has_nsfw_concept}