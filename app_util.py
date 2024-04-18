import os
import argparse
import PIL
import torch
import cv2
import time
import shutil

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image
from itertools import islice
from einops import rearrange, repeat
from torch import autocast
from pytorch_lightning import seed_everything
import gradio as gr

from ldm.util import instantiate_from_config, load_model_from_config, load_img, load_model_and_get_prompt_embedding
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from attention_control.masactrl_utils import regiter_attention_editor_ldm
from attention_control.share_attention import ShareSelfAttentionControl
from torchvision.utils import save_image

def pil_load_img(image, SCALE, pad=False, seg=False, target_size=None):
    w, h = image.size      
    w_,h_=w,h  
    print(f"loaded input image of size ({w}, {h})")
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    w = h = 512
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
        
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    print(f"resize input image of size ({w_}, {h_}) to {w}, {h}")
    
    return 2. * image - 1., w, h 

def tfgph_load(opt, device):
    # Load Model
    config = OmegaConf.load(opt["config"])
    model = load_model_from_config(config, opt["ckpt"])    

    model = model.to(device)
    sampler = DPMSolverSampler(model)

    print("##----Model LOAD Success---##")
    return model,sampler

def tfgph_inverse(ref_img,opt,model,sampler,device,ref1_path=None,ref2_path=None,comp_path=None):
    # Read Image
    ref_image, target_width, target_height = pil_load_img(ref_img, 1)
    ref_image = repeat(ref_image.to(device), '1 ... -> b ...', b=1)
    print("##----Image LOAD Success---##")

    # Reconstruct
    uncond_scale=2.5
    precision_scope = autocast
    with precision_scope("cuda"):
        c, uc, inv_emb = load_model_and_get_prompt_embedding(model, uncond_scale, device, opt["prompt"], inv=True)

        T1 = time.time()
        ref_latent = model.get_first_stage_encoding(model.encode_first_stage(ref_image))
        shape = ref_latent.shape[1:]
        z_ref, _ = sampler.sample(steps=opt["total_steps"],
                                inv_emb=inv_emb,
                                unconditional_conditioning=uc,
                                conditioning=c,
                                batch_size=1,
                                shape=shape,
                                verbose=False,
                                unconditional_guidance_scale=uncond_scale,
                                eta=0,
                                order=opt["order"],
                                x_T=ref_latent,
                                width=512,
                                height=512,
                                DPMencode=True,
                                )
    return z_ref
def tfgph_harmonize(z_ref1, z_ref2, z_comp, opt,model,sampler,device,ref1_path=None,ref2_path=None,comp_path=None):
    precision_scope = autocast
    uncond_scale=2.5
    c, uc, inv_emb = load_model_and_get_prompt_embedding(model, uncond_scale, device, opt["prompt"], inv=True)
    sim_scales = torch.tensor([opt["scale_alpha"],opt["scale_beta"]]).to(device)
    shape=z_ref1.shape[1:]
    with precision_scope("cuda"):
        
        # hijack the attention module (sclaed share)
        editor = ShareSelfAttentionControl(opt["share_step"], opt["share_layer"],scales=sim_scales,total_steps=opt["total_steps"])
        regiter_attention_editor_ldm(model, editor)
        latents_harmonized = sampler.sample(steps=opt["total_steps"],
                                    inv_emb=torch.cat([inv_emb,inv_emb,inv_emb]),
                                    conditioning=torch.cat([c,c,c]),
                                    shape=shape,
                                    verbose=False,
                                    unconditional_guidance_scale=uncond_scale,
                                    unconditional_conditioning=torch.cat([uc,uc,uc]),
                                    eta=0,
                                    order=opt["order"],
                                    x_T=torch.cat([z_ref1,z_ref2,z_comp]),
                                    width=512,
                                    height=512,
                                    )
        x_harmonized = model.decode_first_stage(latents_harmonized)
        x_harmonized = torch.clamp((x_harmonized + 1.0) / 2.0, min=0.0, max=1.0)


    x_sample=x_harmonized[-1]
    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
    img = Image.fromarray(x_sample.astype(np.uint8))

    return img
