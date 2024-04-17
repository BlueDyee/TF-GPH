
import os
import argparse
import PIL
import torch
import cv2
import time

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from itertools import islice
from einops import rearrange, repeat
from torch import autocast
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config, load_model_from_config, load_img, load_model_and_get_prompt_embedding
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from attention_control.masactrl_utils import regiter_attention_editor_ldm
from attention_control.share_attention import ShareSelfAttentionControl
from torchvision.utils import save_image

def tfphd_main(opt):
    # Load Model
    config = OmegaConf.load(opt.config)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Running on {device}")
    model = load_model_from_config(config, opt.ckpt)    

    model = model.to(device)
    sampler = DPMSolverSampler(model)

    print("##----Model LOAD Success---##")

    # Read Image
    ref_1_image, target_width, target_height = load_img(opt.ref1,1)
    ref_1_image = repeat(ref_1_image.to(device), '1 ... -> b ...', b=1)

    ref_2_image, width, height= load_img(opt.ref2, 1)
    ref_2_image = repeat(ref_2_image.to(device), '1 ... -> b ...', b=1)

    composite_image, width, height= load_img(opt.comp, 1)
    composite_image = repeat(composite_image.to(device), '1 ... -> b ...', b=1)

    print("##----Image LOAD Success---##")

    # Reconstruct
    uncond_scale=2.5
    precision_scope = autocast
    with precision_scope("cuda"):
        c, uc, inv_emb = load_model_and_get_prompt_embedding(model, uncond_scale, device, opt.prompt, inv=True)
        print("Condition shape", c.shape,uc.shape,inv_emb.shape)

        T1 = time.time()
        ref_1_latent = model.get_first_stage_encoding(model.encode_first_stage(ref_1_image))
        ref_2_latent = model.get_first_stage_encoding(model.encode_first_stage(ref_2_image))
        composite_latent = model.get_first_stage_encoding(model.encode_first_stage(composite_image))
        shape = ref_1_latent.shape[1:]
        z_ref_1_enc, _ = sampler.sample(steps=opt.total_steps,
                                inv_emb=inv_emb,
                                unconditional_conditioning=uc,
                                conditioning=c,
                                batch_size=1,
                                shape=shape,
                                verbose=False,
                                unconditional_guidance_scale=uncond_scale,
                                eta=0,
                                order=opt.order,
                                x_T=ref_1_latent,
                                width=width,
                                height=height,
                                DPMencode=True,
                                )
        
        z_ref_2_enc, _ = sampler.sample(steps=opt.total_steps,
                                    inv_emb=inv_emb,
                                    unconditional_conditioning=uc,
                                    conditioning=c,
                                    batch_size=1,
                                    shape=shape,
                                    verbose=False,
                                    unconditional_guidance_scale=uncond_scale,
                                    eta=0,
                                    order=opt.order,
                                    x_T=ref_2_latent,
                                    DPMencode=True,
                                    width=width,
                                    height=height,
                                    ref=True,
                                    )
        z_composite_enc, _ = sampler.sample(steps=opt.total_steps,
                                    inv_emb=inv_emb,
                                    unconditional_conditioning=uc,
                                    conditioning=c,
                                    batch_size=1,
                                    shape=shape,
                                    verbose=False,
                                    unconditional_guidance_scale=uncond_scale,
                                    eta=0,
                                    order=opt.order,
                                    x_T=composite_latent,
                                    DPMencode=True,
                                    width=width,
                                    height=height,
                                    ref=True,
                                    )
                    
        samples = sampler.sample(steps=opt.total_steps,
                                    inv_emb=torch.cat([inv_emb,inv_emb,inv_emb]),
                                    conditioning=torch.cat([c,c,c]),
                                    shape=shape,
                                    verbose=False,
                                    unconditional_guidance_scale=uncond_scale,
                                    unconditional_conditioning=torch.cat([uc,uc,uc]),
                                    eta=0,
                                    order=opt.order,
                                    x_T=torch.cat([z_ref_1_enc,z_ref_2_enc,z_composite_enc]),
                                    width=width,
                                    height=height,
                                    )
            
        x_reconstruct = model.decode_first_stage(samples)
        x_reconstruct = torch.clamp((x_reconstruct + 1.0) / 2.0, min=0.0, max=1.0)
        
        T2 = time.time()
        print('Running Time: %s s' % ((T2 - T1)))
    names=["ref_1_recon.png","ref_2_recon.png","composite_recon.png"]
    exp_count=len(os.listdir(opt.outdir))
    exp_path=os.path.join(opt.outdir,f"exp_{exp_count:04d}")
    os.mkdir(exp_path)
    reconstruct_path=os.path.join(exp_path,"reconstruct")
    os.mkdir(reconstruct_path)
    for x_sample,sample_name in zip(x_reconstruct,names):
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        img = Image.fromarray(x_sample.astype(np.uint8))
        img.save(os.path.join(reconstruct_path, sample_name))
    print(f"Reconstruct result saving at {reconstruct_path}")

    print("##----Reconstuct Success---##")


    sim_scales = torch.tensor([opt.scale_alpha,opt.scale_beta]).to(device) 

    with precision_scope("cuda"):
        # hijack the attention module (sclaed share)
        editor = ShareSelfAttentionControl(opt.share_step, opt.share_layer,scales=sim_scales,total_steps=opt.total_steps)
        regiter_attention_editor_ldm(model, editor)
        latents_harmonized = sampler.sample(steps=opt.total_steps,
                                    inv_emb=torch.cat([inv_emb,inv_emb,inv_emb]),
                                    conditioning=torch.cat([c,c,c]),
                                    shape=shape,
                                    verbose=False,
                                    unconditional_guidance_scale=uncond_scale,
                                    unconditional_conditioning=torch.cat([uc,uc,uc]),
                                    eta=0,
                                    order=opt.order,
                                    x_T=torch.cat([z_ref_1_enc,z_ref_2_enc,z_composite_enc]),
                                    width=width,
                                    height=height,
                                    )
        x_harmonized = model.decode_first_stage(latents_harmonized)
        x_harmonized = torch.clamp((x_harmonized + 1.0) / 2.0, min=0.0, max=1.0)
    names=["ref_1_harmonized.png","ref_2_harmonized.png","composite_harmonized.png"]
    
    harmonized_path=os.path.join(exp_path, "harmonized")
    os.mkdir(harmonized_path)
    for x_sample,sample_name in zip(x_harmonized,names):
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        img = Image.fromarray(x_sample.astype(np.uint8))
        img.save(os.path.join(harmonized_path, sample_name))

    composite_name=opt.comp.split("/")[-1][:-4]
    file_name=os.path.join(exp_path, f"{composite_name}_{opt.share_step}S_{opt.share_layer}L_total.png")
    save_image(torch.cat([x_reconstruct,x_harmonized]), file_name,nrow=3)
    print(f"Harmonized Result saved in {harmonized_path}")
    print("Success !!! Enjoy your results ~")



if __name__ == "__main__":
    # TFPHD Hyper Parameter
    parser = argparse.ArgumentParser(description='Hyper-parameter of TFPHD')
    parser.add_argument('--ref1', type=str, default="./inputs/demo_input/kangaroo.jpg", help='The path of FIRST reference image (Foreground)')
    parser.add_argument('--ref2', type=str, default="./inputs/demo_input/starry_night.jpg", help='The path of SECOND reference image (Background)')
    parser.add_argument('--comp', type=str, default="./inputs/demo_input/kangaroo_starry.jpg", help='The path of COMPOSITE reference image (Copy-and-Paste)')
    parser.add_argument('--order', type=int, default=2, help='order=2-->DPM++, order=1-->DDIM ')
    parser.add_argument('--total_steps', type=int, default=20, help='Total Steps of DPM++ or DDIM ')
    parser.add_argument('--share_step', type=int, default=15, help='Which STEP to start share attention module')
    parser.add_argument('--share_layer', type=int, default=12, help='Which LAYER to start share attention module')
    parser.add_argument('--scale_alpha', type=float, default=0.8, help='Strength of rescale REF1')
    parser.add_argument('--scale_beta', type=float, default=1.2, help='Strength of rescale REF2')
    # TF-ICON Hyper Parameter
    parser.add_argument('--outdir', type=str, default="./outputs", help='Directory for saving result')
    parser.add_argument('--seed', type=float, default=7414, help='Radom seed of diffusion model')
    parser.add_argument('--ckpt', type=str, default="v2-1_512-ema-pruned.ckpt", help='ckpt of stable diffusion model')
    parser.add_argument('--config', type=str, default="./configs/stable-diffusion/v2-inference.yaml", help='config of stable diffusion model')
    parser.add_argument('--prompt', type=str, default="", help='prompt for CFG (TFPHD is prompt-free but you can add prompt if you want)')

    opt = parser.parse_args()

    tfphd_main(opt)