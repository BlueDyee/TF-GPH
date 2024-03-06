import os

import torch
import torch.nn.functional as F
import numpy as np

from einops import rearrange

from .masactrl_utils import AttentionBase

from torchvision.utils import save_image

# From diffuser/example/interpolate diffusion model
def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """helper function to spherically interpolate two arrays v1 v2"""


    dot = torch.sum(v0 * v1 / (torch.norm(v0) * torch.norm(v1)))
    if torch.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = torch.arccos(dot)
        sin_theta_0 = torch.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = torch.sin(theta_t)
        s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    return v2

class ShareSelfAttentionControl(AttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }

    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=20,scales=None, model_type="SD"):
        """
        Mutual self-attention control for Stable-Diffusion model
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            model_type: the model type, SD or SDXL
        """
        super().__init__()
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.h_layer=self.total_layers//2
        self.start_step = start_step
        self.start_layer = start_layer
        self.scales=scales
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.total_layers))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))
        print("MasaCtrl at denoising steps: ", self.step_idx)
        print("MasaCtrl at U-Net layers: ", self.layer_idx)
        if self.scales!=None:
            print("Rescales=",self.scales)

    def share_attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads,scales=None, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        b = q.shape[0] // num_heads
        n, d = q.shape[1],q.shape[2]
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)
        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")

        # print("q",q.shape)
        # print("k",k.shape)
        # print("v",v.shape)
        # print("sim",sim.shape)
        # Attention Mask
        mask = torch.zeros_like(input=sim,dtype=torch.bool)
        # mask cur to attend all but not to it self
        # X X X
        # X X X
        # O O X
        mask[:,-n:,:-n]=True
        # mask ref to attend only self
        # O X X
        # X O X
        # X X X
        for ref_idx in range(b-1):
            mask[:,ref_idx*n:(ref_idx+1)*n,ref_idx*n:(ref_idx+1)*n]=True

        max_neg_value = -torch.finfo(sim.dtype).max
        masked_sim=sim.masked_fill_(~mask, max_neg_value)
        
        if scales!=None:
            assert len(scales)==(b-1), "length of scales should equal to batch size-1 (-1 because self-value-ignorance)"
            for cur_scale,ref_idx in zip(scales,range(b-1)):
                masked_sim[:,-n:,ref_idx*n:(ref_idx+1)*n]*=cur_scale
        
        attn = (masked_sim).softmax(dim=-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        # !!! is_cross: using original attention forward
        # self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx :
        # performing multual attention after some step and after some layer
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        # Maintaining two image via batch manner
        # Normal: latents -> [uncond,cond]
        # Mutual: latents -> [ref-uncond,cur-uncond,ref-cond,ref-uncond]
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)
        
        # print("q",q.shape)
        # print("k",k.shape)
        # print("v",v.shape)
        # print("heads",num_heads)
        # qu->[q of ref-uncond, q of cur-uncond]
        # ku->[k of ref-uncond]
        # vu->[v of ref-uncond]
        # out-> [ref-latent, cur-latent]
        out_u = self.share_attn_batch(qu, ku, vu, sim, attnu, is_cross, place_in_unet, num_heads,scales=self.scales, **kwargs)
        out_c = self.share_attn_batch(qc, kc, vc, sim, attnc, is_cross, place_in_unet, num_heads,scales=self.scales, **kwargs)
        out = torch.cat([out_u, out_c], dim=0)
        # !!! Debug

        return out
    
class ShareSelfAttentionControlSlerp(AttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }

    def __init__(self, start_step=4, start_layer=10, layer_idx=None,
                 step_idx=None, total_steps=50,scales=None,slerp_ratio=0.1, model_type="SD"):
        """
        Mutual self-attention control for Stable-Diffusion model
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            model_type: the model type, SD or SDXL
        """
        super().__init__()
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.start_layer = start_layer
        self.scales=scales
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.total_layers))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))
        self.slerp_ratio=slerp_ratio
        print("ShareSelfCtrl at denoising steps: ", self.step_idx)
        print("ShareSelfCtrl at U-Net layers: ", self.layer_idx)
        if self.scales!=None:
            print("Rescales=",self.scales)

    def share_attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads,scales=None, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        b = q.shape[0] // num_heads
        n, d = q.shape[1],q.shape[2]
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)
        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")

        # print("q",q.shape)
        # print("k",k.shape)
        # print("v",v.shape)
        # print("sim",sim.shape)
        # Attention Mask
        mask = torch.zeros_like(input=sim,dtype=torch.bool)
        # mask cur to attend all but not to it self
        # X X X
        # X X X
        # O O X
        mask[:,-n:,:-n]=True
        # mask ref to attend only self
        # O X X
        # X O X
        # X X X
        for ref_idx in range(b-1):
            mask[:,ref_idx*n:(ref_idx+1)*n,ref_idx*n:(ref_idx+1)*n]=True

        max_neg_value = -torch.finfo(sim.dtype).max
        masked_sim=sim.masked_fill_(~mask, max_neg_value)
        
        if scales!=None:
            assert len(scales)==(b-1), "length of scales should equal to batch size-1 (-1 because self-value-ignorance)"
            for cur_scale,ref_idx in zip(scales,range(b-1)):
                masked_sim[:,-n:,ref_idx*n:(ref_idx+1)*n]*=cur_scale
        
        attn = (masked_sim).softmax(dim=-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        # print("out shape",out.shape)
        # (3,64,1280)
        if out.shape[1]<=64:
            # print("H-space")
            # !!!
            tmp_h=slerp(self.slerp_ratio,out[-1],out[-2]/torch.norm(out[-2])*torch.norm(out[-1]))
            out[-1]=tmp_h
            #out[0]=out[1]
            

        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        # !!! is_cross: using original attention forward
        # self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx :
        # performing multual attention after some step and after some layer
        # print(f"Cur layer {self.cur_att_layer}, q.shape{q.shape}, attn.shape{attn.shape}")
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        # Maintaining two image via batch manner
        # Normal: latents -> [uncond,cond]
        # Mutual: latents -> [ref-uncond,cur-uncond,ref-cond,ref-uncond]
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)
        
        # print("q",q.shape)
        # print("k",k.shape)
        # print("v",v.shape)
        # print("heads",num_heads)
        # qu->[q of ref-uncond, q of cur-uncond]
        # ku->[k of ref-uncond]
        # vu->[v of ref-uncond]
        # out-> [ref-latent, cur-latent]
        out_u = self.share_attn_batch(qu, ku, vu, sim, attnu, is_cross, place_in_unet, num_heads,scales=self.scales, **kwargs)
        out_c = self.share_attn_batch(qc, kc, vc, sim, attnc, is_cross, place_in_unet, num_heads,scales=self.scales, **kwargs)
        out = torch.cat([out_u, out_c], dim=0)
        # !!! Debug

        return out
