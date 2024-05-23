import argparse
import os
import json
from collections import defaultdict
from numbers import Number
from typing import Any, Optional, Tuple, Union

import clip
import lpips
import numpy as np
import torch
import torch.nn.functional as F_torch
import torchvision.transforms.functional as F_vision
from PIL import Image
from tqdm import tqdm
import cv2


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--imgs-root", type=str, default="./GPH Benchmark demo data", help="Path to ground truth images"
    )
    parser.add_argument(
        "-cm",
        "--clip-model-type",
        type=str,
        default="ViT-B/32",
        choices=["ViT-B/32", "ViT-L/14"],
        help="CLIP model type",
    )
    parser.add_argument(
        "-b",
        "--box_meter",
        type=bool,
        default=False,
        help="If enable box evaluation",
    )
    args = parser.parse_args()
    return args


def to_rgb_tensor(img: np.ndarray, size: Union[Tuple[int, int]] = None):
    tensor = F_vision.to_tensor(img)
    if size is not None:
        tensor = F_vision.resize(tensor, size, antialias=True)
    return tensor.unsqueeze(0)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: Number, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricMeter(object):
    def __init__(self, delimiter: str = "\t"):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

    def update(self, input_dict: Optional[dict]):
        if input_dict is None:
            return

        if not isinstance(input_dict, dict):
            raise TypeError("Input to MetricMeter.update() must be a dictionary")

        for k, v in input_dict.items():
            if v is not None:
                self.meters[k].update(v)

    def __str__(self):
        output_str = []
        for name, meter in self.meters.items():
            output_str.append(f"{name} {meter.avg:.4f}")
        return self.delimiter.join(output_str)

    def get_log_dict(self):
        log_dict = {}
        for name, meter in self.meters.items():
            log_dict[name] = meter.val
            log_dict[f"avg_{name}"] = meter.avg
        return log_dict


@torch.no_grad()
def compute_lpips(
    source_img: np.ndarray,
    style_img: np.ndarray,
    stylized_img: np.ndarray,
    composite_mask: np.ndarray,
    lpips_model: torch.nn.Module,
):

    non_x, non_y = np.nonzero(composite_mask)
    left, right = non_y.min(), non_y.max()
    top, bottom = non_x.min(), non_x.max()
    #fg_img = (stylized_img * composite_mask[:, :, None])[top:bottom, left:right]
    crop_stylized_img = (stylized_img * composite_mask[:, :, None])[top:bottom, left:right]
    crop_source= (source_img * composite_mask[:, :, None])[top:bottom, left:right]
    bg_stylized_img = stylized_img * (1 - composite_mask[:, :, None])
    bg_style_img = style_img * (1 - composite_mask[:, :, None])
    
    """
    pil_image = Image.fromarray(bg_stylized_img)
    pil_image.save("./test/bg_img.png")

    pil_image = Image.fromarray(bg_style_img)
    pil_image.save("./test/bg_style_img.png")
    exit()
    """
    
    
    fg_score = lpips_model(
        to_rgb_tensor(crop_stylized_img,size=source_img.shape[:2]), to_rgb_tensor(crop_source,size=source_img.shape[:2])
    ).item()
    bg_score = lpips_model(to_rgb_tensor(bg_stylized_img), to_rgb_tensor(bg_style_img)).item()

    return fg_score, bg_score


@torch.no_grad()
def compute_clip_score(
    source_img: np.ndarray,
    style_img: np.ndarray,
    stylized_img: np.ndarray,
    composite_mask: np.ndarray,
    clip_model: Any,
    device: Union[torch.device, str],
    prompt: Union[str] = None,
):
    non_x, non_y = np.nonzero(composite_mask)
    left, right = non_y.min(), non_y.max()
    top, bottom = non_x.min(), non_x.max()
    #fg_img = (stylized_img * composite_mask[:, :, None])[top:bottom, left:right]
    crop_stylized_img = (stylized_img * composite_mask[:, :, None])[top:bottom, left:right]
    crop_source_img= (source_img * composite_mask[:, :, None])[top:bottom, left:right]
    #bg_stylized_img = stylized_img * (1 - composite_mask[:, :, None])
    #bg_style_img = style_img * (1 - composite_mask[:, :, None])

    preprocess_crop_stylized_img = clip_model[1](Image.fromarray(crop_stylized_img)).unsqueeze(0).to(device)
    preprocess_crop_fg_img = clip_model[1](Image.fromarray(crop_source_img)).unsqueeze(0).to(device)

    crop_stylized_features = F_torch.normalize(clip_model[0].encode_image(preprocess_crop_stylized_img), dim=-1)
    crop_fg_features = F_torch.normalize(clip_model[0].encode_image(preprocess_crop_fg_img), dim=-1)

    img_score = (crop_stylized_features .squeeze() @ crop_fg_features.squeeze()).item() * 100

    preprocess_stylized_img = clip_model[1](Image.fromarray(stylized_img)).unsqueeze(0).to(device)
    preprocess_style_img = clip_model[1](Image.fromarray(style_img)).unsqueeze(0).to(device)
    #stylized_features = F_torch.normalize(clip_model[0].encode_image(preprocess_stylized_img), dim=-1)
    style_features = F_torch.normalize(clip_model[0].encode_image(preprocess_style_img), dim=-1)
    
    style_score = (crop_stylized_features .squeeze() @ style_features.squeeze()).item() * 100

    preprocess_source_img = clip_model[1](Image.fromarray(source_img)).unsqueeze(0).to(device)
    source_features = F_torch.normalize(clip_model[0].encode_image(preprocess_source_img), dim=-1)
    dfg=crop_stylized_features-crop_fg_features
    dbg=style_features-source_features
    dir_score=(dfg .squeeze() @ dbg.squeeze()).item() * 100

    text_score = None
    if prompt is not None:
        tokenized_prompt = clip.tokenize([prompt]).to(device)
        text_features = F_torch.normalize(clip_model[0].encode_text(tokenized_prompt), dim=-1)
        #text_score = (stylized_features.squeeze() @ text_features.squeeze()).item() * 100


    return img_score, style_score, dir_score, text_score


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_fn = lpips.LPIPS(net="alex")
    clip_model = clip.load(args.clip_model_type, device=device)

    meter = MetricMeter()
    if args.box_meter:
        box_meter = MetricMeter()
    else:
        box_meter=False
    idxs=[file[:-4] for file in os.listdir(os.path.join(args.imgs_root, "harmonized_data"))]
    for i in tqdm(idxs):
        source_img_path = os.path.join(args.imgs_root, "composite_data", f"{i}.png")
        composite_mask_path = os.path.join(args.imgs_root, "mask_data", f"{i}.png")
        style_img_path = os.path.join(args.imgs_root, "background_data",  f"{i}.png")
        stylized_img_path = os.path.join(args.imgs_root, "harmonized_data",  f"{i}.png")

        source_img = np.array(Image.open(source_img_path).convert("RGB"))
        composite_mask = np.array(Image.open(composite_mask_path).convert("L"))//255
        style_img = np.array(Image.open(style_img_path).convert("RGB"))
        stylized_img = np.array(Image.open(stylized_img_path).convert("RGB"))
        
        fg_lpips, bg_lpips = compute_lpips(
            source_img, style_img, stylized_img, composite_mask, lpips_fn
        )
        img_clip, style_clip, dir_clip, text_clip = compute_clip_score(
            source_img,
            style_img,
            stylized_img,
            composite_mask,
            clip_model,
            device,
            #prompt=f"a photo in {style_text} style",
        )

        meter.update(
            {
                "bg_lpips": bg_lpips,
                "fg_lpips": fg_lpips,
                "img_clip": img_clip,
                "style_clip": style_clip,
                "dir_clip": dir_clip,
                #"text_clip": text_clip,
            }
        )

    print("Segment meter:",meter)


if __name__ == "__main__":
    args = parse_argument()
    main(args)
