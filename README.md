# TF-GPH
Training-and-pormpt Free General Painterly Image Harmonization Using image-wise attention sharing
## Setup

Our codebase is built on [Stable-Diffusion](https://github.com/Stability-AI/stablediffusion)
and has shared dependencies and model architecture. A VRAM of 23 GB is recommended (RTX 3090 for example), though this may vary depending on the input samples (minimum 20 GB). 

This github repo is based on [TF-ICON](https://github.com/Shilin-LU/TF-ICON)  and [MasaCtrl](https://github.com/TencentARC/MasaCtrl/tree/main)
### Creating a Conda Environment

```
git clone https://github.com/BlueDyee/TFPHD.git
cd TFPHD
conda env create -f tfphd_env.yaml
conda activate tfphd
```

### Downloading Stable-Diffusion Weights

Download the StableDiffusion weights from the [Stability AI at Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/blob/main/v2-1_512-ema-pruned.ckpt)
(download the `sd-v2-1_512-ema-pruned.ckpt` file, This will occupy around 5GB storage)
For example

```
wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt?download=true
(But in this way the file name will be "v2-1_512-ema-pruned.ckpt?download=true", you need to manually change to v2-1_512-ema-pruned.ckpt)
```
## Run
We provide three methods to run our repo **py/ipynb/app (gradio)**
### py
```
python tfphd.py
```
### ipynb
```
Runall
```
### app
Working on it

