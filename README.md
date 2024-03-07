# tfphd
Training-Free Painterly Image Harmonization Using Diffusion Model
## Setup

Our codebase is built on [Stable-Diffusion](https://github.com/Stability-AI/stablediffusion)
and has shared dependencies and model architecture. A VRAM of 23 GB is recommended, though this may vary depending on the input samples (minimum 20 GB).

### Creating a Conda Environment

```
git clone https://github.com/BlueDyee/tfphd.git
cd tfphd
conda env create -f tfphd_env.yaml
conda activate tfphd
```

### Downloading Stable-Diffusion Weights

Download the StableDiffusion weights from the [Stability AI at Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/blob/main/v2-1_512-ema-pruned.ckpt)
(download the `sd-v2-1_512-ema-pruned.ckpt` file, This will occupy around 5GB storage)
For example
'''
wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt?download=true
'''
