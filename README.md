[![arXiv](https://img.shields.io/badge/arXiv-2404.12900-b31b1b.svg)](https://arxiv.org/abs/2404.12900)
# TF-GPH
Training-and-pormpt Free General Painterly Image Harmonization Using image-wise attention sharing
![image](https://github.com/BlueDyee/TF-GPH/blob/main/github_source/fig1.png)
![image](https://github.com/BlueDyee/TF-GPH/blob/main/github_source/tf-gph_demo.gif)
## Setup
Our codebase is built on [Stable-Diffusion](https://github.com/Stability-AI/stablediffusion)
and has shared dependencies and model architecture. A VRAM of 23 GB is recommended (RTX 3090 for example), though this may vary depending on the input samples (minimum 20 GB). 

This github repo is based on [TF-ICON](https://github.com/Shilin-LU/TF-ICON)  and [MasaCtrl](https://github.com/TencentARC/MasaCtrl/tree/main)
### Creating a Conda Environment

```
git clone https://github.com/BlueDyee/TF-GPH.git
cd TF-GPH
conda env create -f tfgph_env.yaml
conda activate tfgph
```

### Downloading Stable-Diffusion Weights

Download the StableDiffusion weights from the [Stability AI at Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/blob/main/v2-1_512-ema-pruned.ckpt)
(download the `sd-v2-1_512-ema-pruned.ckpt` file, This will occupy around 5GB storage)
For example

```
wget -O v2-1_512-ema-pruned.ckpt https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt?download=true
```
## Run
We provide three methods to run our repo **web app (gradio)/ipynb/py**
### app
Running the TF-GPH webui
```
python tfgph_app.py
```


### ipynb
```
Runall
```

### py
Using default parameters
```
python tfgph_main.py 
```
Customize parameters
```
python tfgph_main.py --ref1 "./inputs/demo_input/kangaroo.jpg" \
                     --ref2 "./inputs/demo_input/starry_night.jpg" \
                     --comp "./inputs/demo_input/kangaroo_starry.jpg" \
                     --share_step 15 \
                     --share_layer 12 \
```

### More Results
![image](https://github.com/BlueDyee/TF-GPH/blob/main/github_source/fig4.png)
![image](https://github.com/BlueDyee/TF-GPH/blob/main/github_source/fig17.png)
![image](https://github.com/BlueDyee/TF-GPH/blob/main/github_source/fig18.png)
![image](https://github.com/BlueDyee/TF-GPH/blob/main/github_source/fig14.png)
![image](https://github.com/BlueDyee/TF-GPH/blob/main/github_source/fig15.png)


