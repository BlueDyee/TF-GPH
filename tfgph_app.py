import gradio as gr
from app_util import tfgph_load, tfgph_inverse, tfgph_harmonize
import time
import torch
import PIL

opt = {
    "seed": 4753,
    "ckpt": "v2-1_512-ema-pruned.ckpt",
    "config": "./configs/stable-diffusion/v2-inference.yaml",
    "scale": 5,
    "n_samples": 1,
    "f": 16,
    "C": 4,
    "total_steps":25,
    "ddim_eta": 0.0,
    "outdir": "./outputs/",
    "order":2,
    "prompt":""
}
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def app_load_model(x=None,progress=gr.Progress()):
    progress((50,100),desc="Loading model (this might take 0~30 seconds)... ")
    #img = PIL.Image.open("./inputs/demo_input/kangaroo.jpg")
    model,sampler= tfgph_load(opt,device)
    opt["model"]=model
    opt["sampler"]=sampler
    return "Model Loaded 💪"
def app_inv_img(img,idx,progress=gr.Progress()):
    progress((50,100),desc="Prepocessing...")
    #img = PIL.Image.open("./inputs/demo_input/kangaroo.jpg")
    # Resize the image with the LANCZOS resampling filter
    resized_img = img.resize((512, 512), resample=PIL.Image.LANCZOS)
    z_ref=tfgph_inverse(resized_img,opt,opt["model"],opt["sampler"],device)
    opt[f"z_ref{idx}"]=z_ref
    return img
def app_har(share_step,share_layer,alpha,beta):

    opt["share_step"]=int(share_step)
    opt["share_layer"]=int(share_layer)
    opt["scale_alpha"]=alpha
    opt["scale_beta"]=beta

    out=tfgph_harmonize(opt["z_ref1"],opt["z_ref2"],opt["z_ref3"],opt,opt["model"],opt["sampler"],device)
    return out

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    TITLE="""# Welcome to our TF-GPH web-demo🥰,
    the TF-GPH method is aiming to support artistic image generation🖼️ (e.g. Style Transfer, Painterly Harmonization),
    using user-specific image composition as instruction.
    """
    gr.Markdown(TITLE)
    #model, sampler, z_ref1, z_ref2, z_comp=gr.State(), gr.State(), gr.State(), gr.State(), gr.State()
    load_btn=gr.Button("Load TF-GPH model (❗ Insure you have already load the model before you start)")
    load_label = gr.Label(value="Model not yet loadded ❌",label="TF-GPH model status")
    load_btn.click(app_load_model,inputs=None,outputs=[load_label])
    
    gallery_val=["./inputs/demo_input/kangaroo.jpg","./inputs/demo_input/starry_night.jpg","./inputs/demo_input/kangaroo_starry.jpg"]
    gallery = gr.Gallery(value=gallery_val,label="Generated images", show_label=False, elem_id="gallery", columns=[6], rows=[1], object_fit="contain", height="auto")
    with gr.Row():
        layout_h=300
        layout_w=300
        

        ref1=gr.Image(scale=1,height=layout_h,width=layout_w,image_mode="RGB",label="reference 1 (foreground object)",type="pil")
        ref2=gr.Image(scale=1,height=layout_h,width=layout_w,image_mode="RGB",label="reference 2 (background image)",type="pil")
        ref3=gr.Image(scale=1,height=layout_h,width=layout_w,image_mode="RGB",label="composite (composite image)",type="pil")

        ref1.upload(app_inv_img, [ref1,gr.State(value=1)],[ref1])
        ref2.upload(app_inv_img, [ref2,gr.State(value=2)],[ref2])
        ref3.upload(app_inv_img, [ref3,gr.State(value=3)],[ref3])
    with gr.Row():
        with gr.Column():
            share_step=gr.Slider(value=20,minimum=0,maximum=opt["total_steps"],label="share_step (lower->stronger)")
            share_layer=gr.Slider(value=8,minimum=0,maximum=16,label="share_layer (lower->stronger)")
            alpha=gr.Slider(value=0.9,minimum=0,maximum=10,label="ref1 weight (higher->stronger)")
            beta=gr.Slider(value=1.1,minimum=0,maximum=10,label="ref2 weight (higher->stronger)")
            har_btn=gr.Button("Run !")
            

            opt["share_step"]=share_step
            opt["share_layer"]=share_layer
            opt["scale_alpha"]=alpha
            opt["scale_beta"]=beta

        out=gr.Image(scale=1,image_mode="RGB",label="Harmonized image")
        har_btn.click(app_har,inputs=[share_step,share_layer,alpha,beta],outputs=[out])
if __name__ == "__main__":
    demo.launch(debug=True, enable_queue=True)  
    #demo.launch(server_port=8002, debug=True, enable_queue=True)   