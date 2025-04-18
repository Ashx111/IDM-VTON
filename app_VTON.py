import gradio as gr
import argparse, torch, os
from PIL import Image
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)
from diffusers import AutoencoderKL
from typing import List
from util.common import open_folder
from util.image import pil_to_binary_mask, save_output_image
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
from util.pipeline import quantize_4bit, restart_cpu_offload, torch_gc

parser = argparse.ArgumentParser()
parser.add_argument("--share", type=str, default=False, help="Set to True to share the app publicly.")
parser.add_argument("--lowvram", action="store_true", help="Enable CPU offload for model operations.")
parser.add_argument("--load_mode", default=None, type=str, choices=["4bit", "8bit"], help="Quantization mode for optimization memory consumption")
parser.add_argument("--fixed_vae", action="store_true", default=True,  help="Use fixed vae for FP16.")
args = parser.parse_args()

load_mode = args.load_mode
fixed_vae = args.fixed_vae

dtype = torch.float16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_id = 'yisol/IDM-VTON'
vae_model_id = 'madebyollin/sdxl-vae-fp16-fix'

dtypeQuantize = dtype

if(load_mode in ('4bit','8bit')):
    dtypeQuantize = torch.float8_e4m3fn

ENABLE_CPU_OFFLOAD = args.lowvram
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.allow_tf32 = False
need_restart_cpu_offloading = False

unet = None
pipe = None
UNet_Encoder = None
example_path = os.path.join(os.path.dirname(__file__), 'example')

#def start_tryon(dict, garm_img, garment_des, category, is_checked, is_checked_crop, denoise_steps, is_randomize_seed, seed, number_of_images):
def start_tryon(human_img_pil, garm_img_pil, garment_des, category, is_checked, is_checked_crop, denoise_steps, is_randomize_seed, seed, number_of_images):
    global pipe, unet, UNet_Encoder, need_restart_cpu_offloading

    if pipe == None:
        unet = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
            torch_dtype=dtypeQuantize,
        )
        if load_mode == '4bit':
            quantize_4bit(unet)
            
        unet.requires_grad_(False)
       
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            model_id,
            subfolder="image_encoder",
            torch_dtype=torch.float16,
            )
        if load_mode == '4bit':
            quantize_4bit(image_encoder)
        
        if fixed_vae:
            vae = AutoencoderKL.from_pretrained(vae_model_id, torch_dtype=dtype)
        else:            
            vae = AutoencoderKL.from_pretrained(model_id,
                                                subfolder="vae",
                                                torch_dtype=dtype,
            )

        # "stabilityai/stable-diffusion-xl-base-1.0",
        UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
            model_id,
            subfolder="unet_encoder",
            torch_dtype=dtypeQuantize,
        )
     
        if load_mode == '4bit':
            quantize_4bit(UNet_Encoder)

        UNet_Encoder.requires_grad_(False)
        image_encoder.requires_grad_(False)
        vae.requires_grad_(False)
        unet.requires_grad_(False)
              
        pipe_param = {
                'pretrained_model_name_or_path': model_id,
                'unet': unet,     
                'torch_dtype': dtype,   
                'vae': vae,
                'image_encoder': image_encoder,
                'feature_extractor': CLIPImageProcessor(),
            }
        
        pipe = TryonPipeline.from_pretrained(**pipe_param).to(device)
        pipe.unet_encoder = UNet_Encoder    
        pipe.unet_encoder.to(pipe.unet.device)

        if load_mode == '4bit':
            if pipe.text_encoder is not None:
                quantize_4bit(pipe.text_encoder)
            if pipe.text_encoder_2 is not None:
                quantize_4bit(pipe.text_encoder_2)
           
    else:
        if ENABLE_CPU_OFFLOAD:
            need_restart_cpu_offloading =True
    
    torch_gc() 
    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)
    openpose_model.preprocessor.body_estimation.model.to(device)
    tensor_transfrom = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ]
            )
    
    if need_restart_cpu_offloading:
        restart_cpu_offload(pipe, load_mode)
    elif ENABLE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()

    #if load_mode != '4bit' :
    #    pipe.enable_xformers_memory_efficient_attention()    

    garm_img = garm_img_pil.convert("RGB").resize((768, 1024))

    # Determine human_img_orig and potential input mask based on input type
    human_img_orig = None
    input_mask_pil = None
    if isinstance(human_img_pil, dict) and 'image' in human_img_pil:
        # Input from UI with sketchpad (might have mask)
        human_img_orig = human_img_pil['image'].convert("RGB")
        if 'mask' in human_img_pil and human_img_pil['mask'] is not None:
            input_mask_pil = human_img_pil['mask'].convert("RGB")
            print("Processing mask from UI sketchpad.")
        else:
             print("Processing image from UI sketchpad (no mask drawn).")
    elif isinstance(human_img_pil, Image.Image):
        # Input from API or UI without sketchpad use
        human_img_orig = human_img_pil.convert("RGB")
        print("Processing input as plain PIL Image (API call or no UI mask).")
    else:
        raise TypeError(f"Unexpected type for human image input: {type(human_img_pil)}")

    # Apply cropping to human_img_orig if requested
    crop_size = (768, 1024) # Default if no crop
    if is_checked_crop:
        print("Applying auto-crop...")
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size # Store actual crop size
        human_img = cropped_img.resize((768, 1024))
        print(f"Cropped to {crop_size}, Resized to (768, 1024)")
    else:
        print("No auto-crop. Resizing original to (768, 1024)")
        human_img = human_img_orig.resize((768, 1024))
        # Note: crop_size remains (768, 1024) which is correct for paste-back logic if no crop applied

    # Determine the final mask to use
    mask = None
    if is_checked:
        print("Generating auto-mask...")
        # Note: Depending on get_mask_location, it might return one or two values.
        # Assuming it returns (PIL mask, PIL mask_gray) based on original code structure.
        keypoints = openpose_model(human_img.resize((384, 512)))
        model_parse, _ = parsing_model(human_img.resize((384, 512)))
        # *** Adjust the line below based on what get_mask_location ACTUALLY returns ***
        mask, mask_gray_pil = get_mask_location('hd', category, model_parse, keypoints) # Example assumption
        mask = mask.resize((768, 1024))
        # If get_mask_location only returns mask, calculate mask_gray here:
        # mask_gray_pil = to_pil_image(((1 - transforms.ToTensor()(mask)) * tensor_transfrom(human_img) + 1.0) / 2.0)
        print("Auto-mask generated.")
    elif input_mask_pil is not None:
        # Use the mask provided from the sketchpad
        print("Using provided sketchpad mask...")
        mask = pil_to_binary_mask(input_mask_pil.resize((768, 1024)))
        # Calculate mask_gray based on this mask
        mask_gray_pil = to_pil_image(((1 - transforms.ToTensor()(mask)) * tensor_transfrom(human_img) + 1.0) / 2.0)
        print("Sketchpad mask processed.")
    else:
        # Manual mask selected, but none provided (e.g., API call) -> Use default black mask
        print("Warning: Manual masking selected, but no mask provided. Using default black mask.")
        mask = Image.new('L', (768, 1024), 0) # Black mask
        # Calculate mask_gray based on black mask
        mask_gray_pil = to_pil_image(((1 - transforms.ToTensor()(mask)) * tensor_transfrom(human_img) + 1.0) / 2.0)

    # Ensure mask_gray_pil is assigned (should be handled above, but as fallback)
    if 'mask_gray_pil' not in locals():
         print("Calculating mask_gray as fallback.")
         mask_gray_pil = to_pil_image(((1 - transforms.ToTensor()(mask)) * tensor_transfrom(human_img) + 1.0) / 2.0)


    # Prepare DensePose input using the final processed human_img
    print("Generating DensePose...")
    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args_densepose = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    pose_img = args_densepose.func(args_densepose, human_img_arg)
    pose_img = pose_img[:, :, ::-1] # BGR to RGB
    pose_img = Image.fromarray(pose_img).resize((768, 1024))
    print("DensePose generated.")

    # Move text encoders to device
    if pipe.text_encoder is not None:
        pipe.text_encoder.to(device)
    if pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.to(device)


    with torch.no_grad():
        # Extract the images
        with torch.cuda.amp.autocast(dtype=dtype):
            with torch.no_grad():
                prompt = "model is wearing " + garment_des
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                with torch.inference_mode():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )
                                    
                    prompt = "a photo of " + garment_des
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                    if not isinstance(prompt, List):
                        prompt = [prompt] * 1
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * 1
                    with torch.inference_mode():
                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                        )

                    pose_img =  tensor_transfrom(pose_img).unsqueeze(0).to(device,dtype)
                    garm_tensor =  tensor_transfrom(garm_img).unsqueeze(0).to(device,dtype)
                    results = []
                    current_seed = seed
                    print(f"Debug: Received number_of_images = {number_of_images} (type: {type(number_of_images)})")
                    for i in range(int(number_of_images)):  
                        if is_randomize_seed:
                            current_seed = torch.randint(0, 2**32, size=(1,)).item() 
                        print(f"Debug: Using seed = {current_seed} (type: {type(current_seed)}) for generator.")
                        print(f"Debug: Original seed param = {seed} (type: {type(seed)})")
                        seed_for_generator = int(current_seed) if is_randomize_seed else int(seed)
                        generator = torch.Generator(device).manual_seed(seed_for_generator) if seed != -1 else None                     
                        current_seed = seed_for_generator + i

                        num_inference_steps_int = int(denoise_steps)

                        images = pipe(
                            prompt_embeds=prompt_embeds.to(device,dtype),
                            negative_prompt_embeds=negative_prompt_embeds.to(device,dtype),
                            pooled_prompt_embeds=pooled_prompt_embeds.to(device,dtype),
                            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device,dtype),
                            # --- Use the integer version ---
                            num_inference_steps=num_inference_steps_int,
                            generator=generator,
                            strength = 1.0,
                            pose_img = pose_img.to(device,dtype),
                            text_embeds_cloth=prompt_embeds_c.to(device,dtype),
                            cloth = garm_tensor.to(device,dtype),
                            mask_image=mask,
                            image=human_img,
                            height=1024,
                            width=768,
                            ip_adapter_image = garm_img.resize((768,1024)), # garm_img should already be 768x1024, resize is safe
                            guidance_scale=2.0,
                            dtype=dtype,
                            device=device,
                        )[0]
                        output_image_pil = None
                        if is_checked_crop:
                            out_img = images[0].resize(crop_size)
                            # Create a copy to avoid modifying the original if needed elsewhere
                            final_image_pil = human_img_orig.copy()
                            final_image_pil.paste(out_img, (int(left), int(top)))
                            output_image_pil = final_image_pil # Assign the pasted image
                            # REMOVE: img_path = save_output_image(human_img_orig, base_path="outputs", base_filename='img', seed=current_seed)
                            # results.append(img_path) # Don't append path
                        else:
                            output_image_pil = images[0] # Assign the direct output image
                            # REMOVE: img_path = save_output_image(images[0], base_path="outputs", base_filename='img')
                            # results.append(img_path) # Don't append path

                        # Append the actual PIL image object to the results list
                        if output_image_pil:
                             results.append(output_image_pil)
                            
                        return results, mask_gray_pil
    
garm_list = os.listdir(os.path.join(example_path,"cloth"))
garm_list_path = [os.path.join(example_path,"cloth",garm) for garm in garm_list]

human_list = os.listdir(os.path.join(example_path,"human"))
human_list_path = [os.path.join(example_path,"human",human) for human in human_list]

human_ex_list = []
for ex_human in human_list_path:
    # Only include paths that meet the criteria
    if "Jensen" in ex_human or "sam1 (1)" in ex_human:
        human_ex_list.append(ex_human)

image_blocks = gr.Blocks().queue()
with image_blocks as demo:
    gr.Markdown("## V14 - IDM-VTON 👕👔👚 improved by SECourses : 1-Click Installers Latest Version On : https://www.patreon.com/posts/103022942")
    gr.Markdown("Virtual Try-on with your image and garment image. Check out the [source codes](https://github.com/yisol/IDM-VTON) and the [model](https://huggingface.co/yisol/IDM-VTON)")
    with gr.Row():
        with gr.Column():
            #imgs = gr.ImageEditor(sources='upload', type="pil", label='Human. Mask with pen or use auto-masking', interactive=True , height=550)
            imgs = gr.Image(tool='sketchpad', type="pil", label='Human. Mask with pen or use auto-masking', interactive=True , height=550)
            with gr.Row():
                category = gr.Radio(choices=["upper_body", "lower_body", "dresses"], label="Select Garment Category", value="upper_body")
                is_checked = gr.Checkbox(label="Yes", info="Use auto-generated mask (Takes 5 seconds)",value=True)
            with gr.Row():
                is_checked_crop = gr.Checkbox(label="Yes", info="Use auto-crop & resizing",value=True)

            example = gr.Examples(
                inputs=imgs,
                examples_per_page=2,
                examples=human_ex_list
            )

        with gr.Column():
            #garm_img = gr.Image(label="Garment", sources='upload', type="pil")
            garm_img = gr.Image(label="Garment", type="pil")
            with gr.Row(elem_id="prompt-container"):
                with gr.Row():
                    prompt = gr.Textbox(placeholder="Description of garment ex) Short Sleeve Round Neck T-shirts", show_label=False, elem_id="prompt")
            example = gr.Examples(
                inputs=garm_img,
                examples_per_page=8,
                examples=garm_list_path)
        with gr.Column():
            with gr.Row():
            # image_out = gr.Image(label="Output", elem_id="output-img", height=400)
                masked_img = gr.Image(label="Masked image output", elem_id="masked-img",show_share_button=False)
            with gr.Row():
                btn_open_outputs = gr.Button("Open Outputs Folder")
                btn_open_outputs.click(fn=open_folder)
        with gr.Column():
            with gr.Row():
            # image_out = gr.Image(label="Output", elem_id="output-img", height=400)
                image_gallery = gr.Gallery(label="Generated Images", show_label=True)
            with gr.Row():
                try_button = gr.Button(value="Try-on")
                denoise_steps = gr.Number(label="Denoising Steps", minimum=20, maximum=120, value=30, step=1)
                seed = gr.Number(label="Seed", minimum=-1, maximum=2147483647, step=1, value=1)
                is_randomize_seed = gr.Checkbox(label="Randomize seed for each generated image", value=True)  
                number_of_images = gr.Number(label="Number Of Images To Generate (it will start from your input seed and increment by 1)", minimum=1, maximum=9999, value=1, step=1)


    try_button.click(fn=start_tryon, inputs=[imgs, garm_img, prompt, category, is_checked, is_checked_crop, denoise_steps, is_randomize_seed, seed, number_of_images], outputs=[image_gallery, masked_img],api_name='tryon')

image_blocks.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=False)
