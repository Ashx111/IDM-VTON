import torch
from flask import Flask
import gc
import os
from torchvision import transforms
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)
from diffusers import AutoencoderKL
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

app = Flask(__name__)

# --- Load models globally ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pipe = None
unet = None
UNet_Encoder = None
parsing_model = None
openpose_model = None
tensor_transfrom = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

def load_models():
    print("loading models")
    global pipe, unet, UNet_Encoder, parsing_model, openpose_model
    dtype = torch.float16
    dtypeQuantize = dtype
    load_mode = '8bit'  # Or whatever your desired load mode is
    if load_mode in ('4bit', '8bit'):
        dtypeQuantize = torch.float8_e4m3fn

    model_id = 'yisol/IDM-VTON'
    vae_model_id = 'madebyollin/sdxl-vae-fp16-fix'

    unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        torch_dtype=dtypeQuantize,
    ).to(device)
    unet.requires_grad_(False)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        model_id,
        subfolder="image_encoder",
        torch_dtype=torch.float16,
    ).to(device)
    image_encoder.requires_grad_(False)

    if True:
        vae = AutoencoderKL.from_pretrained(vae_model_id, torch_dtype=dtype).to(device)
    else:
        vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=dtype
        ).to(device)
    vae.requires_grad_(False)

    UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
        model_id,
        subfolder="unet_encoder",
        torch_dtype=dtypeQuantize,
    ).to(device)
    UNet_Encoder.requires_grad_(False)

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

    if load_mode == '4bit':  # Keep your 4-bit logic if you use it
        from util.pipeline import quantize_4bit  # Import here if needed

        if pipe.text_encoder is not None:
            quantize_4bit(pipe.text_encoder)
            pipe.text_encoder.to(device)
        if pipe.text_encoder_2 is not None:
            quantize_4bit(pipe.text_encoder_2)
            pipe.text_encoder_2.to(device)

    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)
    openpose_model.preprocessor.body_estimation.model.to(device)
    print("loading models completed!!!!!!!!!!!!")

# Your API endpoints and other Flask code here...
import base64
from io import BytesIO
from flask import request, jsonify
from PIL import Image
from pydantic import BaseModel
from typing import List
from util.common import open_folder
from util.image import pil_to_binary_mask, save_output_image
from utils_mask import get_mask_location
import apply_net
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image

class TryonRequest(BaseModel):
    human_image: str
    garment_image: str
    garment_description: str
    category: str
    is_checked: bool
    is_checked_crop: bool
    denoise_steps: int
    is_randomize_seed: bool
    seed: int
    number_of_images: int

@app.route('/tryon', methods=['POST'])
def tryon_image():
    global pipe, parsing_model, openpose_model, tensor_transfrom
    print("***** Entered Tryon *****")
    # Validate input
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    required_fields = ['human_image_base64', 'garment_image_base64', 'garment_description', 
                      'category', 'is_checked', 'is_checked_crop', 'denoise_steps',
                      'is_randomize_seed', 'seed', 'number_of_images']
    
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        # Decode input images
        human_image = Image.open(BytesIO(base64.b64decode(data['human_image_base64']))).convert("RGB")
        garment_image = Image.open(BytesIO(base64.b64decode(data['garment_image_base64']))).convert("RGB")
        
        # Process human image
        human_img_orig = human_image.copy()
        if data['is_checked_crop']:
            width, height = human_img_orig.size
            target_width = min(width, int(height * 0.75))
            target_height = min(height, int(width * 1.333))
            left = (width - target_width) // 2
            top = (height - target_height) // 2
            human_img = human_img_orig.crop((left, top, left+target_width, top+target_height)).resize((768, 1024))
        else:
            human_img = human_img_orig.resize((768, 1024))

        # Process garment image
        garm_img = garment_image.resize((768, 1024))

        # Generate mask
        if data['is_checked']:
            keypoints = openpose_model(human_img.resize((384, 512)))
            model_parse, _ = parsing_model(human_img.resize((384, 512)))
            mask, _ = get_mask_location('hd', data['category'], model_parse, keypoints)
            mask = mask.resize((768, 1024))
        else:
            mask = pil_to_binary_mask(human_img)

        # Generate pose image
        human_img_arg = convert_PIL_to_numpy(_apply_exif_orientation(human_img.resize((384, 512))), "BGR")
        pose_img = apply_net.create_argument_parser().parse_args(
            ('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', 
             './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', 
             '--opts', 'MODEL.DEVICE', 'cuda')
        ).func(human_img_arg)
        pose_img = Image.fromarray(pose_img[:, :, ::-1]).resize((768, 1024))

        # Prepare tensors
        pose_tensor = tensor_transfrom(pose_img).unsqueeze(0).to(device, torch.float16)
        garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).to(device, torch.float16)
        mask_tensor = tensor_transfrom(mask).unsqueeze(0).to(device, torch.float16)

        # Generate images
        results = []
        current_seed = data['seed']
        
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            # Encode prompts once
            main_prompt = f"model is wearing {data['garment_description']}"
            cloth_prompt = f"a photo of {data['garment_description']}"
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            
            # Main prompt encoding
            prompt_embeds, negative_embeds, pooled_embeds, negative_pooled = pipe.encode_prompt(
                main_prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt
            )
            
            # Clothing prompt encoding
            cloth_embeds, *_ = pipe.encode_prompt(
                cloth_prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=negative_prompt
            )

            for idx in range(data['number_of_images']):
                # Seed management
                if data['is_randomize_seed']:
                    current_seed = torch.randint(0, 2**32 - 1, (1,)).item()
                generator = torch.Generator(device).manual_seed(current_seed)
                
                # Generate image
                output = pipe(
                    prompt_embeds=prompt_embeds.to(device),
                    negative_prompt_embeds=negative_embeds.to(device),
                    pooled_prompt_embeds=pooled_embeds.to(device),
                    negative_pooled_prompt_embeds=negative_pooled.to(device),
                    num_inference_steps=data['denoise_steps'],
                    generator=generator,
                    pose_img=pose_tensor,
                    text_embeds_cloth=cloth_embeds.to(device),
                    cloth=garm_tensor,
                    mask_image=mask_tensor,
                    image=human_img,
                    ip_adapter_image=garm_img.resize((768, 1024)),
                    guidance_scale=2.0
                ).images[0]

                # Post-process output
                if data['is_checked_crop']:
                    output = output.resize(human_img.size)
                    human_img_orig.paste(output, (left, top))
                    final_img = human_img_orig.copy()
                else:
                    final_img = output

                # Convert to base64
                buffered = BytesIO()
                final_img.save(buffered, format="JPEG", quality=95)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                results.append({
                    "seed": current_seed,
                    "image": img_str,
                    "format": "image/jpeg"
                })
                
                current_seed += 1  # Increment seed for next image

        return jsonify({
            "success": True,
            "results": results,
            "count": len(results)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500

    finally:
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()
        print("***** Exited Tryon *****")

@app.route("/check")
def check():
    return "API is running"

if __name__ == "__main__":
    # Load models only if not in debug mode's reloader
    if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        with app.app_context():
            load_models()
    app.run(host="0.0.0.0", port=5000, debug=False)
