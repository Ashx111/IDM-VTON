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
    global pipe, parsing_model, openpose_model, tensor_transfrom  # Access the globally loaded models

    print("Entering tryon_image function")
    data = request.get_json()
    print("Received JSON data:", data)  # Check if JSON is parsed correctly

    try:
        print("Attempting to open human image:", data.get('human_image'))
        human_image = Image.open(data['human_image'])
        print("Human image opened successfully")
    except FileNotFoundError as e:
        print(f"Error opening human image: {e}")
        return jsonify({"error": f"Human image not found: {e}"}), 500
    except Exception as e:
        print(f"Error opening human image: {e}")
        return jsonify({"error": f"Error opening human image: {e}"}), 500

    try:
        print("Attempting to open garment image:", data.get('garment_image'))
        garment_image = Image.open(data['garment_image'])
        print("Garment image opened successfully")
    except FileNotFoundError as e:
        print(f"Error opening garment image: {e}")
        return jsonify({"error": f"Garment image not found: {e}"}), 500
    except Exception as e:
        print(f"Error opening garment image: {e}")
        return jsonify({"error": f"Error opening garment image: {e}"}), 500

    try:
        print("Converting and resizing garment image")
        garm_img = garment_image.convert("RGB").resize((768, 1024))
        print("Garment image converted and resized")
    except Exception as e:
        print(f"Error converting/resizing garment image: {e}")
        return jsonify({"error": f"Error processing garment image: {e}"}), 500

    try:
        print("Converting human image for original size")
        human_img_orig = human_image.convert("RGB")
        print("Human image converted for original size")
    except Exception as e:
        print(f"Error converting human image for original size: {e}")
        return jsonify({"error": f"Error processing human image: {e}"}), 500

    try:
        print("Checking for crop:", data.get('is_checked_crop'))
        if data['is_checked_crop']:
            print("Cropping enabled")
            width, height = human_img_orig.size
            target_width = int(min(width, height * (3 / 4)))
            target_height = int(min(height, width * (4 / 3)))
            left = (width - target_width) / 2
            top = (height - target_height) / 2
            right = (width + target_width) / 2
            bottom = (height + target_height) / 2
            cropped_img = human_img_orig.crop((left, top, right, bottom))
            crop_size = cropped_img.size
            human_img = cropped_img.resize((768, 1024))
            print("Human image cropped and resized")
        else:
            print("Cropping disabled")
            human_img = human_img_orig.resize((768, 1024))
            print("Human image resized")
    except Exception as e:
        print(f"Error during cropping/resizing: {e}")
        return jsonify({"error": f"Error processing human image for cropping: {e}"}), 500

    try:
        print("Checking for pose estimation:", data.get('is_checked'))
        if data['is_checked']:
            print("Pose estimation enabled")
            keypoints = openpose_model(human_img.resize((384, 512)))
            print("OpenPose keypoints:", keypoints)
            model_parse, _ = parsing_model(human_img.resize((384, 512)))
            print("Parsing model output:", model_parse)
            mask, mask_gray = get_mask_location('hd', data['category'], model_parse, keypoints)
            mask = mask.resize((768, 1024))
            print("Mask generated")
        else:
            print("Pose estimation disabled")
            mask = pil_to_binary_mask(human_image.convert("RGB").resize((768, 1024)))
            print("Mask generated from image")
    except Exception as e:
        print(f"Error during pose estimation/masking: {e}")
        return jsonify({"error": f"Error processing pose/mask: {e}"}), 500

    try:
        print("Creating mask_gray")
        mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
        mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)
        print("mask_gray created")
    except Exception as e:
        print(f"Error creating mask_gray: {e}")
        return jsonify({"error": f"Error creating mask_gray: {e}"}), 500

    try:
        print("Preparing human image for pose estimation network")
        human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
        human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
        print("Human image prepared")
    except Exception as e:
        print(f"Error preparing human image for pose network: {e}")
        return jsonify({"error": f"Error preparing human image for pose network: {e}"}), 500

    try:
        print("Running DensePose")
        args = apply_net.create_argument_parser().parse_args(
            ('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda')
        )
        pose_img = args.func(args, human_img_arg)
        pose_img = pose_img[:, :, ::-1]
        pose_img = Image.fromarray(pose_img).resize((768, 1024))
        print("DensePose completed")
    except Exception as e:
        print(f"Error during DensePose: {e}")
        return jsonify({"error": f"Error during DensePose: {e}"}), 500

    try:
        print("Entering inference mode")
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                with torch.no_grad():
                    print("Encoding prompts")
                    prompt = "model is wearing " + data['garment_description']
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
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
                    print("First prompt encoding complete")

                    prompt = "a photo of " + data['garment_description']
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                    if not isinstance(prompt, List):
                        prompt = [prompt] * 1
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * 1
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
                    print("Second prompt encoding complete")

                    pose_img_tensor = tensor_transfrom(pose_img).unsqueeze(0).to(device, torch.float16)
                    garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).to(device, torch.float16)
                    results = []
                    current_seed = data['seed']
                    print("Starting image generation loop")
                    for i in range(data['number_of_images']):
                        print(f"Generating image {i+1}/{data['number_of_images']}")
                        if data['is_randomize_seed']:
                            current_seed = torch.randint(0, 2**32, size=(1,)).item()
                        generator = torch.Generator(device).manual_seed(current_seed) if data['seed'] != -1 else None
                        current_seed = current_seed + i
                        print(f"Using seed: {current_seed}")

                        images = pipe(
                            prompt_embeds=prompt_embeds.to(device, torch.float16),
                            negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                            pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                            num_inference_steps=data['denoise_steps'],
                            generator=generator,
                            strength=1.0,
                            pose_img=pose_img_tensor.to(device, torch.float16),
                            text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                            cloth=garm_tensor.to(device, torch.float16),
                            mask_image=mask,
                            image=human_img,
                            height=1024,
                            width=768,
                            ip_adapter_image=garm_img.resize((768, 1024)),
                            guidance_scale=2.0,
                            dtype=torch.float16,
                            device=device,
                        )[0]
                        print("Image generated")
                        if data['is_checked_crop']:
                            print("Pasting on original image")
                            out_img = images[0].resize(crop_size)
                            human_img_orig.paste(out_img, (int(left), int(top)))
                            img_path = save_output_image(human_img_orig, base_path="outputs", base_filename='img', seed=current_seed)
                            results.append(img_path)
                            print(f"Image saved to: {img_path}")
                        else:
                            img_path = save_output_image(images[0], base_path="outputs", base_filename='img')
                            results.append(img_path)
                            print(f"Image saved to: {img_path}")
                    print("Image generation loop complete")
                    return jsonify({"results": results})
    except Exception as e:
        print(f"An error occurred during the main process: {e}")
        return jsonify({"error": f"An error occurred during processing: {e}"}), 500
    finally:
        print("Exiting tryon_image function")

@app.route("/check")
def check():
    return "API is running"

if __name__ == "__main__":
    # Load models only if not in debug mode's reloader
    if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        with app.app_context():
            load_models()
    app.run(host="0.0.0.0", port=5000, debug=False)
