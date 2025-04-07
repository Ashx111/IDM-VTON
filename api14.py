import torch
from flask import Flask, request, jsonify
import gc
import os
import base64
from io import BytesIO
from PIL import Image
from torchvision import transforms
# ... other necessary imports ...
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
import apply_net
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image

app = Flask(__name__)

# --- Device Setup (Single GPU) ---
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"Using single GPU: {device}")
    # Set default device - might help aux models if they don't specify device
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")


# --- Global variables ---
pipe = None
unet = None
UNet_Encoder = None
image_encoder = None
vae = None
parsing_model = None
openpose_model = None
tensor_transfrom = None
clip_image_processor = None
# Densepose model isn't loaded globally, args are created per request

# --- Load Models Function (Single GPU Float16) ---
def load_models():
    print(f"loading models onto single device {device} using float16...")
    global pipe, unet, UNet_Encoder, image_encoder, vae, parsing_model, openpose_model, tensor_transfrom, clip_image_processor

    dtype = torch.float16
    fixed_vae = True

    model_id = 'yisol/IDM-VTON'
    vae_model_id = 'madebyollin/sdxl-vae-fp16-fix'

    # --- Load ALL Models onto the single device ---
    print(f"Loading unet...")
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=dtype
    ).to(device)
    unet.requires_grad_(False)

    print(f"Loading image_encoder...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        model_id, subfolder="image_encoder", torch_dtype=dtype
    ).to(device)
    image_encoder.requires_grad_(False)

    vae_load_dtype = dtype
    vae_source = vae_model_id if fixed_vae else model_id
    vae_subfolder = None if fixed_vae else "vae"
    print(f"Loading vae from '{vae_source}'...")
    vae_params = {"torch_dtype": vae_load_dtype}
    if vae_subfolder: vae_params["subfolder"] = vae_subfolder
    vae = AutoencoderKL.from_pretrained(vae_source, **vae_params).to(device)
    vae.requires_grad_(False)

    print(f"Loading UNet_Encoder...")
    UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
        model_id, subfolder="unet_encoder", torch_dtype=dtype
    ).to(device)
    UNet_Encoder.requires_grad_(False)

    clip_image_processor = CLIPImageProcessor()

    pipe_param = {
        'pretrained_model_name_or_path': model_id, 'unet': unet, 'torch_dtype': dtype,
        'vae': vae, 'image_encoder': image_encoder, 'feature_extractor': clip_image_processor,
    }

    print(f"Initializing TryonPipeline...")
    pipe = TryonPipeline.from_pretrained(**pipe_param)
    pipe.unet_encoder = UNet_Encoder
    pipe.to(device) # Ensure whole pipe is on the device

    # --- Load Auxiliary Models onto the SAME device ---
    print(f"Loading parsing model...")
    # Use index 0 if CUDA, -1 if CPU
    parsing_device_idx = device.index if device.type == 'cuda' else -1
    parsing_model = Parsing(parsing_device_idx)

    print(f"Loading openpose model...")
    openpose_device_idx = device.index if device.type == 'cuda' else -1
    openpose_model = OpenPose(openpose_device_idx)

    try: # Move OpenPose sub-model explicitly
        if hasattr(openpose_model, 'preprocessor.body_estimation.model') and \
           openpose_model.preprocessor.body_estimation.model is not None:
                 openpose_model.preprocessor.body_estimation.model.to(device)
                 print(f"OpenPose sub-model moved to {device}.")
        else: print("Warning: Could not find/move OpenPose sub-model.")
    except Exception as e: print(f"Warning: Error moving OpenPose model: {e}")

    # DensePose is loaded/run per request

    tensor_transfrom = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
    ])

    print("loading models completed!!!!!!!!!!!!")
    gc.collect()
    if device.type == 'cuda': torch.cuda.empty_cache()


# --- Helper Function (keep as before) ---
def pil_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

# --- Tryon Endpoint (Single GPU Float16 w/ Cleanup) ---
@app.route('/tryon', methods=['POST'])
def tryon_image():
    # Use global models placed on 'device'
    global pipe, parsing_model, openpose_model, tensor_transfrom

    print("Entering tryon_image function (Single GPU)")
    data = request.get_json()
    dtype = torch.float16
    output_images_base64 = [] # Use a different name

    # --- Define variables accessed in finally ---
    human_img_tensor_temp = None
    mask_tensor_temp = None
    mask_gray_tensor = None
    mask = None
    mask_gray = None
    keypoints_result = None
    parse_result_pil = None
    human_img_arg = None
    densepose_args = None
    pose_output_np = None
    pose_img = None
    pose_img_tensor = None
    garm_tensor = None
    prompt_embeds = None
    neg_prompt_embeds = None
    pooled_prompt_embeds = None
    neg_pooled_prompt_embeds = None
    prompt_embeds_c = None
    images = None

    try: # Master try block
        # --- Image Loading ---
        try:
            human_image_data = base64.b64decode(data['human_image_base64'])
            human_image = Image.open(BytesIO(human_image_data)).convert("RGB")
            garment_image_data = base64.b64decode(data['garment_image_base64'])
            garment_image = Image.open(BytesIO(garment_image_data)).convert("RGB")
            print("Images decoded successfully.")
            del human_image_data, garment_image_data # Cleanup
        except Exception as e:
            print(f"Error decoding images: {e}")
            return jsonify({"error": f"Error decoding images: {e}"}), 500

        # --- Image Preprocessing ---
        garm_img = garment_image.resize((768, 1024)) # PIL
        human_img_orig = human_image # PIL

        if data['is_checked_crop']:
            width, height = human_img_orig.size
            target_width = int(min(width, height * (3 / 4)))
            target_height = int(min(height, width * (4 / 3)))
            left = (width - target_width) / 2; top = (height - target_height) / 2
            right = (width + target_width) / 2; bottom = (height + target_height) / 2
            cropped_img = human_img_orig.crop((left, top, right, bottom))
            crop_size = cropped_img.size
            human_img = cropped_img.resize((768, 1024)) # PIL
            print("Human image cropped and resized")
            del cropped_img # Cleanup
        else:
            human_img = human_img_orig.resize((768, 1024)) # PIL
            crop_size = human_img.size
            left, top = 0, 0
            print("Human image resized")

        # --- Mask Generation ---
        try:
            human_img_resized_for_aux_pil = human_img.resize((384, 512)) # PIL
            if data['is_checked']:
                print(f"Running OpenPose/Parsing on {device}")
                # Models are already on 'device', input is PIL
                with torch.no_grad():
                     if openpose_model is None or parsing_model is None:
                         raise ValueError("OpenPose or Parsing model not loaded.")
                     keypoints_result = openpose_model(human_img_resized_for_aux_pil)
                     parse_result_pil, _ = parsing_model(human_img_resized_for_aux_pil)

                if not (isinstance(keypoints_result, dict) and 'pose_keypoints_2d' in keypoints_result):
                     raise ValueError("OpenPose did not return expected keypoints format.")

                mask, mask_gray = get_mask_location('hd', data['category'], parse_result_pil, keypoints_result)
                mask = mask.resize((768, 1024)) # PIL
                mask_gray = mask_gray.resize((768, 1024)) # PIL
                print("Auto mask generated.")
                # del keypoints_result, parse_result_pil # Cleanup moved to finally
            else:
                # Manual mask logic
                print("Using manual mask...")
                if 'mask_image_base64' in data:
                     mask_data = base64.b64decode(data['mask_image_base64'])
                     mask = Image.open(BytesIO(mask_data)).convert("L").resize((768, 1024))
                     del mask_data
                else:
                     mask = pil_to_binary_mask(human_img.convert("RGB").resize((768, 1024)))

                mask_tensor_temp = transforms.ToTensor()(mask) # CPU
                human_img_tensor_temp = tensor_transfrom(human_img) # CPU [-1,1]
                mask_gray_tensor = (1.0 - mask_tensor_temp) * human_img_tensor_temp # CPU
                mask_gray = to_pil_image((mask_gray_tensor + 1.0) / 2.0) # PIL [0,255]
                print("Manual mask processed.")
                # del mask_tensor_temp, human_img_tensor_temp, mask_gray_tensor # Cleanup moved to finally

            del human_img_resized_for_aux_pil # Cleanup

        except Exception as e:
            print(f"Error during pose/mask generation phase: {e}")
            return jsonify({"error": f"Error processing pose/mask: {e}"}), 500


        # --- DensePose ---
        try:
            human_img_for_densepose = human_img.resize((384, 512)) # PIL
            human_img_arg = _apply_exif_orientation(human_img_for_densepose)
            human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR") # Numpy CPU

            print(f"Running DensePose on {device}")
            densepose_device_str = f"{device.type}:{device.index}" if device.type == 'cuda' else 'cpu'
            densepose_model_path = './ckpt/densepose/model_final_162be9.pkl'
            if not os.path.exists(densepose_model_path): raise FileNotFoundError(f"DensePose model not found: {densepose_model_path}")

            densepose_args = apply_net.create_argument_parser().parse_args(
                ('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml',
                 densepose_model_path, 'dp_segm', '-v',
                 '--opts', 'MODEL.DEVICE', densepose_device_str)
            )
            # Run DensePose - it should handle internal device placement
            # Expect numpy output on CPU
            pose_output_np = densepose_args.func(densepose_args, human_img_arg)

            pose_output_np = pose_output_np[:, :, ::-1] # BGR to RGB
            pose_img = Image.fromarray(pose_output_np).resize((768, 1024)) # PIL Image
            print("DensePose completed")
            # del human_img_arg, densepose_args, pose_output_np # Cleanup moved to finally
            del human_img_for_densepose

        except Exception as e:
            print(f"Error during DensePose on {device}: {e}")
            # Check if OOM before returning
            if 'CUDA out of memory' in str(e):
                 print("OOM Error during DensePose detected.")
                 # Consider returning a specific OOM error message
                 return jsonify({"error": f"CUDA out of memory during DensePose: {e}"}), 500
            return jsonify({"error": f"Error during DensePose: {e}"}), 500
        finally:
            # Crucial: Clear cache immediately after DensePose if on CUDA
             if device.type == 'cuda': torch.cuda.empty_cache()


        # --- Main Inference ---
        try:
            print(f"Entering inference mode on {device}")
            # Convert PIL to Tensor and move to device
            pose_img_tensor = tensor_transfrom(pose_img).unsqueeze(0).to(device, dtype=dtype)
            garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).to(device, dtype=dtype)

            with torch.no_grad():
                print("Encoding prompts...")
                prompt = "model is wearing " + data['garment_description']
                neg_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds = pipe.encode_prompt(
                    prompt, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=neg_prompt # Device handled by pipe
                )
                prompt_c = "a photo of " + data['garment_description']
                prompt_embeds_c, _, _, _ = pipe.encode_prompt(
                    prompt_c, num_images_per_prompt=1, do_classifier_free_guidance=False, negative_prompt=neg_prompt # Device handled by pipe
                )
                print("Prompts encoded")

                current_seed = data['seed']
                print("Starting image generation loop")
                for i in range(data['number_of_images']):
                    print(f"Generating image {i+1}/{data['number_of_images']}")
                    if data['is_randomize_seed']:
                        current_seed = torch.randint(0, 2**32, (1,)).item()

                    generator = torch.Generator(device=device).manual_seed(current_seed) if current_seed != -1 else None
                    seed_for_gen = current_seed + i
                    if generator: generator.manual_seed(seed_for_gen)
                    print(f"Using seed: {seed_for_gen}")

                    # Ensure pipe components are still on the device (should be)
                    images = pipe(
                         prompt_embeds=prompt_embeds, negative_prompt_embeds=neg_prompt_embeds,
                         pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=neg_pooled_prompt_embeds,
                         num_inference_steps=data['denoise_steps'], generator=generator, strength=1.0,
                         pose_img=pose_img_tensor, text_embeds_cloth=prompt_embeds_c, cloth=garm_tensor,
                         mask_image=mask, image=human_img, height=1024, width=768,
                         ip_adapter_image=garm_img, guidance_scale=2.0,
                     ).images
                    print("Image generated")
                    output_image = images[0] # PIL Image

                    if data['is_checked_crop']:
                        print("Pasting result onto original image")
                        out_img_resized = output_image.resize(crop_size)
                        final_image = human_img_orig.copy()
                        final_image.paste(out_img_resized, (int(left), int(top)))
                        base64_image = pil_to_base64(final_image)
                        del out_img_resized, final_image # Cleanup
                    else:
                        base64_image = pil_to_base64(output_image)

                    output_images_base64.append(base64_image)
                    print("Image processed and converted to base64")

                    # Cleanup per image
                    del images, output_image, base64_image
                    gc.collect()
                    if device.type == 'cuda': torch.cuda.empty_cache()

                print("Image generation loop complete")

        except Exception as e:
            print(f"An error occurred during the main inference process: {e}")
            if 'CUDA out of memory' in str(e): print(f"OOM Error during main inference.")
            return jsonify({"error": f"An error occurred during processing: {e}"}), 500

        # --- Return Success ---
        return jsonify({"base64_images": output_images_base64})

    # --- Master Finally Block for Cleanup ---
    finally:
        print("Exiting tryon_image function - Cleaning up intermediate data")
        # Delete potentially large objects
        del human_img_tensor_temp, mask_tensor_temp, mask_gray_tensor
        del mask, mask_gray, keypoints_result, parse_result_pil
        del human_img_arg, densepose_args, pose_output_np, pose_img
        del pose_img_tensor, garm_tensor
        del prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds, prompt_embeds_c
        if 'images' in locals() and images is not None: del images # If loop failed early

        # Call garbage collector and empty cache one last time
        collected = gc.collect()
        print(f"Garbage collector collected {collected} items.")
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            print("CUDA cache cleared.")


# --- Health Check Endpoint ---
@app.route("/check")
def check():
    return "API is running"

# --- Main Execution Block ---
if __name__ == "__main__":
    if not torch.cuda.is_available(): print("WARNING: CUDA is not available.")

    if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        with app.app_context():
            load_models() # Load models onto single device

    app.run(host="0.0.0.0", port=5000, debug=False)
