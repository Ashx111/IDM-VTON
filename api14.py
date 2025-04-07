# --- Imports ---
import torch
from flask import Flask, request, jsonify
import gc
import os
import base64
from io import BytesIO
from PIL import Image
from torchvision import transforms
import apply_net
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
import traceback # For better error logging

# --- Model/Pipeline Imports (Ensure these are correct!) ---
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel # Make sure this import is present
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)
from diffusers import AutoencoderKL
# --- Aux Model Imports ---
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
# --- Utility Imports ---
from utils_mask import get_mask_location
from util.image import pil_to_binary_mask


app = Flask(__name__)

# --- Device Setup ---
if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
    print("Found 2 or more GPUs. Using cuda:0 for main pipeline and cuda:1 for on-demand auxiliary models.")
    device_0 = torch.device("cuda:0") # Main pipeline
    device_1 = torch.device("cuda:1") # Aux models (loaded on demand)
elif torch.cuda.is_available():
    print("Found only one GPU. Placing all models on cuda:0. High OOM risk.")
    device_0 = torch.device("cuda:0")
    device_1 = torch.device("cuda:0") # Fallback to GPU 0
else:
    print("CUDA not available. Using CPU.")
    device_0 = torch.device("cpu")
    device_1 = torch.device("cpu")

# --- Global variables for PERMANENT models (on device_0) ---
pipe = None
unet = None
UNet_Encoder = None
image_encoder = None
vae = None
tensor_transfrom = None # Defined in load_models
clip_image_processor = None # Defined in load_models

# --- Load Models Function (Load only main pipeline on device_0) ---
def load_models():
    print(f"Loading main pipeline components onto {device_0} using float16...")
    global pipe, unet, UNet_Encoder, image_encoder, vae, tensor_transfrom, clip_image_processor

    # Only load components that stay persistent on device_0
    dtype = torch.float16
    fixed_vae = True
    model_id = 'yisol/IDM-VTON'
    vae_model_id = 'madebyollin/sdxl-vae-fp16-fix'

    print(f"Loading unet...")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype).to(device_0)
    unet.requires_grad_(False)

    print(f"Loading image_encoder...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=dtype).to(device_0)
    image_encoder.requires_grad_(False)

    vae_load_dtype = dtype
    vae_source = vae_model_id if fixed_vae else model_id
    vae_subfolder = None if fixed_vae else "vae"
    print(f"Loading vae from '{vae_source}'...")
    vae_params = {"torch_dtype": vae_load_dtype}
    if vae_subfolder: vae_params["subfolder"] = vae_subfolder
    vae = AutoencoderKL.from_pretrained(vae_source, **vae_params).to(device_0)
    vae.requires_grad_(False)

    print(f"Loading UNet_Encoder...")
    UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(model_id, subfolder="unet_encoder", torch_dtype=dtype).to(device_0)
    UNet_Encoder.requires_grad_(False)

    clip_image_processor = CLIPImageProcessor()

    pipe_param = {
        'pretrained_model_name_or_path': model_id, 'unet': unet, 'torch_dtype': dtype,
        'vae': vae, 'image_encoder': image_encoder, 'feature_extractor': clip_image_processor,
    }
    print(f"Initializing TryonPipeline...")
    pipe = TryonPipeline.from_pretrained(**pipe_param)
    pipe.unet_encoder = UNet_Encoder
    pipe.to(device_0) # Ensure whole pipe is on device_0

    # --- DO NOT load Parsing, OpenPose, or DensePose here ---
    print("Auxiliary models (Parsing, OpenPose, DensePose) will be loaded on demand.")

    tensor_transfrom = transforms.Compose([ # Define transform for later use
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
    ])

    print("Main pipeline loading completed!!!!!!!!!!!!")
    gc.collect()
    if device_0.type == 'cuda': torch.cuda.empty_cache()


# --- Helper Function ---
def pil_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

# --- Tryon Endpoint (Multi-GPU On-Demand Loading) ---
@app.route('/tryon', methods=['POST'])
def tryon_image():
    # Access global pipeline components on device_0
    global pipe, tensor_transfrom
    print(f"Entering tryon_image: Main pipe on {device_0}, Aux on {device_1} (on-demand)")
    data = request.get_json()
    dtype = torch.float16
    output_images_base64 = []

    # --- Intermediate Data Variables (defined for cleanup) ---
    mask = None; mask_gray = None; pose_img = None;
    human_img = None; garm_img = None; human_img_orig = None;

    try: # Master try block for overall request
        # --- 1. Image Loading & Initial Processing (CPU) ---
        try:
            human_image_data = base64.b64decode(data['human_image_base64'])
            human_image_pil = Image.open(BytesIO(human_image_data)).convert("RGB")
            garment_image_data = base64.b64decode(data['garment_image_base64'])
            garment_image_pil = Image.open(BytesIO(garment_image_data)).convert("RGB")
            print("Images decoded successfully.")
            del human_image_data, garment_image_data
        except Exception as e:
            print(f"Error decoding images: {e}")
            return jsonify({"error": f"Error decoding images: {e}"}), 500

        garm_img = garment_image_pil.resize((768, 1024)) # PIL for pipe
        human_img_orig = human_image_pil # Keep original PIL

        if data['is_checked_crop']:
            # ... crop logic ...
            width, height = human_img_orig.size
            target_width = int(min(width, height * (3 / 4)))
            target_height = int(min(height, width * (4 / 3)))
            left = (width - target_width) / 2; top = (height - target_height) / 2
            right = (width + target_width) / 2; bottom = (height + target_height) / 2
            cropped_img = human_img_orig.crop((left, top, right, bottom))
            crop_size = cropped_img.size
            human_img = cropped_img.resize((768, 1024)) # PIL for pipe
            print("Human image cropped and resized")
            del cropped_img
        else:
            human_img = human_img_orig.resize((768, 1024)) # PIL for pipe
            crop_size = human_img.size
            left, top = 0, 0
            print("Human image resized")

        # --- 2. Mask Generation (On-Demand on device_1) ---
        openpose_model = None # Ensure cleanup even if loading fails
        parsing_model = None
        keypoints_result = None
        parse_result_pil = None
        input_tensor_aux = None
        human_img_resized_for_aux_pil = human_img.resize((384, 512)) # PIL Input

        if data['is_checked']: # Only load if needed
            print(f"--- Loading OpenPose/Parsing models onto {device_1} ---")
            try:
                parsing_device_idx = device_1.index if device_1.type == 'cuda' else -1
                openpose_device_idx = device_1.index if device_1.type == 'cuda' else -1

                openpose_model = OpenPose(openpose_device_idx)
                parsing_model = Parsing(parsing_device_idx)

                # Attempt explicit move AFTER init
                if device_1.type == 'cuda':
                    if hasattr(openpose_model, 'preprocessor.body_estimation.model') and \
                       openpose_model.preprocessor.body_estimation.model is not None:
                        openpose_model.preprocessor.body_estimation.model.to(device_1)
                        print("Moved OpenPose sub-model to", device_1)
                    else:
                        print("Warning: Could not move OpenPose sub-model, trying top-level.")
                        openpose_model.to(device_1)

                    if hasattr(parsing_model, 'model'):
                        parsing_model.model.to(device_1)
                        print("Moved Parsing model to", device_1)
                    else:
                        print("Warning: Could not move Parsing sub-model, trying top-level.")
                        parsing_model.to(device_1)

                # Prepare input tensor explicitly ON device_1
                simple_to_tensor = transforms.ToTensor()
                input_tensor_aux = simple_to_tensor(human_img_resized_for_aux_pil).unsqueeze(0).to(device_1)
                print(f"Input tensor for Aux created on {input_tensor_aux.device}")

                # Run models
                with torch.no_grad():
                    keypoints_result = openpose_model(input_tensor_aux) # Pass tensor
                    parse_result_tensor, _ = parsing_model(input_tensor_aux) # Pass tensor

                # Process results (move parse tensor to CPU)
                parse_result_pil = transforms.ToPILImage()(parse_result_tensor.squeeze(0).cpu())
                del parse_result_tensor # Cleanup GPU tensor

                if not (isinstance(keypoints_result, dict) and 'pose_keypoints_2d' in keypoints_result):
                    raise ValueError(f"OpenPose didn't return expected dict. Got: {type(keypoints_result)}")

                mask, mask_gray = get_mask_location('hd', data['category'], parse_result_pil, keypoints_result)
                mask = mask.resize((768, 1024)) # PIL
                mask_gray = mask_gray.resize((768, 1024)) # PIL
                print("Auto mask generated.")

            except Exception as e:
                print(f"!!! Error during On-Demand Aux model execution on {device_1}: {e}")
                print(traceback.format_exc()) # Print full traceback
                return jsonify({"error": f"Error during Auto-Masking: {e}"}), 500
            finally:
                # *** Crucial: Cleanup Aux models from device_1 ***
                print(f"--- Cleaning up OpenPose/Parsing models from {device_1} ---")
                del keypoints_result, parse_result_pil, input_tensor_aux # Delete data first
                del openpose_model, parsing_model # Delete model objects
                gc.collect()
                if device_1.type == 'cuda':
                    print(f"Clearing CUDA cache for {device_1}")
                    torch.cuda.empty_cache() # Clear cache specifically for device_1 if possible (may clear all)
        else:
             # Manual mask logic (CPU/PIL based)
            print("Using manual mask...")
            # ... (manual mask logic as before) ...
            if 'mask_image_base64' in data:
                 mask_data = base64.b64decode(data['mask_image_base64'])
                 mask = Image.open(BytesIO(mask_data)).convert("L").resize((768, 1024))
                 del mask_data
            else:
                 mask = pil_to_binary_mask(human_img.convert("RGB").resize((768, 1024))) # PIL

            mask_tensor_temp = transforms.ToTensor()(mask) # CPU
            human_img_tensor_temp = tensor_transfrom(human_img) # CPU [-1,1]
            mask_gray_tensor = (1.0 - mask_tensor_temp) * human_img_tensor_temp # CPU
            mask_gray = to_pil_image((mask_gray_tensor + 1.0) / 2.0) # PIL [0,255]
            del mask_tensor_temp, human_img_tensor_temp, mask_gray_tensor # Cleanup
            print("Manual mask processed.")

        del human_img_resized_for_aux_pil # Cleanup input PIL


        # --- 3. DensePose Generation (On-Demand/Args on device_1) ---
        human_img_arg = None
        densepose_args = None
        pose_output_np = None
        try:
            print(f"--- Preparing for DensePose on {device_1} ---")
            human_img_for_densepose = human_img.resize((384, 512)) # PIL
            human_img_arg = _apply_exif_orientation(human_img_for_densepose)
            human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR") # Numpy CPU
            del human_img_for_densepose

            densepose_device_str = f"{device_1.type}:{device_1.index}" if device_1.type == 'cuda' else 'cpu'
            densepose_config = './configs/densepose_rcnn_R_50_FPN_s1x.yaml'
            densepose_model_path = './ckpt/densepose/model_final_162be9.pkl'
            if not os.path.exists(densepose_model_path): raise FileNotFoundError(f"DensePose model not found: {densepose_model_path}")
            if not os.path.exists(densepose_config): raise FileNotFoundError(f"DensePose config not found: {densepose_config}")

            print(f"Running DensePose via apply_net targeting {densepose_device_str}")
            densepose_args = apply_net.create_argument_parser().parse_args(
                ('show', densepose_config, densepose_model_path, 'dp_segm', '-v',
                 '--opts', 'MODEL.DEVICE', densepose_device_str)
            )
            # apply_net loads the model internally based on args
            pose_output_np = densepose_args.func(densepose_args, human_img_arg) # Should run on device_1 internally

            pose_output_np = pose_output_np[:, :, ::-1] # BGR to RGB
            pose_img = Image.fromarray(pose_output_np).resize((768, 1024)) # PIL Image
            print("DensePose completed.")

        except Exception as e:
            print(f"!!! Error during DensePose execution targeting {device_1}: {e}")
            print(traceback.format_exc())
            # Check if OOM before returning
            if 'CUDA out of memory' in str(e): print("OOM Error during DensePose detected.")
            return jsonify({"error": f"Error during DensePose: {e}"}), 500
        finally:
             # *** Crucial: Cleanup DensePose resources from device_1 ***
             print(f"--- Cleaning up DensePose resources from {device_1} ---")
             del human_img_arg, densepose_args, pose_output_np # Delete intermediate data
             # We can't explicitly delete the model loaded by apply_net unless it returns it
             gc.collect()
             if device_1.type == 'cuda':
                 print(f"Clearing CUDA cache for {device_1}")
                 torch.cuda.empty_cache()


        # --- 4. Main Inference (on device_0) ---
        pose_img_tensor = None
        garm_tensor = None
        prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds, prompt_embeds_c = None, None, None, None, None
        images = None
        try:
            print(f"--- Entering main inference on {device_0} ---")
            if pose_img is None: raise ValueError("Pose image is missing before main inference.")
            if mask is None: raise ValueError("Mask image is missing before main inference.")
            if garm_img is None: raise ValueError("Garment image is missing.")
            if human_img is None: raise ValueError("Human image is missing.")

            # Move required tensors to device_0
            pose_img_tensor = tensor_transfrom(pose_img).unsqueeze(0).to(device_0, dtype=dtype)
            garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).to(device_0, dtype=dtype)
            print(f"Pose tensor on {pose_img_tensor.device}, Garment tensor on {garm_tensor.device}")

            with torch.no_grad():
                # Prompt Encoding (on device_0 via pipe)
                print("Encoding prompts...")
                # ... prompt encoding logic ...
                prompt = "model is wearing " + data['garment_description']
                neg_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds = pipe.encode_prompt(
                    prompt, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=neg_prompt
                )
                prompt_c = "a photo of " + data['garment_description']
                prompt_embeds_c, _, _, _ = pipe.encode_prompt(
                    prompt_c, num_images_per_prompt=1, do_classifier_free_guidance=False, negative_prompt=neg_prompt
                )
                print("Prompts encoded.")

                # Generation Loop
                current_seed = data['seed']
                print("Starting image generation loop")
                for i in range(data['number_of_images']):
                    print(f"Generating image {i+1}/{data['number_of_images']} on {device_0}")
                    # ... seed and generator logic ...
                    if data['is_randomize_seed']: current_seed = torch.randint(0, 2**32, (1,)).item()
                    generator = torch.Generator(device=device_0).manual_seed(current_seed) if current_seed != -1 else None
                    seed_for_gen = current_seed + i
                    if generator: generator.manual_seed(seed_for_gen)
                    print(f"Using seed: {seed_for_gen}")


                    # Main pipe call
                    images = pipe(
                         prompt_embeds=prompt_embeds.to(device_0), # Ensure on device_0
                         negative_prompt_embeds=neg_prompt_embeds.to(device_0),
                         pooled_prompt_embeds=pooled_prompt_embeds.to(device_0),
                         negative_pooled_prompt_embeds=neg_pooled_prompt_embeds.to(device_0),
                         num_inference_steps=data['denoise_steps'], generator=generator, strength=1.0,
                         pose_img=pose_img_tensor.to(device_0), # Ensure on device_0
                         text_embeds_cloth=prompt_embeds_c.to(device_0),
                         cloth=garm_tensor.to(device_0),
                         mask_image=mask, # PIL
                         image=human_img, # PIL
                         height=1024, width=768,
                         ip_adapter_image=garm_img, # PIL
                         guidance_scale=2.0,
                     ).images
                    print("Image generated")
                    output_image = images[0] # PIL

                    # Process output
                    if data['is_checked_crop']:
                         # ... crop pasting ...
                        print("Pasting result onto original image")
                        out_img_resized = output_image.resize(crop_size)
                        final_image = human_img_orig.copy()
                        final_image.paste(out_img_resized, (int(left), int(top)))
                        base64_image = pil_to_base64(final_image)
                        del out_img_resized, final_image
                    else:
                         base64_image = pil_to_base64(output_image)

                    output_images_base64.append(base64_image)
                    print("Image processed and converted to base64")

                    # Cleanup per image
                    del images, output_image, base64_image
                    gc.collect()
                    if device_0.type == 'cuda': torch.cuda.empty_cache()

                print("Image generation loop complete")

        except Exception as e:
            print(f"!!! Error during main inference on {device_0}: {e}")
            print(traceback.format_exc())
            if "LayerNormKernelImpl" in str(e): print("!!! LayerNorm error occurred.")
            if 'CUDA out of memory' in str(e): print(f"OOM Error during main inference.")
            return jsonify({"error": f"An error occurred during main processing: {e}"}), 500
        finally:
            # Cleanup main inference tensors
            print(f"--- Cleaning up main inference resources from {device_0} ---")
            del pose_img_tensor, garm_tensor
            del prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds, prompt_embeds_c
            if 'images' in locals() and images is not None: del images
            gc.collect()
            if device_0.type == 'cuda': torch.cuda.empty_cache()

        # --- 5. Return Success ---
        print("Request completed successfully.")
        return jsonify({"base64_images": output_images_base64})

    # --- Master Finally Block ---
    finally:
        print("--- Final cleanup for request ---")
        # Delete PIL images and other potential leftovers
        del mask, mask_gray, pose_img, human_img, garm_img, human_img_orig
        # Ensure aux models are deleted if error happened before their finally block
        if 'openpose_model' in locals() and openpose_model is not None: del openpose_model
        if 'parsing_model' in locals() and parsing_model is not None: del parsing_model
        collected = gc.collect()
        print(f"Garbage collector collected {collected} items.")
        # Clear cache on all devices just in case
        if torch.cuda.is_available(): torch.cuda.empty_cache()


# --- Health Check & Main Execution ---
@app.route("/check")
def check(): return "API is running"

if __name__ == "__main__":
    # Ensure necessary directories/files exist
    os.makedirs("./outputs", exist_ok=True) # Example if needed
    # Add checks for essential model/config files here if desired

    if not torch.cuda.is_available(): print("WARNING: CUDA is not available.")
    elif torch.cuda.device_count() < 2: print("WARNING: Fewer than 2 GPUs detected, multi-GPU code will fallback.")

    if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        with app.app_context():
            load_models() # Load only main pipeline

    app.run(host="0.0.0.0", port=5000, debug=False) # debug=False is important
