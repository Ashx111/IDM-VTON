# --- Imports (keep all previous imports) ---
import torch
from flask import Flask, request, jsonify
import gc
import os
import base64
from io import BytesIO
from PIL import Image
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
import apply_net
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
# from pydantic import BaseModel # Keep if you were using it, removed for brevity here

app = Flask(__name__)

# --- Device Setup ---
if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
    print("Found 2 or more GPUs. Using cuda:0 for main pipeline and cuda:1 for auxiliary models.")
    device_0 = torch.device("cuda:0") # Main pipeline
    device_1 = torch.device("cuda:1") # Aux models (OpenPose, Parsing, DensePose)
elif torch.cuda.is_available():
    print("Found only one GPU. Placing all models on cuda:0. May encounter OOM.")
    device_0 = torch.device("cuda:0")
    device_1 = torch.device("cuda:0") # Fallback to GPU 0
else:
    print("CUDA not available. Using CPU.")
    device_0 = torch.device("cpu")
    device_1 = torch.device("cpu")

# --- Global variables for models ---
pipe = None
unet = None
UNet_Encoder = None
image_encoder = None # Add image_encoder here
vae = None           # Add vae here
parsing_model = None
openpose_model = None
tensor_transfrom = None
clip_image_processor = None # Add processor here

# --- Load Models Function (Modified for Multi-GPU) ---
def load_models():
    print("loading models using float16 across GPUs...")
    global pipe, unet, UNet_Encoder, image_encoder, vae, parsing_model, openpose_model, tensor_transfrom, clip_image_processor

    dtype = torch.float16
    fixed_vae = True

    model_id = 'yisol/IDM-VTON'
    vae_model_id = 'madebyollin/sdxl-vae-fp16-fix'

    # --- Load Main Pipeline Components onto device_0 ---
    print(f"Loading unet onto {device_0} with dtype: {dtype}")
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=dtype
    ).to(device_0)
    unet.requires_grad_(False)

    print(f"Loading image_encoder onto {device_0} with dtype: {dtype}")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        model_id, subfolder="image_encoder", torch_dtype=dtype
    ).to(device_0)
    image_encoder.requires_grad_(False)

    vae_load_dtype = dtype
    vae_source = vae_model_id if fixed_vae else model_id
    vae_subfolder = None if fixed_vae else "vae"
    print(f"Loading vae from '{vae_source}' onto {device_0} with dtype: {vae_load_dtype}")
    vae_params = {"torch_dtype": vae_load_dtype}
    if vae_subfolder: vae_params["subfolder"] = vae_subfolder
    vae = AutoencoderKL.from_pretrained(vae_source, **vae_params).to(device_0)
    vae.requires_grad_(False)

    print(f"Loading UNet_Encoder onto {device_0} with dtype: {dtype}")
    UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
        model_id, subfolder="unet_encoder", torch_dtype=dtype
    ).to(device_0)
    UNet_Encoder.requires_grad_(False)

    clip_image_processor = CLIPImageProcessor()

    pipe_param = {
        'pretrained_model_name_or_path': model_id, 'unet': unet, 'torch_dtype': dtype,
        'vae': vae, 'image_encoder': image_encoder, 'feature_extractor': clip_image_processor,
    }

    print(f"Initializing TryonPipeline on {device_0}...")
    # Initialize pipeline with components already on device_0
    pipe = TryonPipeline.from_pretrained(**pipe_param)
    pipe.unet_encoder = UNet_Encoder # Already on device_0
    # Ensure the entire pipe object structure is assigned to device_0
    pipe.to(device_0)
    print("Main pipeline components loaded on device_0.")

    # --- Load Auxiliary Models onto device_1 ---
    print(f"Loading parsing model onto {device_1}...")
    # Pass the device index (0 or 1) to the class constructor if supported
    parsing_model = Parsing(device_1.index if device_1.type == 'cuda' else -1) # Use index for CUDA, -1 for CPU

    print(f"Loading openpose model onto {device_1}...")
    # Pass the device index (0 or 1) to the class constructor if supported
    openpose_model = OpenPose(device_1.index if device_1.type == 'cuda' else -1) # Use index for CUDA, -1 for CPU

    # Explicitly move OpenPose sub-model to device_1
    try:
        if hasattr(openpose_model, 'preprocessor') and \
           hasattr(openpose_model.preprocessor, 'body_estimation') and \
           hasattr(openpose_model.preprocessor.body_estimation, 'model'):
            if openpose_model.preprocessor.body_estimation.model is not None:
                 openpose_model.preprocessor.body_estimation.model.to(device_1)
                 print(f"OpenPose sub-model moved to {device_1}.")
            else: print("Warning: OpenPose body estimation model is None.")
        else: print("Warning: Could not find expected path to OpenPose model.")
    except Exception as e: print(f"Warning: Error moving OpenPose model to {device_1}: {e}")

    # --- DensePose will be configured to run on device_1 within the endpoint ---
    print(f"DensePose will be configured to run on {device_1} during requests.")

    # --- Initialize Transforms (CPU operation) ---
    tensor_transfrom = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
    ])

    print("loading models completed!!!!!!!!!!!!")
    gc.collect()
    torch.cuda.empty_cache()


# --- Helper Function (keep as before) ---
def pil_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

# --- Tryon Endpoint (Modified for Multi-GPU) ---
@app.route('/tryon', methods=['POST'])
def tryon_image():
    global pipe, parsing_model, openpose_model, tensor_transfrom, clip_image_processor

    print("Entering tryon_image function")
    data = request.get_json()
    dtype = torch.float16 # Define dtype for endpoint usage

    # --- Image Loading (No changes needed here) ---
    try:
        human_image_data = base64.b64decode(data['human_image_base64'])
        human_image = Image.open(BytesIO(human_image_data)).convert("RGB")
        garment_image_data = base64.b64decode(data['garment_image_base64'])
        garment_image = Image.open(BytesIO(garment_image_data)).convert("RGB")
        print("Images decoded successfully.")
    except Exception as e:
        print(f"Error decoding images: {e}")
        return jsonify({"error": f"Error decoding images: {e}"}), 500

    # --- Image Preprocessing (No device changes yet) ---
    garm_img = garment_image.resize((768, 1024)) # Keep as PIL
    human_img_orig = human_image # Keep original PIL

    if data['is_checked_crop']:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2; top = (height - target_height) / 2
        right = (width + target_width) / 2; bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((768, 1024)) # Keep as PIL
        print("Human image cropped and resized")
    else:
        human_img = human_img_orig.resize((768, 1024)) # Keep as PIL
        crop_size = human_img.size # Use full size if not cropped
        left, top = 0, 0 # Define left/top for potential pasting later
        print("Human image resized")

    # --- Mask Generation (Using device_1) ---
    mask = None
    mask_gray = None
    try:
        human_img_resized_for_aux = human_img.resize((384, 512)) # PIL for aux models
        if data['is_checked']:
            print(f"Running OpenPose/Parsing on {device_1}")
            # Assuming OpenPose/Parsing models handle internal device placement based on init
            keypoints = openpose_model(human_img_resized_for_aux)
            print("OpenPose keypoints:", keypoints) # Debug
            model_parse, _ = parsing_model(human_img_resized_for_aux)
            print("Parsing model output:", model_parse) # Debug
            # get_mask_location likely returns PIL images
            mask, mask_gray = get_mask_location('hd', data['category'], model_parse, keypoints)
            mask = mask.resize((768, 1024)) # PIL Image
            mask_gray = mask_gray.resize((768, 1024)) # PIL Image
            print("Auto mask generated")
        else:
            # Fallback if using provided mask - Assuming input is part of human_image structure
            # This part might need adjustment based on how you'd provide a manual mask via API
            print("Using manual mask (logic might need API adjustment)")
            # Example: assuming a 'mask_image_base64' is provided
            if 'mask_image_base64' in data:
                 mask_data = base64.b64decode(data['mask_image_base64'])
                 mask = Image.open(BytesIO(mask_data)).convert("L").resize((768, 1024)) # Ensure L mode
            else:
                 # If no manual mask provided, generate a dummy black mask? Or raise error?
                 # For now, let's try a simple binary mask from the human image as fallback
                 mask = pil_to_binary_mask(human_img.convert("RGB").resize((768, 1024))) # PIL Image

            # Create mask_gray manually if needed
            mask_tensor_temp = transforms.ToTensor()(mask) # Temp tensor on CPU
            human_img_tensor_temp = tensor_transfrom(human_img) # Temp tensor on CPU
            mask_gray_tensor = (1.0 - mask_tensor_temp) * human_img_tensor_temp
            mask_gray = to_pil_image((mask_gray_tensor + 1.0) / 2.0) # Back to PIL
            print("Manual mask processed")

    except Exception as e:
        print(f"Error during pose/mask generation on {device_1}: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        return jsonify({"error": f"Error processing pose/mask: {e}"}), 500

    # --- DensePose (Using device_1) ---
    pose_img = None
    try:
        print(f"Preparing human image for DensePose on {device_1}")
        human_img_for_densepose = human_img.resize((384, 512)) # Use the already resized PIL
        human_img_arg = _apply_exif_orientation(human_img_for_densepose)
        human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR") # Numpy array on CPU

        print(f"Running DensePose on {device_1}")
        # Modify args to specify cuda:1
        densepose_device_str = f"{device_1.type}:{device_1.index}" if device_1.type == 'cuda' else 'cpu'
        args = apply_net.create_argument_parser().parse_args(
            ('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml',
             './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v',
             '--opts', 'MODEL.DEVICE', densepose_device_str) # Use variable here
        )
        # args.func likely handles moving data to the specified device (device_1)
        pose_output_np = args.func(args, human_img_arg) # Numpy output, likely on CPU after GPU processing
        # Convert result back to PIL Image
        pose_output_np = pose_output_np[:, :, ::-1] # BGR to RGB
        pose_img = Image.fromarray(pose_output_np).resize((768, 1024)) # PIL Image
        print("DensePose completed")
        del human_img_arg, pose_output_np, args # Clean up numpy arrays and args
        gc.collect()
        if device_1.type == 'cuda': torch.cuda.empty_cache() # Clear cache on device_1

    except Exception as e:
        print(f"Error during DensePose on {device_1}: {e}")
        # Check if it's an OOM error specifically
        if 'CUDA out of memory' in str(e):
            print(f"OOM Error occurred on {device_1} during DensePose.")
            # Potentially add more specific logging or handling here
        torch.cuda.empty_cache()
        gc.collect()
        return jsonify({"error": f"Error during DensePose: {e}"}), 500


    # --- Main Inference (Using device_0) ---
    results = []
    try:
        print(f"Entering inference mode on {device_0}")
        # Move necessary tensors to device_0
        # Pose image is PIL, tensor_transfrom makes it CPU tensor, then move to device_0
        pose_img_tensor = tensor_transfrom(pose_img).unsqueeze(0).to(device_0, dtype=dtype)
        # Garm image is PIL, tensor_transfrom makes it CPU tensor, then move to device_0
        garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).to(device_0, dtype=dtype)
        # Mask is PIL, pipe should handle it. human_img is PIL. ip_adapter_image is PIL.

        with torch.no_grad():
            # Prompts are processed by the pipe on device_0
            print("Encoding prompts on device_0")
            prompt = "model is wearing " + data['garment_description']
            neg_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds = pipe.encode_prompt(
                prompt, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=neg_prompt, device=device_0
            )

            prompt_c = "a photo of " + data['garment_description']
            prompt_embeds_c, _, _, _ = pipe.encode_prompt(
                prompt_c, num_images_per_prompt=1, do_classifier_free_guidance=False, negative_prompt=neg_prompt, device=device_0
            )
            print("Prompts encoded")

            current_seed = data['seed']
            print("Starting image generation loop")
            for i in range(data['number_of_images']):
                print(f"Generating image {i+1}/{data['number_of_images']} on {device_0}")
                if data['is_randomize_seed']:
                    current_seed = torch.randint(0, 2**32, (1,)).item()
                
                # Generator needs to be on the target device for the pipe
                generator = torch.Generator(device=device_0).manual_seed(current_seed) if current_seed != -1 else None
                
                seed_for_gen = current_seed + i # Use consistent seed increment logic
                if generator: generator.manual_seed(seed_for_gen) # Re-seed generator for loop
                print(f"Using seed: {seed_for_gen}")


                # --- Run the pipeline ---
                # Ensure all tensor inputs are on device_0 (checked above)
                # PIL inputs (mask_image, image, ip_adapter_image) are handled by the pipe
                images = pipe(
                    prompt_embeds=prompt_embeds, # Already on device_0
                    negative_prompt_embeds=neg_prompt_embeds, # Already on device_0
                    pooled_prompt_embeds=pooled_prompt_embeds, # Already on device_0
                    negative_pooled_prompt_embeds=neg_pooled_prompt_embeds, # Already on device_0
                    num_inference_steps=data['denoise_steps'],
                    generator=generator, # On device_0
                    strength=1.0,
                    pose_img=pose_img_tensor, # Explicitly moved to device_0
                    text_embeds_cloth=prompt_embeds_c, # Already on device_0
                    cloth=garm_tensor, # Explicitly moved to device_0
                    mask_image=mask,           # PIL Image
                    image=human_img,         # PIL Image
                    height=1024, width=768,
                    ip_adapter_image=garm_img, # PIL Image
                    guidance_scale=2.0,
                    # Dtype is implicitly float16 from pipe loading, autocast handles ops
                ).images # Access the .images attribute

                print("Image generated")
                output_image = images[0] # PIL Image output

                # --- Process Output ---
                if data['is_checked_crop']:
                    print("Pasting result onto original image")
                    out_img_resized = output_image.resize(crop_size)
                    # Create a copy to avoid modifying the original PIL object if it's reused
                    final_image = human_img_orig.copy()
                    final_image.paste(out_img_resized, (int(left), int(top)))
                    base64_image = pil_to_base64(final_image)
                else:
                    base64_image = pil_to_base64(output_image)

                results.append(base64_image)
                print("Image processed and converted to base64")

                # --- Clean up per-image ---
                del images, output_image, base64_image # Maybe final_image if created
                if 'final_image' in locals(): del final_image
                if 'out_img_resized' in locals(): del out_img_resized
                gc.collect()
                if device_0.type == 'cuda': torch.cuda.empty_cache()

            print("Image generation loop complete")
            # Clean up tensors used across loops
            del pose_img_tensor, garm_tensor, prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds, prompt_embeds_c
            return jsonify({"base64_images": results})

    except Exception as e:
        print(f"An error occurred during the main inference process on {device_0}: {e}")
        # Check if it's an OOM error specifically
        if 'CUDA out of memory' in str(e):
            print(f"OOM Error occurred on {device_0} during main inference.")
        return jsonify({"error": f"An error occurred during processing: {e}"}), 500
    finally:
        print("Exiting tryon_image function - Cleaning up")
        # General cleanup
        del mask, mask_gray, pose_img # Delete PIL images
        gc.collect()
        torch.cuda.empty_cache() # Clear cache on all GPUs

# --- Health Check Endpoint (keep as before) ---
@app.route("/check")
def check():
    return "API is running"

# --- Main Execution Block (Modified for Multi-GPU Check) ---
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available, running on CPU.")
    elif torch.cuda.device_count() < 2:
        print("WARNING: Fewer than 2 GPUs detected. Running on single GPU or CPU. OOM risk is higher.")

    # Load models using app context
    if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        with app.app_context():
            load_models() # Call the updated load_models function

    app.run(host="0.0.0.0", port=5000, debug=False) # Keep debug=False
