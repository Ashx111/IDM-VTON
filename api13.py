# --- Imports (keep all previous imports) ---
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

# --- Device Setup (keep as before) ---
if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
    print("Found 2 or more GPUs. Using cuda:0 for main pipeline and cuda:1 for auxiliary models.")
    device_0 = torch.device("cuda:0")
    device_1 = torch.device("cuda:1")
elif torch.cuda.is_available():
    print("Found only one GPU. Placing all models on cuda:0.")
    device_0 = torch.device("cuda:0")
    device_1 = torch.device("cuda:0")
else:
    print("CUDA not available. Using CPU.")
    device_0 = torch.device("cpu")
    device_1 = torch.device("cpu")

# --- Global variables (keep as before) ---
pipe = None
unet = None
UNet_Encoder = None
image_encoder = None
vae = None
parsing_model = None
openpose_model = None
tensor_transfrom = None
clip_image_processor = None

# --- load_models function (keep the multi-GPU version from previous step) ---
def load_models():
    # --- Use the Multi-GPU load_models function provided previously ---
    # --- It should place pipe components on device_0 ---
    # --- and Parsing/OpenPose models on device_1 ---
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
    pipe = TryonPipeline.from_pretrained(**pipe_param)
    pipe.unet_encoder = UNet_Encoder
    pipe.to(device_0)
    print("Main pipeline components loaded on device_0.")

    # --- Load Auxiliary Models onto device_1 ---
    print(f"Loading parsing model onto {device_1}...")
    parsing_model = Parsing(device_1.index if device_1.type == 'cuda' else -1)

    print(f"Loading openpose model onto {device_1}...")
    openpose_model = OpenPose(device_1.index if device_1.type == 'cuda' else -1)

    try: # Move OpenPose sub-model explicitly
        if hasattr(openpose_model, 'preprocessor') and \
           hasattr(openpose_model.preprocessor, 'body_estimation') and \
           hasattr(openpose_model.preprocessor.body_estimation, 'model'):
            if openpose_model.preprocessor.body_estimation.model is not None:
                 openpose_model.preprocessor.body_estimation.model.to(device_1)
                 print(f"OpenPose sub-model moved to {device_1}.")
            else: print("Warning: OpenPose body estimation model is None.")
        else: print("Warning: Could not find expected path to OpenPose model.")
    except Exception as e: print(f"Warning: Error moving OpenPose model to {device_1}: {e}")

    print(f"DensePose will be configured to run on {device_1} during requests.")

    tensor_transfrom = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
    ])

    print("loading models completed!!!!!!!!!!!!")
    gc.collect()
    torch.cuda.empty_cache()


# --- Helper Function (keep as before) ---
def pil_to_base64(image):
    # ... (implementation) ...
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


# --- Tryon Endpoint (Modified for Explicit Input Tensor Device) ---
@app.route('/tryon', methods=['POST'])
def tryon_image():
    global pipe, parsing_model, openpose_model, tensor_transfrom, clip_image_processor

    print("Entering tryon_image function")
    data = request.get_json()
    dtype = torch.float16

    # --- Image Loading (No changes) ---
    try:
        human_image_data = base64.b64decode(data['human_image_base64'])
        human_image = Image.open(BytesIO(human_image_data)).convert("RGB")
        garment_image_data = base64.b64decode(data['garment_image_base64'])
        garment_image = Image.open(BytesIO(garment_image_data)).convert("RGB")
        print("Images decoded successfully.")
    except Exception as e:
        # ... error handling ...
        print(f"Error decoding images: {e}")
        return jsonify({"error": f"Error decoding images: {e}"}), 500


    # --- Image Preprocessing (No device changes yet) ---
    garm_img = garment_image.resize((768, 1024)) # PIL
    human_img_orig = human_image # PIL

    if data['is_checked_crop']:
        # ... (cropping logic) ...
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2; top = (height - target_height) / 2
        right = (width + target_width) / 2; bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((768, 1024)) # PIL
        print("Human image cropped and resized")
    else:
        human_img = human_img_orig.resize((768, 1024)) # PIL
        crop_size = human_img.size
        left, top = 0, 0
        print("Human image resized")

    # --- Mask Generation (Using device_1 + Explicit Input Tensor) ---
    mask = None
    mask_gray = None
    keypoints_result = None
    parse_result_pil = None
    input_tensor_for_aux = None # Define variable to ensure cleanup

    try:
        human_img_resized_for_aux_pil = human_img.resize((384, 512)) # PIL Image

        # **** THE FIX: Create Tensor on device_1 BEFORE passing to models ****
        # Note: Check if Parsing/OpenPose need specific normalization different from ToTensor()
        simple_to_tensor = transforms.ToTensor() # Simple 0-1 conversion
        input_tensor_for_aux = simple_to_tensor(human_img_resized_for_aux_pil).unsqueeze(0).to(device_1)
        print(f"Input tensor created for aux models on device: {input_tensor_for_aux.device}")
        # *********************************************************************

        if data['is_checked']:
            print(f"Running OpenPose/Parsing on {device_1}")

            # Pass the TENSOR on device_1 to the models
            # !! IMPORTANT: This ASSUMES OpenPose/Parsing can accept a pre-normalized tensor.
            # If they expect PIL or do their OWN ToTensor internally, this won't work
            # and the internal ToTensor needs fixing in those classes.
            with torch.no_grad(): # Use no_grad for inference
                 # Check if models exist before calling
                if openpose_model is None or parsing_model is None:
                    raise ValueError("OpenPose or Parsing model not loaded properly.")

                # --- Call Models with Tensor ---
                # You might need to adjust based on what OpenPose returns with tensor input
                # Assuming keypoints are extractable even with tensor input
                keypoints_data = openpose_model(input_tensor_for_aux)
                print("OpenPose finished.")

                # Assuming parsing model returns a tensor mask
                parse_result_tensor, _ = parsing_model(input_tensor_for_aux)
                print("Parsing finished.")


            # --- Process Results ---
            # Keypoints might need specific processing depending on 'keypoints_data' format
            # For now, assume it's compatible with get_mask_location or extract manually
            keypoints_result = keypoints_data # Store raw output for now
            print("OpenPose raw output format:", type(keypoints_result)) # DEBUG

            # Convert parsing tensor result back to PIL for get_mask_location
            # This assumes parse_result_tensor is [1, C, H, W] or similar
            # Adjust normalization/mode if needed
            parse_result_pil = transforms.ToPILImage()(parse_result_tensor.squeeze(0).cpu())
            print("Parsing result converted back to PIL.")

            # get_mask_location expects PIL mask and keypoints dictionary
            # We need to ensure keypoints_result is in the expected dictionary format.
            # If openpose_model(tensor) doesn't return the dict, this needs adjustment.
            # *** Placeholder: Assuming keypoints_result IS the dict for now ***
            if isinstance(keypoints_result, dict) and 'pose_keypoints_2d' in keypoints_result:
                 print("Keypoints seem to be in expected dict format.")
            else:
                 print("WARNING: OpenPose output format might not be the expected dictionary for get_mask_location when using tensor input. Manual extraction might be needed.")
                 # If it's just the tensor of points, you'd need to format it.
                 # This is a potential point of failure depending on OpenPose model's behavior.

            mask, mask_gray = get_mask_location('hd', data['category'], parse_result_pil, keypoints_result)
            mask = mask.resize((768, 1024))
            mask_gray = mask_gray.resize((768, 1024))
            print("Auto mask generated from tensor results.")

        else: # Manual mask logic (mostly unchanged)
            # ... (manual mask logic remains the same) ...
            print("Using manual mask...")
            if 'mask_image_base64' in data:
                 mask_data = base64.b64decode(data['mask_image_base64'])
                 mask = Image.open(BytesIO(mask_data)).convert("L").resize((768, 1024))
            else:
                 mask = pil_to_binary_mask(human_img.convert("RGB").resize((768, 1024)))

            mask_tensor_temp = transforms.ToTensor()(mask) # CPU tensor
            human_img_tensor_temp = tensor_transfrom(human_img) # CPU tensor, normalized -0.5 to 0.5
            mask_gray_tensor = (1.0 - mask_tensor_temp) * human_img_tensor_temp
            mask_gray = to_pil_image((mask_gray_tensor + 1.0) / 2.0) # PIL, denormalized
            print("Manual mask processed")


    except Exception as e:
        print(f"Error during pose/mask generation phase: {e}")
        if "Expected all tensors to be on the same device" in str(e):
            print("DEVICE MISMATCH ERROR still occurred. This likely means OpenPose/Parsing models cannot accept pre-made tensors OR have deeper internal device issues.")
        torch.cuda.empty_cache()
        gc.collect()
        return jsonify({"error": f"Error processing pose/mask: {e}"}), 500
    finally:
        # Clean up the tensor we created
        if input_tensor_for_aux is not None:
            del input_tensor_for_aux
        # Clean potential intermediate tensor
        if 'parse_result_tensor' in locals():
             del parse_result_tensor
        gc.collect()
        if device_1.type == 'cuda': torch.cuda.empty_cache()


    # --- DensePose (keep targeting device_1) ---
    pose_img = None
    try:
        # ... (DensePose logic remains the same) ...
        print(f"Preparing human image for DensePose on {device_1}")
        human_img_for_densepose = human_img.resize((384, 512))
        human_img_arg = _apply_exif_orientation(human_img_for_densepose)
        human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR") # Numpy

        print(f"Running DensePose on {device_1}")
        densepose_device_str = f"{device_1.type}:{device_1.index}" if device_1.type == 'cuda' else 'cpu'
        densepose_model_path = './ckpt/densepose/model_final_162be9.pkl' # Check path
        if not os.path.exists(densepose_model_path): raise FileNotFoundError(f"DensePose model not found: {densepose_model_path}")

        args = apply_net.create_argument_parser().parse_args(
            ('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml',
             densepose_model_path, 'dp_segm', '-v',
             '--opts', 'MODEL.DEVICE', densepose_device_str)
        )
        pose_output_np = args.func(args, human_img_arg) # Numpy output

        pose_output_np = pose_output_np[:, :, ::-1] # BGR to RGB
        pose_img = Image.fromarray(pose_output_np).resize((768, 1024)) # PIL Image
        print("DensePose completed")
        del human_img_arg, pose_output_np, args
        gc.collect()
        if device_1.type == 'cuda': torch.cuda.empty_cache()

    except Exception as e:
        # ... (error handling) ...
        print(f"Error during DensePose on {device_1}: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        return jsonify({"error": f"Error during DensePose: {e}"}), 500


    # --- Main Inference (keep targeting device_0) ---
    results = []
    try:
        # ... (Main inference logic remains the same) ...
        print(f"Entering inference mode on {device_0}")
        pose_img_tensor = tensor_transfrom(pose_img).unsqueeze(0).to(device_0, dtype=dtype)
        garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).to(device_0, dtype=dtype)

        with torch.no_grad():
            print("Encoding prompts on device_0")
            # ... (prompt encoding) ...
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
                # ... (generation loop logic) ...
                 print(f"Generating image {i+1}/{data['number_of_images']} on {device_0}")
                 if data['is_randomize_seed']:
                     current_seed = torch.randint(0, 2**32, (1,)).item()

                 generator = torch.Generator(device=device_0).manual_seed(current_seed) if current_seed != -1 else None
                 seed_for_gen = current_seed + i
                 if generator: generator.manual_seed(seed_for_gen)
                 print(f"Using seed: {seed_for_gen}")

                 images = pipe(
                     prompt_embeds=prompt_embeds, negative_prompt_embeds=neg_prompt_embeds,
                     pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=neg_pooled_prompt_embeds,
                     num_inference_steps=data['denoise_steps'], generator=generator, strength=1.0,
                     pose_img=pose_img_tensor, text_embeds_cloth=prompt_embeds_c, cloth=garm_tensor,
                     mask_image=mask, image=human_img, height=1024, width=768,
                     ip_adapter_image=garm_img, guidance_scale=2.0,
                 ).images
                 print("Image generated")
                 output_image = images[0]

                 if data['is_checked_crop']:
                     # ... (crop pasting) ...
                    print("Pasting result onto original image")
                    out_img_resized = output_image.resize(crop_size)
                    final_image = human_img_orig.copy()
                    final_image.paste(out_img_resized, (int(left), int(top)))
                    base64_image = pil_to_base64(final_image)
                 else:
                     base64_image = pil_to_base64(output_image)

                 results.append(base64_image)
                 print("Image processed and converted to base64")

                 # ... (cleanup per image) ...
                 del images, output_image, base64_image
                 if 'final_image' in locals(): del final_image
                 if 'out_img_resized' in locals(): del out_img_resized
                 gc.collect()
                 if device_0.type == 'cuda': torch.cuda.empty_cache()

            print("Image generation loop complete")
            # ... (cleanup tensors) ...
            del pose_img_tensor, garm_tensor, prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds, prompt_embeds_c
            return jsonify({"base64_images": results})

    except Exception as e:
        # ... (error handling) ...
        print(f"An error occurred during the main inference process on {device_0}: {e}")
        if 'CUDA out of memory' in str(e): print(f"OOM Error occurred on {device_0} during main inference.")
        return jsonify({"error": f"An error occurred during processing: {e}"}), 500
    finally:
        print("Exiting tryon_image function - Cleaning up")
        # ... (cleanup PIL images, etc.) ...
        if mask is not None: del mask
        if mask_gray is not None: del mask_gray
        if pose_img is not None: del pose_img
        if keypoints_result is not None: del keypoints_result
        if parse_result_pil is not None: del parse_result_pil
        gc.collect()
        torch.cuda.empty_cache() # Clear cache on all devices

# --- Health Check Endpoint (keep as before) ---
@app.route("/check")
def check():
    return "API is running"

# --- Main Execution Block (keep as before) ---
if __name__ == "__main__":
    # ... (GPU checks) ...
    if not torch.cuda.is_available(): print("WARNING: CUDA is not available.")
    elif torch.cuda.device_count() < 2: print("WARNING: Fewer than 2 GPUs detected.")

    if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        with app.app_context():
            load_models()

    app.run(host="0.0.0.0", port=5000, debug=False)
