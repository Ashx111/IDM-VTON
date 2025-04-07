# --- Imports (keep all previous imports) ---
import torch
# ... other imports ...
from flask import Flask, request, jsonify
import gc
import os
# ... etc ...
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
# ... etc ...


app = Flask(__name__)

# --- Device Setup (Single GPU - keep as before) ---
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"Using single GPU: {device}")
    torch.cuda.set_device(device) # Set default device
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")


# --- Global variables (keep as before) ---
pipe = None
unet = None
# ... other globals ...
parsing_model = None
openpose_model = None
tensor_transfrom = None
# ...

# --- Load Models Function (Single GPU Float16 - **REVISED AUX LOADING**) ---
def load_models():
    print(f"loading models onto single device {device} using float16...")
    global pipe, unet, UNet_Encoder, image_encoder, vae, parsing_model, openpose_model, tensor_transfrom, clip_image_processor

    dtype = torch.float16
    fixed_vae = True
    model_id = 'yisol/IDM-VTON'
    vae_model_id = 'madebyollin/sdxl-vae-fp16-fix'

    # --- Load Main Pipeline Components (Keep as before) ---
    print(f"Loading unet...")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype).to(device)
    unet.requires_grad_(False)
    # ... load image_encoder, vae, UNet_Encoder onto device ...
    print(f"Loading image_encoder...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=dtype).to(device)
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
    UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(model_id, subfolder="unet_encoder", torch_dtype=dtype).to(device)
    UNet_Encoder.requires_grad_(False)


    clip_image_processor = CLIPImageProcessor()

    pipe_param = { # Args for pipeline init
        'pretrained_model_name_or_path': model_id, 'unet': unet, 'torch_dtype': dtype,
        'vae': vae, 'image_encoder': image_encoder, 'feature_extractor': clip_image_processor,
    }
    print(f"Initializing TryonPipeline...")
    pipe = TryonPipeline.from_pretrained(**pipe_param)
    pipe.unet_encoder = UNet_Encoder
    pipe.to(device) # Ensure whole pipe is on the device

    # --- Load Auxiliary Models ---
    # Initialize first (potentially on CPU or default device)
    parsing_device_idx = device.index if device.type == 'cuda' else -1
    openpose_device_idx = device.index if device.type == 'cuda' else -1

    print(f"Initializing parsing model (target device index: {parsing_device_idx})...")
    parsing_model = Parsing(parsing_device_idx)
    print(f"Initializing openpose model (target device index: {openpose_device_idx})...")
    openpose_model = OpenPose(openpose_device_idx)

    # *** Explicitly move aux models to the target device AFTER initialization ***
    try:
        if parsing_model is not None:
            parsing_model.model.to(device) # Assuming the main model is under a .model attribute
            # Add check:
            p_param = next(parsing_model.model.parameters())
            print(f"Parsing model parameter device after move: {p_param.device}")
        else:
            print("Warning: Parsing model is None, cannot move.")
    except AttributeError:
         print(f"Warning: Could not access parsing_model.model. Trying to move top-level object.")
         try:
              if parsing_model is not None:
                   parsing_model.to(device) # Fallback: try moving the whole object
                   # Re-check after moving top-level
                   if hasattr(parsing_model, 'model'):
                        p_param = next(parsing_model.model.parameters())
                        print(f"Parsing model parameter device after top-level move: {p_param.device}")
                   else: # If no .model, check parameters directly if possible
                        p_param = next(parsing_model.parameters())
                        print(f"Parsing model top-level parameter device after move: {p_param.device}")

         except Exception as e:
              print(f"Error moving parsing_model fallback: {e}")
    except Exception as e:
        print(f"Error moving parsing_model: {e}")


    try:
        if openpose_model is not None:
            # OpenPose has a nested structure, move the relevant part
            if hasattr(openpose_model, 'preprocessor.body_estimation.model') and \
               openpose_model.preprocessor.body_estimation.model is not None:
                 openpose_model.preprocessor.body_estimation.model.to(device)
                 # Add check:
                 op_param = next(openpose_model.preprocessor.body_estimation.model.parameters())
                 print(f"OpenPose sub-model parameter device after move: {op_param.device}")
            else:
                 print(f"Warning: Could not find OpenPose sub-model at expected path. Trying to move top-level.")
                 openpose_model.to(device) # Fallback: try moving the whole object
                 # Re-check after moving top-level
                 if hasattr(openpose_model, 'preprocessor.body_estimation.model') and openpose_model.preprocessor.body_estimation.model is not None:
                      op_param = next(openpose_model.preprocessor.body_estimation.model.parameters())
                      print(f"OpenPose sub-model parameter device after top-level move: {op_param.device}")
                 else: # Check top-level parameters if possible
                      op_param = next(openpose_model.parameters())
                      print(f"OpenPose top-level parameter device after move: {op_param.device}")

        else:
            print("Warning: OpenPose model is None, cannot move.")

    except Exception as e:
        print(f"Error moving openpose_model: {e}")


    # --- Initialize Transforms (CPU operation) ---
    tensor_transfrom = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
    ])

    print("loading models completed!!!!!!!!!!!!")
    gc.collect()
    if device.type == 'cuda': torch.cuda.empty_cache()


# --- tryon_image Endpoint (Keep the Single GPU Float16 w/ Cleanup version) ---
@app.route('/tryon', methods=['POST'])
def tryon_image():
    # --- Use the tryon_image function from the previous "Single GPU Float16 w/ Cleanup" step ---
    # --- No changes needed here for now ---
    global pipe, parsing_model, openpose_model, tensor_transfrom

    print("Entering tryon_image function (Single GPU)")
    data = request.get_json()
    dtype = torch.float16
    output_images_base64 = []

    # Define variables accessed in finally
    human_img_tensor_temp, mask_tensor_temp, mask_gray_tensor = None, None, None
    mask, mask_gray, keypoints_result, parse_result_pil = None, None, None, None
    human_img_arg, densepose_args, pose_output_np, pose_img = None, None, None, None
    pose_img_tensor, garm_tensor = None, None
    prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds, prompt_embeds_c = None, None, None, None, None
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
            # ... crop logic ...
            width, height = human_img_orig.size
            target_width = int(min(width, height * (3 / 4)))
            target_height = int(min(height, width * (4 / 3)))
            left = (width - target_width) / 2; top = (height - target_height) / 2
            right = (width + target_width) / 2; bottom = (height + target_height) / 2
            cropped_img = human_img_orig.crop((left, top, right, bottom))
            crop_size = cropped_img.size
            human_img = cropped_img.resize((768, 1024)) # PIL
            print("Human image cropped and resized")
            del cropped_img
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
                # Models should now be on 'device'
                with torch.no_grad():
                     if openpose_model is None or parsing_model is None:
                         raise ValueError("OpenPose or Parsing model not loaded.")

                     # *** Crucial: Check if models now handle device correctly internally,
                     # *** OR if we still need to put input tensor on device manually.
                     # *** Let's ASSUME for now the models MIGHT handle PIL correctly if they are on the right device.
                     # *** If this fails with device mismatch again, uncomment the explicit tensor creation below.

                     # print("DEBUG: Creating input tensor manually for aux models")
                     # simple_to_tensor = transforms.ToTensor()
                     # input_tensor_for_aux = simple_to_tensor(human_img_resized_for_aux_pil).unsqueeze(0).to(device)

                     # keypoints_result = openpose_model(input_tensor_for_aux)
                     # parse_result_tensor, _ = parsing_model(input_tensor_for_aux) # Assuming tensor output
                     # # Convert parse_result_tensor back to PIL
                     # parse_result_pil = transforms.ToPILImage()(parse_result_tensor.squeeze(0).cpu())
                     # del input_tensor_for_aux, parse_result_tensor # cleanup tensor


                     # --- Try passing PIL directly first, assuming models are now on GPU ---
                     keypoints_result = openpose_model(human_img_resized_for_aux_pil)
                     parse_result_pil, _ = parsing_model(human_img_resized_for_aux_pil)
                     # --- End Try passing PIL ---


                if not (isinstance(keypoints_result, dict) and 'pose_keypoints_2d' in keypoints_result):
                     raise ValueError(f"OpenPose did not return expected keypoints format. Got: {type(keypoints_result)}")

                mask, mask_gray = get_mask_location('hd', data['category'], parse_result_pil, keypoints_result)
                mask = mask.resize((768, 1024)) # PIL
                mask_gray = mask_gray.resize((768, 1024)) # PIL
                print("Auto mask generated.")

            else: # Manual mask logic
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
                del mask_tensor_temp, human_img_tensor_temp, mask_gray_tensor # Cleanup

            del human_img_resized_for_aux_pil # Cleanup

        except Exception as e:
            print(f"Error during pose/mask generation phase: {e}")
            # Check for the specific error
            if "Input type" in str(e) and "weight type" in str(e):
                 print("!!! Device mismatch error during Aux models STILL occurred.")
                 print("!!! This likely means internal model processing defaults to CPU or wrong device.")
            return jsonify({"error": f"Error processing pose/mask: {e}"}), 500
        finally:
             gc.collect()
             if device.type == 'cuda': torch.cuda.empty_cache()


        # --- DensePose (Keep as before - relies on args for device) ---
        try:
            # ... DensePose logic ...
            human_img_for_densepose = human_img.resize((384, 512))
            human_img_arg = _apply_exif_orientation(human_img_for_densepose)
            human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

            print(f"Running DensePose on {device}")
            densepose_device_str = f"{device.type}:{device.index}" if device.type == 'cuda' else 'cpu'
            densepose_model_path = './ckpt/densepose/model_final_162be9.pkl'
            if not os.path.exists(densepose_model_path): raise FileNotFoundError(f"DensePose model not found: {densepose_model_path}")

            densepose_args = apply_net.create_argument_parser().parse_args(
                ('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml',
                 densepose_model_path, 'dp_segm', '-v',
                 '--opts', 'MODEL.DEVICE', densepose_device_str)
            )
            pose_output_np = densepose_args.func(densepose_args, human_img_arg)
            pose_output_np = pose_output_np[:, :, ::-1] # BGR to RGB
            pose_img = Image.fromarray(pose_output_np).resize((768, 1024))
            print("DensePose completed")
            del human_img_arg, densepose_args, pose_output_np
            del human_img_for_densepose

        except Exception as e:
            # ... DensePose error handling ...
            print(f"Error during DensePose on {device}: {e}")
            if 'CUDA out of memory' in str(e): print("OOM Error during DensePose detected.")
            return jsonify({"error": f"Error during DensePose: {e}"}), 500
        finally:
            # Clear cache immediately after DensePose
             gc.collect()
             if device.type == 'cuda': torch.cuda.empty_cache()


        # --- Main Inference (Keep as before) ---
        try:
            # ... Main inference logic ...
             print(f"Entering inference mode on {device}")
             pose_img_tensor = tensor_transfrom(pose_img).unsqueeze(0).to(device, dtype=dtype)
             garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).to(device, dtype=dtype)

             with torch.no_grad():
                # ... prompt encoding ...
                print("Encoding prompts...")
                prompt = "model is wearing " + data['garment_description']
                neg_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds = pipe.encode_prompt(
                    prompt, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=neg_prompt
                )
                prompt_c = "a photo of " + data['garment_description']
                prompt_embeds_c, _, _, _ = pipe.encode_prompt(
                    prompt_c, num_images_per_prompt=1, do_classifier_free_guidance=False, negative_prompt=neg_prompt
                )
                print("Prompts encoded")


                current_seed = data['seed']
                print("Starting image generation loop")
                for i in range(data['number_of_images']):
                    # ... generation loop ...
                     print(f"Generating image {i+1}/{data['number_of_images']}")
                     if data['is_randomize_seed']:
                         current_seed = torch.randint(0, 2**32, (1,)).item()
                     generator = torch.Generator(device=device).manual_seed(current_seed) if current_seed != -1 else None
                     seed_for_gen = current_seed + i
                     if generator: generator.manual_seed(seed_for_gen)
                     print(f"Using seed: {seed_for_gen}")

                     images = pipe( # Main pipeline call
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
                         # ... pasting logic ...
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
                     if device.type == 'cuda': torch.cuda.empty_cache()

                print("Image generation loop complete")

        except Exception as e:
            # ... error handling for main inference ...
             print(f"An error occurred during the main inference process: {e}")
             # Check for the LayerNorm error specifically on subsequent requests
             if "LayerNormKernelImpl" in str(e):
                  print("!!! LayerNorm error occurred during main pipe call.")
             if 'CUDA out of memory' in str(e): print(f"OOM Error during main inference.")
             return jsonify({"error": f"An error occurred during processing: {e}"}), 500


        # --- Return Success ---
        return jsonify({"base64_images": output_images_base64})

    # --- Master Finally Block (Keep comprehensive cleanup) ---
    finally:
        print("Exiting tryon_image function - Cleaning up intermediate data")
        # ... comprehensive del statements ...
        del human_img_tensor_temp, mask_tensor_temp, mask_gray_tensor
        del mask, mask_gray, keypoints_result, parse_result_pil
        del human_img_arg, densepose_args, pose_output_np, pose_img
        del pose_img_tensor, garm_tensor
        del prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds, prompt_embeds_c
        if 'images' in locals() and images is not None: del images

        collected = gc.collect()
        print(f"Garbage collector collected {collected} items.")
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            print("CUDA cache cleared.")


# --- Health Check & Main Execution (Keep as before) ---
@app.route("/check")
def check():
    return "API is running"

if __name__ == "__main__":
    if not torch.cuda.is_available(): print("WARNING: CUDA is not available.")
    if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        with app.app_context():
            load_models()
    app.run(host="0.0.0.0", port=5000, debug=False)
