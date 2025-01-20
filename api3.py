from flask import Flask, request, jsonify
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
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
from util.pipeline import quantize_4bit, restart_cpu_offload, torch_gc
from io import BytesIO
import base64
import gc  # Import the gc module
import threading

app = Flask(__name__)

# --- Initialize your models and pipeline here (same as in your Gradio code) ---
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
parsing_model = None  # Initialize globally
openpose_model = None # Initialize globally
example_path = os.path.join(os.path.dirname(__file__), 'example')
tensor_transfrom = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ]
            )

# --- Load models and pipeline only once ---
def load_models():
    global pipe, unet, UNet_Encoder, parsing_model, openpose_model
    print("Loading models...")
    unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        torch_dtype=dtypeQuantize,
    ).to(device)
    if load_mode == '4bit':
        quantize_4bit(unet)
    unet.requires_grad_(False)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        model_id,
        subfolder="image_encoder",
        torch_dtype=torch.float16,
    ).to(device)
    if load_mode == '4bit':
        quantize_4bit(image_encoder)
    image_encoder.requires_grad_(False)

    if fixed_vae:
        vae = AutoencoderKL.from_pretrained(vae_model_id, torch_dtype=dtype).to(device)
    else:
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype).to(device)
    vae.requires_grad_(False)

    UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
        model_id,
        subfolder="unet_encoder",
        torch_dtype=dtypeQuantize,
    ).to(device)
    if load_mode == '4bit':
        quantize_4bit(UNet_Encoder)
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

    if load_mode == '4bit':
        if pipe.text_encoder is not None:
            quantize_4bit(pipe.text_encoder)
            pipe.text_encoder.to(device)
        if pipe.text_encoder_2 is not None:
            quantize_4bit(pipe.text_encoder_2)
            pipe.text_encoder_2.to(device)

    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)
    openpose_model.preprocessor.body_estimation.model.to(device)
    print("Models loaded successfully!")

# Load models when the app starts
with app.app_context():
    load_models()

# --- Request Queue ---
request_queue = threading.Queue()

def process_queue():
    while True:
        try:
            func, args, kwargs = request_queue.get()
            func(*args, **kwargs)
            request_queue.task_done()
        except Exception as e:
            print(f"Error processing queue item: {e}")

queue_thread = threading.Thread(target=process_queue, daemon=True)
queue_thread.start()

# --- Your start_tryon function (modified for API input/output) ---
def start_tryon_internal(human_img_file, garm_img_file, garment_des, category, is_checked, is_checked_crop, denoise_steps, is_randomize_seed, seed, number_of_images):
    global pipe, need_restart_cpu_offloading, parsing_model, openpose_model, tensor_transfrom

    results = []
    masked_image_base64 = None

    try:
        if pipe is None:
            raise Exception("Pipeline not initialized")

        if ENABLE_CPU_OFFLOAD and need_restart_cpu_offloading:
            restart_cpu_offload(pipe, load_mode)
            need_restart_cpu_offloading = False
        elif ENABLE_CPU_OFFLOAD:
            pipe.enable_model_cpu_offload()

        torch_gc()

        human_img_orig = Image.open(human_img_file).convert("RGB")
        garm_img = Image.open(garm_img_file).convert("RGB").resize((768,1024))

        if is_checked_crop:
            width, height = human_img_orig.size
            target_width = int(min(width, height * (3 / 4)))
            target_height = int(min(height, width * (4 / 3)))
            left = (width - target_width) / 2
            top = (height - target_height) / 2
            right = (width + target_width) / 2
            bottom = (height + target_height) / 2
            cropped_img = human_img_orig.crop((left, top, right, bottom))
            crop_size = cropped_img.size
            human_img = cropped_img.resize((768,1024))
        else:
            human_img = human_img_orig.resize((768,1024))

        if is_checked:
            keypoints = openpose_model(human_img.resize((384,512)))
            model_parse, _ = parsing_model(human_img.resize((384,512)))
            mask, mask_gray = get_mask_location('hd', category, model_parse, keypoints)
            mask = mask.resize((768,1024))
        else:
            # Assuming the mask is provided as another image file
            mask_file = request.files.get('mask_image')
            if not mask_file:
                raise ValueError("Mask image is required when not using auto-masking")
            mask = pil_to_binary_mask(Image.open(mask_file).convert("RGB").resize((768, 1024)))

        mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
        mask_gray = to_pil_image((mask_gray+1.0)/2.0)

        buffered_mask = BytesIO()
        mask_gray.save(buffered_mask, format="PNG")
        masked_image_base64 = base64.b64encode(buffered_mask.getvalue()).decode('utf-8')

        human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
        human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

        args_dp = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
        pose_img = args_dp.func(args_dp,human_img_arg)
        pose_img = pose_img[:,:,::-1]
        pose_img = Image.fromarray(pose_img).resize((768,1024))

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype), torch.inference_mode():
            prompt = "model is wearing " + garment_des
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
                prompt, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=negative_prompt
            )

            prompt_c = "a photo of " + garment_des
            negative_prompt_c = "monochrome, lowres, bad anatomy, worst quality, low quality"
            prompt_embeds_c, _, _, _ = pipe.encode_prompt(
                prompt_c, num_images_per_prompt=1, do_classifier_free_guidance=False, negative_prompt=negative_prompt_c
            )

            pose_img_tensor = tensor_transfrom(pose_img).unsqueeze(0).to(device,dtype)
            garm_tensor =  tensor_transfrom(garm_img).unsqueeze(0).to(device,dtype)
            current_seed = seed
            for i in range(number_of_images):
                if is_randomize_seed:
                    current_seed = torch.randint(0, 2**32, size=(1,)).item()
                generator = torch.Generator(device).manual_seed(current_seed) if seed != -1 else None
                current_seed = current_seed + i

                images = pipe(
                    prompt_embeds=prompt_embeds.to(device,dtype),
                    negative_prompt_embeds=negative_prompt_embeds.to(device,dtype),
                    pooled_prompt_embeds=pooled_prompt_embeds.to(device,dtype),
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device,dtype),
                    num_inference_steps=denoise_steps,
                    generator=generator,
                    strength = 1.0,
                    pose_img = pose_img_tensor.to(device,dtype),
                    text_embeds_cloth=prompt_embeds_c.to(device,dtype),
                    cloth = garm_tensor.to(device,dtype),
                    mask_image=mask,
                    image=human_img,
                    height=1024,
                    width=768,
                    ip_adapter_image = garm_img.resize((768,1024)),
                    guidance_scale=2.0,
                    dtype=dtype,
                    device=device,
                ).images

                for img in images:
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    results.append(img_base64)

            del pose_img_tensor, garm_tensor, prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds, prompt_embeds_c
            torch.cuda.empty_cache()
            torch_gc()

        return jsonify({"generated_images": results, "masked_image": masked_image_base64})

    except Exception as e:
        print(f"Error during processing: {e}")
        torch.cuda.empty_cache()
        torch_gc()
        return jsonify({"error": str(e)}), 500

@app.route('/tryon', methods=['POST'])
def tryon_endpoint():
    human_img_file = request.files.get('human_image')
    garm_img_file = request.files.get('garment_image')
    mask_image_file = request.files.get('mask_image') # Allow mask image to be passed

    if not human_img_file or not garm_img_file:
        return jsonify({"error": "Both human_image and garment_image are required"}), 400

    try:
        garment_des = request.form.get('garment_des')
        category = request.form.get('category', 'upper_body')
        is_checked = request.form.get('is_checked') == 'true'
        is_checked_crop = request.form.get('is_checked_crop') == 'true'
        denoise_steps = int(request.form.get('denoise_steps', 30))
        is_randomize_seed = request.form.get('is_randomize_seed') == 'true'
        seed = int(request.form.get('seed', 1))
        number_of_images = int(request.form.get('number_of_images', 1))

        # Add the request to the queue
        request_queue.put((start_tryon_internal, (human_img_file, garm_img_file), {
            'garment_des': garment_des,
            'category': category,
            'is_checked': is_checked,
            'is_checked_crop': is_checked_crop,
            'denoise_steps': denoise_steps,
            'is_randomize_seed': is_randomize_seed,
            'seed': seed,
            'number_of_images': number_of_images
        }))

        return jsonify({"message": "Request added to queue"}), 202 # Accepted

    except Exception as e:
        print(f"Error receiving request: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)
