from fastapi import FastAPI, UploadFile, File
from PIL import Image
from pydantic import BaseModel
import torch
from typing import List
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)
from diffusers import AutoencoderKL
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

app = FastAPI()

# Load models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pipe = None
unet = None
UNet_Encoder = None

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

def load_models():
    global pipe, unet, UNet_Encoder
    dtype = torch.float16
    dtypeQuantize = dtype
    load_mode = '8bit'
    if(load_mode in ('4bit','8bit')):
        dtypeQuantize = torch.float8_e4m3fn

    model_id = 'yisol/IDM-VTON'
    vae_model_id = 'madebyollin/sdxl-vae-fp16-fix'

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

    if True:
        vae = AutoencoderKL.from_pretrained(vae_model_id, torch_dtype=dtype)
    else:            
        vae = AutoencoderKL.from_pretrained(model_id,
                                            subfolder="vae",
                                            torch_dtype=dtype,
        )

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

load_models()

@app.post("/tryon/")
async def tryon_image(request: TryonRequest):
    # Process request
    human_image = Image.open(request.human_image)
    garment_image = Image.open(request.garment_image)

    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)
    openpose_model.preprocessor.body_estimation.model.to(device)
    tensor_transfrom = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ]
            )

    garm_img= garment_image.convert("RGB").resize((768,1024))
    human_img_orig = human_image.convert("RGB")    
    
    if request.is_checked_crop:
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

    if request.is_checked:
        keypoints = openpose_model(human_img.resize((384,512)))
        model_parse, _ = parsing_model(human_img.resize((384,512)))
        mask, mask_gray = get_mask_location('hd', request.category, model_parse, keypoints)
        mask = mask.resize((768,1024))
    else:
        mask = pil_to_binary_mask(human_image.convert("RGB").resize((768, 1024)))
    
    mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray+1.0)/2.0)

    human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    pose_img = args.func(args,human_img_arg)    
    pose_img = pose_img[:,:,::-1]    
    pose_img = Image.fromarray(pose_img).resize((768,1024))
    
    if pipe.text_encoder is not None:        
        pipe.text_encoder.to(device)

    if pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.to(device)

    with torch.no_grad():
        # Extract the images
        with torch.cuda.amp.autocast(dtype=torch.float16):
            with torch.no_grad():
                prompt = "model is wearing " + request.garment_description
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
                                    
                    prompt = "a photo of " + request.garment_description
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

                    pose_img =  tensor_transfrom(pose_img).unsqueeze(0).to(device,torch.float16)
                    garm_tensor =  tensor_transfrom(garm_img).unsqueeze(0).to(device,torch.float16)
                    results = []
                    current_seed = request.seed
                    for i in range(request.number_of_images):  
                        if request.is_randomize_seed:
                            current_seed = torch.randint(0, 2**32, size=(1,)).item()                        
                        generator = torch.Generator(device).manual_seed(current_seed) if request.seed != -1 else None                     
                        current_seed = current_seed + i

                        images = pipe(
                            prompt_embeds=prompt_embeds.to(device,torch.float16),
                            negative_prompt_embeds=negative_prompt_embeds.to(device,torch.float16),
                            pooled_prompt_embeds=pooled_prompt_embeds.to(device,torch.float16),
                            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device,torch.float16),
                            num_inference_steps=request.denoise_steps,
                            generator=generator,
                            strength = 1.0,
                            pose_img = pose_img.to(device,torch.float16),
                            text_embeds_cloth=prompt_embeds_c.to(device,torch.float16),
                            cloth = garm_tensor.to(device,torch.float16),
                            mask_image=mask,
                            image=human_img, 
                            height=1024,
                            width=768,
                            ip_adapter_image = garm_img.resize((768,1024)),
                            guidance_scale=2.0,
                            dtype=torch.float16,
                            device=device,
                        )[0]
                        if request.is_checked_crop:
                            out_img = images[0].resize(crop_size)        
                            human_img_orig.paste(out_img, (int(left), int(top)))   
                            img_path = save_output_image(human_img_orig, base_path="outputs", base_filename='img', seed=current_seed)
                            results.append(img_path)
                        else:
                            img_path = save_output_image(images[0], base_path="outputs", base_filename='img')
                            results.append(img_path)
                    return {"results": results}

@app.get("/check")
async def check():
    return "API is running"
