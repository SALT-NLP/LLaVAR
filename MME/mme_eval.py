import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from LLaVA.llava.conversation import conv_templates, SeparatorStyle
from LLaVA.llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from LLaVA.llava.model import *
from LLaVA.llava.model.utils import KeywordsStoppingCriteria

from PIL import Image,ImageOps

import os
import requests
from PIL import Image
from io import BytesIO

args = argparse.Namespace()
args.model_name = "/path/to/model/"
args.conv_mode = "llava_v1"

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# Model
disable_torch_init()
model_name = os.path.expanduser(args.model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if "mpt" in model_name.lower():
    model = LlavaMPTForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).cuda()
else:
    model = LlavaLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).cuda()
image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)

mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
if mm_use_im_start_end:
    tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

vision_tower = model.get_model().vision_tower[0]
if vision_tower.device.type == 'meta':
    vision_tower = CLIPVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()
    model.get_model().vision_tower[0] = vision_tower
else:
    vision_tower.to(device='cuda', dtype=torch.float16)
vision_config = vision_tower.config
vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
vision_config.use_im_start_end = mm_use_im_start_end
if mm_use_im_start_end:
    vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def resize_image(image, target_size):
    width, height = image.size
    aspect_ratio = width / height
    if aspect_ratio > 1:
        new_width = target_size[0]
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_size[1]
        new_width = int(new_height * aspect_ratio)
    image = image.resize((new_width, new_height))
    width_diff = target_size[0] - image.size[0]
    height_diff = target_size[1] - image.size[1]
    left_padding = 0
    top_padding = 0
    right_padding = width_diff - left_padding
    bottom_padding = height_diff - top_padding
    padded_image = ImageOps.expand(image, border=(left_padding, top_padding, right_padding, bottom_padding), fill=0)
    return padded_image


def eval_model(args, image):
    qs = args.query
    if mm_use_im_start_end:
        qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
    else:
        qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

    if "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt_multimodal"
    else:
        conv_mode = "multimodal"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # print(prompt)
    inputs = tokenizer([prompt])

    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    # print(outputs)
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs


import os
from tqdm import tqdm

import os

def contains_image_files(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            return True

    return False

print(args.model_name)
files = os.listdir('/path/to/MME_Benchmark_release_version/eval_tool/Your_Results')
for file in files:
    if file[-4:] == ".txt":
        print(file)
        c_cat = file[:-4]
        print(c_cat)
        image_folder = f'/path/to/MME_Benchmark_release_version/{c_cat}/'
        
        if not contains_image_files(image_folder):
            image_folder = f'/path/to/MME_Benchmark_release_version/{c_cat}/images/'
        
        print(image_folder)
        with open(f'/path/to/MME_Benchmark_release_version/eval_tool/Your_Results/{file}', 'r') as qfile:
            q_data = qfile.readlines()
        qa_data = []
        for qa in tqdm(q_data, total=len(q_data)):
            qa = qa.replace("\n", "")
            img_file, q, gt = qa.split("\t")
            img_path = image_folder + img_file
            
            args.image_file = img_path
            args.query = q

            image = load_image(args.image_file)
            image = resize_image(image, (336, 336))

            c_res = eval_model(args, image)
            c_res = c_res.replace("\n", "")
            qa_data.append(img_file + "\t" + q + "\t" + gt + "\t" + c_res + "\n")
        assert len(q_data) == len(qa_data)
        with open(f'/path/to/mme/MME_Benchmark_release_version/eval_tool/LLaVAR/{file}', 'w') as qafile:
            qafile.writelines(qa_data)
        # break