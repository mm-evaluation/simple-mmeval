import re
import copy

import math
import torch
import numpy as np
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens


R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    if isinstance(image_file, Image.Image):
        image = image_file
    else:
        image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    return pixel_values_list, num_patches_list

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.num_segments = 8
        self.default_model_kwargs = {"device_map": "auto", "low_cpu_mem_usage": True}
        # self.default_gen_kwargs = {"max_new_tokens": 1024, "do_sample": True}
        self.default_gen_kwargs = {"num_beams": 1, "top_k": 50, "top_p": 0.9, "sample": False, "max_new_tokens": 1024}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)

    def load_model(self, args):
        self.model = AutoModel.from_pretrained(
            args.model_name_or_path,
            torch_dtype=self.dtype,
            load_in_8bit=False,
            use_flash_attn=True,
            trust_remote_code=True,
            **self.model_kwargs).eval()
        self.model.system_message = R1_SYSTEM_PROMPT
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=False)
        
    def _parse_input(self, msg):
        question = msg["prompt"]
        q_chunks = re.split(r'(<(?:image|video)>)', question)
        question = question.replace("<image>", "<image>\n")
        question = question.replace("<video>", ''.join([f'Frame{i+1}: <image>\n' for i in range(self.num_segments)]))

        media_list = copy.deepcopy(msg["media"])
        pixel_values_list = []
        num_patches_list = []
        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            if chunk == constants.image:
                image = media_list.pop(0)
                image_pixel_values = load_image(image, max_num=12)
                pixel_values_list.append(image_pixel_values)
                num_patches_list.append(image_pixel_values.size(0))
            elif chunk == constants.video:
                video = media_list.pop(0)
                video_pixel_values_list, video_num_patches_list = load_video(video, num_segments=self.num_segments, max_num=1)
                pixel_values_list.extend(video_pixel_values_list)
                num_patches_list.extend(video_num_patches_list)

        if len(pixel_values_list) == 0:
            return question, None, None

        pixel_values = torch.cat(pixel_values_list, dim=0).to(self.dtype).to(self.device)  
                
        return question, pixel_values, num_patches_list

    def _generate_response(self, question, pixel_values, num_patches_list, history=None):
        response, history = self.model.chat(self.tokenizer, pixel_values, question, self.gen_kwargs,
                                            num_patches_list=num_patches_list, history=history, return_history=True)
        return response, history
    
    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        responses = []
        history = None
        all_pixel_values_list = []  # Accumulate pixel values across turns
        all_num_patches_list = []   # Accumulate num_patches across turns
        prev_image_count = 0  # Track images from previous turns

        for msg in sample["messages"]:
            question, new_pixel_values, new_num_patches_list = self._parse_input(msg)
            
            # Accumulate pixel values from this turn
            if new_pixel_values is not None:
                all_pixel_values_list.append(new_pixel_values)
                all_num_patches_list.extend(new_num_patches_list)
            
            # Concatenate ALL accumulated images for model.chat()
            if all_pixel_values_list:
                pixel_values = torch.cat(all_pixel_values_list, dim=0)
            else:
                pixel_values = None
            
            # Prepend placeholders for previous images so model.chat() uses all images
            if prev_image_count > 0:
                prefix = "<image>\n" * prev_image_count
                question = prefix + question
            
            if not self.args.score_target:
                response, history = self._generate_response(question, pixel_values, all_num_patches_list, history)
                responses.append(response)
            
            # Update previous image count for next turn
            prev_image_count = len(all_num_patches_list)

        ori_sample["response"] = responses
        return ori_sample

    
if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()