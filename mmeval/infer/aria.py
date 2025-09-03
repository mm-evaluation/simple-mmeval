"""
Aria is a multimodal model from Rhymes AI that handles text, image, and video.
https://huggingface.co/rhymes-ai/Aria
"""
import re
import copy
import torch
import numpy as np
from PIL import Image
from transformers import AriaProcessor, AriaForConditionalGeneration

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.default_model_kwargs = {"device_map": "auto"}
        self.default_gen_kwargs = {"max_new_tokens": 100, "do_sample": False, "stop_strings": ["<|im_end|>"]}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)
    
    def load_model(self, args):
        self.model = AriaForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            **self.model_kwargs
        )
        self.processor = AriaProcessor.from_pretrained(args.model_name_or_path)
        self.tokenizer = self.processor.tokenizer
    
    def _parse_input(self, sample:dict):
        prompt = sample["prompt"]
        q_chunks = re.split(r'(<(?:image|video)>)', prompt)
        media = copy.deepcopy(sample['media'])

        messages = [
            {
                "role": "user",
                "content": []
            }
        ]

        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            if chunk == constants.image:
                media_file = media.pop(0)
                messages[0]["content"].append(
                    {
                        "type": "image",
                        "image": media_file
                    }
                )
            elif chunk == constants.video:
                media_file = media.pop(0)
                messages[0]["content"].append(
                    {
                        "type": "video",
                        "video": media_file
                    }
                )
            else:
                messages[0]["content"].append(
                    {
                        "type": "text",
                        "text": chunk
                    }
                )

        return messages
    
    def _generate_response(self, text, messages):
        images = []
        videos = []
        
        for content in messages[0]["content"]:
            if content["type"] == "image":
                images.append(Image.open(content["image"]))
            elif content["type"] == "video":
                videos.append(content["video"])
        
        # Process inputs
        inputs = self.processor(
            text=text, 
            images=images if images else None, 
            videos=videos if videos else None,
            return_tensors="pt"
        )
        
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(self.dtype)
        if 'video_pixel_values' in inputs:
            inputs['video_pixel_values'] = inputs['video_pixel_values'].to(self.dtype)
        
        inputs = inputs.to(self.device)
        
        # Generate response
        output = self.model.generate(
            **inputs,
            **self.gen_kwargs,
            tokenizer=self.processor.tokenizer,
        )
        
        # Decode the response
        output_ids = output[0][inputs["input_ids"].shape[1]:]
        response = self.processor.decode(output_ids, skip_special_tokens=True)
        
        # Remove stop tokens if they appear at the end
        if response.endswith("<|im_end|>"):
            response = response[:-len("<|im_end|>")].strip()
        
        return response

    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        messages = self._parse_input(ori_sample)
        
        # Apply chat template to get the prompt
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(text, messages)
        else:
            ori_sample.update(self._score_choices(text, messages, sample))
        
        return ori_sample
    
    def _score_choices(self, text, messages, sample):
        pass

if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()