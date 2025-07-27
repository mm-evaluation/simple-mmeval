"""
Aria is a multimodal model from Rhymes AI that handles images and text.
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
from mmeval.utils.argparser import parse_args
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(args)
    
    def load_model(self, args):
        self.model = AriaForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.processor = AriaProcessor.from_pretrained(args.model_name_or_path)
        self.tokenizer = self.processor.tokenizer
    
    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        messages = self.parse_input(sample)
        
        # Extract image path from messages
        image_path = None
        for content in messages[0]["content"]:
            if content["type"] == "image":
                image_path = content["image"]
                break
        
        # Apply chat template to get the prompt
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(text, image_path)
        else:
            ori_sample.update(self._score_choices(text, image_path, sample))
        
        return ori_sample
    
    def _generate_response(self, text, image_path):
        # Load image if provided
        image = None
        if image_path:
            image = Image.open(image_path)
        
        # Process inputs
        inputs = self.processor(text=text, images=image, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
        inputs = inputs.to(self.device)
        
        # Generate response
        output = self.model.generate(
            **inputs,
            max_new_tokens=256,
            stop_strings=["<|im_end|>"],
            tokenizer=self.processor.tokenizer,
            do_sample=True,
            temperature=0.9,
        )
        
        # Decode the response
        output_ids = output[0][inputs["input_ids"].shape[1]:]
        response = self.processor.decode(output_ids, skip_special_tokens=True)
        
        # Remove stop tokens if they appear at the end
        if response.endswith("<|im_end|>"):
            response = response[:-len("<|im_end|>")].strip()
        
        return response
    
    def _score_choices(self, text, image_path, sample):
        contents = sample.get("choices")
        
        # Load image if provided
        image = None
        if image_path:
            image = Image.open(image_path)
        
        # Prepare full prompts with each choice
        full = [text + content for content in contents]
        
        # Process each full prompt with the image
        full_encoded = []
        for prompt in full:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16) if 'pixel_values' in inputs else None
            inputs = inputs.to(self.device)
            full_encoded.append(inputs)
        
        # Process the prompt alone
        prompt_encoded = self.processor(text=text, images=image, return_tensors="pt")
        prompt_encoded['pixel_values'] = prompt_encoded['pixel_values'].to(torch.bfloat16) if 'pixel_values' in prompt_encoded else None
        prompt_encoded = prompt_encoded.to(self.device)
        
        # Get target tokens
        target_toks = target_tokens(self.tokenizer, contents)
        
        # Score using IncrementalLMScorer
        scorer = IncrementalLMScorer(self.model, self.device, tokenizer=self.tokenizer)
        scores = scorer.conditional_score(target_toks, full_encoded, prompt_encoded)
        
        return {
            "score": scores,
            "response": contents[np.argmax(scores)]
        }
    
    def parse_input(self, sample: dict):
        question = sample["prompt"]
        # placeholder <>, can be image, video, audio, etc.
        q_chunks = re.split(r'(<[^>]*>)', question)
        images = copy.deepcopy(sample['media'])
        
        messages = [
            {
                "role": "user",
                "content": []
            }
        ]
        
        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            
            # Check if chunk contains any placeholder
            if chunk in constants.all:
                # Aria supports images
                if chunk == constants.image:
                    media_file = images.pop(0)
                    messages[0]["content"].append(
                        {
                            "type": "image",
                            "image": media_file
                        }
                    )
                else:
                    raise ValueError(f"Aria does not support {chunk} placeholder")
            else:
                messages[0]["content"].append(
                    {
                        "type": "text",
                        "text": chunk
                    }
                )
        
        return messages


if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()