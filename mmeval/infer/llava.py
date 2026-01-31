import re
import copy
import torch

from transformers import AutoProcessor, LlavaForConditionalGeneration

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.default_model_kwargs = {"attn_implementation": "flash_attention_2", "device_map": "auto", "low_cpu_mem_usage": True}
        self.default_gen_kwargs = {"max_new_tokens": 200, "do_sample": False}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)

    def load_model(self, args):
        self.model = LlavaForConditionalGeneration.from_pretrained(
            args.model_name_or_path, 
            torch_dtype=self.dtype, 
            load_in_4bit=False,
            **self.model_kwargs
        ).to(0)

        self.processor = AutoProcessor.from_pretrained(args.model_name_or_path)
        
    def _parse_input(self, msg):
        prompt = msg["prompt"]
        has_image = bool(msg["media"])

        if has_image:
            prompt = prompt.replace("<image>", "")
            content = [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]
        else:
            content = [{"type": "text", "text": prompt}]

        conversation = [
            {
                "role": "user",
                "content": content,
            },
        ]

        return conversation

    def _generate_response(self, inputs):
        output = self.model.generate(**inputs, **self.gen_kwargs)

        return self.processor.decode(output[0][2:], skip_special_tokens=True)
    
    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        responses = []
        for msg in sample["messages"]:
            conversation = self._parse_input(msg)
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

            if msg["media"]:
                image = msg["media"][0]
                inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(0, self.dtype)
            else:
                inputs = self.processor(text=prompt, return_tensors="pt").to(0, self.dtype)

            if not self.args.score_target:
                responses.append(self._generate_response(inputs))
            else:
                pass

        ori_sample["response"] = responses
        return ori_sample

    
if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
