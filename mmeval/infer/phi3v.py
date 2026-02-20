import re
import copy
import torch

from transformers import AutoModelForCausalLM, AutoProcessor

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.dtype = getattr(args, "dtype") or "auto"
        self.default_model_kwargs = {"device_map": "auto"}
        self.default_gen_kwargs = {"max_new_tokens": 100, "do_sample": False, "temperature": 0.0}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)
        
    def load_model(self, args):
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            _attn_implementation='eager', # set to flash_attention_2 if flash attention is supported
            **self.model_kwargs
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            num_crops=4
        )
        
    def _parse_input(self, message:dict):
        prompt = message["prompt"]
        # placeholder <>, can be image, video, etc.
        q_chunks = re.split(r'(<(?:image|video)>)', prompt)
        media_list = message.get('media', [])
        images = []
        media_idx = 0
        content_parts = []
        
        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            if chunk == constants.image:
                images.append(media_list[media_idx])
                content_parts.append(f"<|image_{media_idx + 1}|>")
                media_idx += 1
            else:
                content_parts.append(chunk)
        
        full_content = "".join(content_parts)
        
        messages = [
            {
                "role": "user",
                "content": full_content
            }
        ]

        return messages, images

    def _generate_response(self, inputs):
        with torch.inference_mode():
            generate_ids = self.model.generate(
                **inputs,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                **self.gen_kwargs
            )
            
            # Remove input tokens
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            response = self.processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

        return response
    
    def run_sample(self, sample: dict):
        message = sample["messages"][0]
        ori_sample = copy.deepcopy(sample)
        messages, images = self._parse_input(message)

        # Apply chat template
        prompt = self.processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process with images if available, otherwise just text
        if images:
            inputs = self.processor(prompt, images, return_tensors="pt").to(self.model.device)
        else:
            inputs = self.processor(prompt, return_tensors="pt").to(self.model.device)

        if not self.args.score_target:
            response = self._generate_response(inputs)
            ori_sample["messages"].append({"role": "assistant", "response": response})
        else:
            pass

        return ori_sample

    
if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()