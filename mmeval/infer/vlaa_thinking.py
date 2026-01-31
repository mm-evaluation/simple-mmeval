import re
import copy
import torch
import os
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs


def get_model_processor(model_dir, device='cuda', dtype=torch.bfloat16, **model_kwargs):
    if any([_ in model_dir.lower() for _ in ['2vl', '2-vl']]):
        target_class = Qwen2VLForConditionalGeneration
    if any([_ in model_dir.lower() for _ in ['2.5vl', '2.5-vl']]):
        target_class = Qwen2_5_VLForConditionalGeneration

    model = target_class.from_pretrained(
        model_dir, 
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        **model_kwargs
    ).to(device)

    # default processer
    processor = AutoProcessor.from_pretrained(
        model_dir, 
        # # if not enough memory:
        # min_pixels=min_pixels, 
        # max_pixels=max_pixels,
    )

    return model, processor 


class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.default_model_kwargs = {}
        self.default_gen_kwargs = {"max_new_tokens": 1000, "do_sample": False, "use_cache": True}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)

    def load_model(self, args):
        self.model, self.processor = get_model_processor(
            args.model_name_or_path,
            device='cuda',
            dtype=self.dtype,
            **self.model_kwargs
        )
        
    def _parse_input(self, msg):
        question = msg["prompt"]
        # placeholder <>, can be image, video, etc.
        q_chunks = re.split(r'(<(?:image|video)>)', question)
        images = copy.deepcopy(msg["media"])

        messages = [
            {
                'role': 'system',
                'content': (
                    "You are VL-Thinking🤔, a helpful assistant with excellent reasoning ability."
                    " A user asks you a question, and you should try to solve it."
                    " You should first think about the reasoning process in the mind and then provides the user with the answer."
                    " The reasoning process and answer are enclosed within <think> </think> and"
                    " <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>"
                    " <answer> answer here </answer>"
                )
            },
            {
                "role": "user",
                "content": []
            }
        ]

        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            
            if chunk == constants.image:
                media_file = images.pop(0)
                messages[1]["content"].append(
                    {
                        "type": "image",
                        "image": media_file
                    }
                )       
            else:
                messages[1]["content"].append(
                    {
                        "type": "text",
                        "text": chunk
                    }
                )
        
        return messages

    def _generate_response(self, messages):   
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to('cuda')

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, **self.gen_kwargs)
        torch.cuda.empty_cache()
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]
    
    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        responses = []
        for msg in sample["messages"]:
            messages = self._parse_input(msg)
            if not self.args.score_target:
                responses.append(self._generate_response(messages))
        ori_sample["response"] = responses
        return ori_sample

    
if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
