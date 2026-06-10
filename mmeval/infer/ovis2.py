"""ovis2 — AIDC-AI/Ovis2-{1B,2B,4B,8B,16B,34B}.

HF: https://huggingface.co/AIDC-AI/Ovis2-8B
GH: https://github.com/AIDC-AI/Ovis
Paper: https://arxiv.org/abs/2405.20797
"""
import re
import copy

import torch
from PIL import Image
from transformers import AutoModelForCausalLM

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs


class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.default_model_kwargs = {}
        self.default_gen_kwargs = {
            "max_new_tokens": 1024,
            "do_sample": False,
        }
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)

    def load_model(self, args):
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            multimodal_max_length=32768,
            **self.model_kwargs,
        ).cuda().eval()
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()
        self.max_partition = 9

    def parse_input(self, message):
        question = message["prompt"]
        q_chunks = re.split(r'(<(?:image|video)>)', question)
        media_list = message.get('media', [])

        text_parts = []
        images = []
        media_idx = 0
        for chunk in q_chunks:
            if not chunk.strip():
                continue
            if chunk == constants.image:
                img = media_list[media_idx]
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                elif hasattr(img, "convert"):
                    img = img.convert("RGB")
                images.append(img)
                text_parts.append("<image>")
                media_idx += 1
            elif chunk == constants.video:
                raise NotImplementedError("ovis2 video input not implemented")
            else:
                text_parts.append(chunk)
        return "".join(text_parts).strip(), images

    def run_sample(self, sample: dict):
        if self.args.score_target:
            raise NotImplementedError(
                "ovis2: score_target is not implemented yet"
            )

        ori_sample = copy.deepcopy(sample)
        message = sample["messages"][0]
        query, images = self.parse_input(message)
        if images and "<image>" not in query:
            query = "<image>\n" + query

        prompt, input_ids, pixel_values = self.model.preprocess_inputs(
            query, images if images else None, max_partition=self.max_partition,
        )
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.model.device)
        if pixel_values is not None:
            pixel_values = [
                pixel_values.to(dtype=self.visual_tokenizer.dtype,
                                device=self.visual_tokenizer.device)
            ]
        else:
            pixel_values = [None]

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                **self.gen_kwargs,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
                use_cache=True,
            )[0]
        response = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
        ori_sample["messages"].append({"role": "assistant", "response": response})
        return ori_sample


if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
