"""glm_4d1v — GLM-4.1V-9B-Thinking.

HF: https://huggingface.co/THUDM/GLM-4.1V-9B-Thinking
GH: https://github.com/THUDM/GLM-V
"""
import re
import copy

import torch
from PIL import Image
from transformers import AutoProcessor, Glm4vForConditionalGeneration

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs


class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.default_model_kwargs = {"device_map": "auto"}
        self.default_gen_kwargs = {"max_new_tokens": 2048}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)

    def load_model(self, args):
        self.model = Glm4vForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            torch_dtype=self.dtype,
            **self.model_kwargs,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            args.model_name_or_path, use_fast=True,
        )

    def parse_input(self, message):
        question = message["prompt"]
        q_chunks = re.split(r'(<(?:image|video)>)', question)
        media_list = message.get('media', [])

        content = []
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
                content.append({"type": "image", "image": img})
                media_idx += 1
            elif chunk == constants.video:
                content.append({"type": "video", "video": media_list[media_idx]})
                media_idx += 1
            else:
                content.append({"type": "text", "text": chunk})
        return [{"role": "user", "content": content}]

    def _generate_response(self, inputs):
        generated_ids = self.model.generate(**inputs, **self.gen_kwargs)
        trimmed = generated_ids[0][inputs["input_ids"].shape[1]:]
        response = self.processor.decode(trimmed, skip_special_tokens=False)
        # Strip the trailing chat-template boundary token. The semantic
        # <|begin_of_box|>...<|end_of_box|> (and <answer></answer>) markers
        # are kept since they delimit the model's structured answer.
        if response.endswith("<|user|>"):
            response = response[:-len("<|user|>")]
        return response.rstrip()

    def run_sample(self, sample: dict):
        if self.args.score_target:
            raise NotImplementedError(
                "glm_4d1v: score_target is not implemented yet"
            )

        ori_sample = copy.deepcopy(sample)
        message = sample["messages"][0]
        messages = self.parse_input(message)

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        response = self._generate_response(inputs)
        ori_sample["messages"].append({"role": "assistant", "response": response})
        return ori_sample


if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
