"""aria — rhymes-ai/Aria.

HF: https://huggingface.co/rhymes-ai/Aria
GH: https://github.com/rhymes-ai/Aria
Paper: https://arxiv.org/abs/2410.05993
"""
import re
import copy

import torch
from PIL import Image
from transformers import AriaForConditionalGeneration, AriaProcessor

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs


class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.default_model_kwargs = {"device_map": "auto"}
        self.default_gen_kwargs = {
            "max_new_tokens": 2048,
            "do_sample": True,
            "temperature": 0.9,
        }
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)

    def load_model(self, args):
        self.model = AriaForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            torch_dtype=self.dtype,
            **self.model_kwargs,
        ).eval()
        self.processor = AriaProcessor.from_pretrained(args.model_name_or_path)

    def parse_input(self, message):
        question = message["prompt"]
        q_chunks = re.split(r'(<(?:image|video)>)', question)
        media_list = message.get('media', [])

        content = []
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
                content.append({"type": "image"})
                media_idx += 1
            elif chunk == constants.video:
                raise NotImplementedError("aria video input not implemented")
            else:
                content.append({"type": "text", "text": chunk})
        return [{"role": "user", "content": content}], images

    def _generate_response(self, inputs):
        output = self.model.generate(
            **inputs,
            stop_strings=["<|im_end|>"],
            tokenizer=self.processor.tokenizer,
            **self.gen_kwargs,
        )
        trimmed = output[0][inputs["input_ids"].shape[1]:]
        response = self.processor.decode(trimmed, skip_special_tokens=True)
        # Aria emits the chat-template boundary token <|im_end|> even when it
        # is passed as a stop string; strip a single trailing occurrence.
        if response.endswith("<|im_end|>"):
            response = response[:-len("<|im_end|>")]
        return response.rstrip()

    def run_sample(self, sample: dict):
        if self.args.score_target:
            raise NotImplementedError(
                "aria: score_target is not implemented yet"
            )

        ori_sample = copy.deepcopy(sample)
        message = sample["messages"][0]
        messages, images = self.parse_input(message)

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        proc_kwargs = {"text": text, "return_tensors": "pt"}
        if images:
            proc_kwargs["images"] = images
        inputs = self.processor(**proc_kwargs)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        inputs = inputs.to(self.model.device)

        response = self._generate_response(inputs)
        ori_sample["messages"].append({"role": "assistant", "response": response})
        return ori_sample


if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
