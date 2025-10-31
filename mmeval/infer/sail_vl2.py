import copy
from typing import List, Tuple

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import (
    parse_args,
    parse_gen_kwargs,
    parse_model_kwargs,
)


COT_PROMPT = (
    r"You FIRST think about the reasoning process as an internal monologue and then "
    r"provide the final answer. The reasoning process MUST BE enclosed within <think> "
    r"</think> tags. The final answer MUST BE put in \\boxed{}."
)


class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.device = torch.device(
            f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        )
        self.default_gen_kwargs = {"max_new_tokens": 512}
        self.model_kwargs = parse_model_kwargs(args)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)

    def load_model(self, args):
        model_path = args.model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        self.model = (
            AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                **self.model_kwargs,
            )
            .to(self.device)
            .eval()
        )

        model_id = model_path if isinstance(model_path, str) else ""
        self.is_thinking_model = "thinking" in model_id.lower()

    def parse_input(self, sample: dict) -> Tuple[List[dict], List[Image.Image]]:
        prompt = sample["prompt"]
        media_paths = sample.get("media", [])
        messages_content: List[dict] = []

        prompt_segments = prompt.split(constants.image)
        for idx, segment in enumerate(prompt_segments):
            if idx > 0 and (idx - 1) < len(media_paths):
                messages_content.append(
                    {"type": "image", "image": media_paths[idx - 1]}
                )
            if segment:
                text_content = segment
                if idx == len(prompt_segments) - 1 and self.is_thinking_model:
                    text_content = text_content + COT_PROMPT
                messages_content.append({"type": "text", "text": text_content})

        if not messages_content and self.is_thinking_model:
            messages_content.append({"type": "text", "text": COT_PROMPT})

        messages = [{"role": "user", "content": messages_content}]

        images: List[Image.Image] = []
        for image_path in media_paths:
            with Image.open(image_path) as img:
                images.append(img.convert("RGB"))

        return messages, images

    def _generate_response(self, inputs):
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **self.gen_kwargs)
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            response = response.split("<|im_end|>")[0].strip()
        return response

    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        messages, images = self.parse_input(ori_sample)

        chat_prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        if images:
            image_inputs = images if len(images) > 1 else images[0]
        else:
            image_inputs = None

        model_inputs = self.processor(
            images=image_inputs,
            text=chat_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        model_inputs = model_inputs.to(self.model.device)
        if "pixel_values" in model_inputs:
            model_inputs["pixel_values"] = model_inputs["pixel_values"].to(self.dtype)

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(model_inputs)
        else:
            pass

        return ori_sample


if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
