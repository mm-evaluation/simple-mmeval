import copy
import re
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoProcessor, Glm4vForConditionalGeneration

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs


PLACEHOLDER_TO_TYPE = {constants.image: "image", constants.video: "video"}


def _collect_media_entries(entry: Any, declared_type: str | None = None) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []

    if entry is None:
        return items

    if isinstance(entry, dict):
        media_type = entry.get("type") or entry.get("media_type") or declared_type
        value = entry.get("value") or entry.get("path") or entry.get("url")

        if isinstance(value, (list, tuple)):
            for nested in value:
                items.extend(_collect_media_entries(nested, media_type))
        elif value is not None:
            items.extend(_collect_media_entries(value, media_type))
        else:
            for nested in entry.values():
                items.extend(_collect_media_entries(nested, media_type))

        return items

    if isinstance(entry, (list, tuple)):
        for nested in entry:
            items.extend(_collect_media_entries(nested, declared_type))
        return items

    path = str(entry).strip()
    if path:
        items.append((declared_type or "unknown", path))

    return items


def normalize_media(media: Any) -> List[Tuple[str, str]]:
    if not media:
        return []

    if isinstance(media, dict):
        collected: List[Tuple[str, str]] = []
        for key in ("image", "images", "video", "videos"):
            if key in media:
                declared = "image" if "image" in key else "video"
                collected.extend(_collect_media_entries(media[key], declared))
        if collected:
            return collected
        return _collect_media_entries(media)

    return _collect_media_entries(media)


class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        dtype_arg = getattr(args, "dtype")
        if isinstance(dtype_arg, str) and hasattr(torch, dtype_arg):
            dtype_arg = getattr(torch, dtype_arg)
        self.torch_dtype = dtype_arg or "auto"

        self.default_model_kwargs = {"device_map": "auto"}
        self.default_gen_kwargs = {
            "max_new_tokens": 8192,
            "repetition_penalty": 1.1,
            "top_k": 1,
            "top_p": 1e-5,
            "temperature": 1.0,
        }
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)

    def load_model(self, args):
        self.processor = AutoProcessor.from_pretrained(
            args.model_name_or_path,
            use_fast=True,
        )
        self.model = Glm4vForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            torch_dtype=self.torch_dtype,
            **self.model_kwargs,
        )
        self.model.eval()

    def parse_input(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = sample["prompt"]
        chunks = re.split(r"(<(?:image|video)>)", prompt)
        media_tokens = [chunk for chunk in chunks if chunk in PLACEHOLDER_TO_TYPE]
        media_entries = normalize_media(sample.get("media"))

        def pop_media(token: str) -> str:
            token_type = PLACEHOLDER_TO_TYPE[token]
            for idx, (declared_type, path) in enumerate(media_entries):
                if declared_type in (token_type, "unknown"):
                    media_entries.pop(idx)
                    return path
            raise ValueError(
                f"Encountered a {token_type} placeholder without a corresponding media path"
            )

        if len(media_entries) < len(media_tokens):
            raise ValueError("Not enough media items to satisfy prompt placeholders")

        messages: List[Dict[str, Any]] = [{"role": "user", "content": []}]

        for chunk in chunks:
            if not chunk:
                continue
            if chunk in PLACEHOLDER_TO_TYPE:
                media_path = pop_media(chunk)
                messages[0]["content"].append(
                    {"type": PLACEHOLDER_TO_TYPE[chunk], "url": media_path}
                )
            elif chunk.strip():
                messages[0]["content"].append({"type": "text", "text": chunk})

        if media_entries:
            raise ValueError("Received more media items than prompt placeholders")

        return messages

    def _generate_response(self, inputs: Dict[str, Any]) -> str:
        gen_kwargs = dict(self.gen_kwargs)
        temperature = gen_kwargs.get("temperature")
        if "do_sample" not in gen_kwargs:
            gen_kwargs["do_sample"] = bool(temperature and temperature > 0)

        if temperature is not None and temperature <= 0:
            gen_kwargs["temperature"] = None

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        input_length = inputs["input_ids"].shape[1]
        output_tokens = generated_ids[0][input_length:]
        if output_tokens.numel() > 0:
            output_tokens = output_tokens[:-1]

        return self.processor.decode(output_tokens, skip_special_tokens=False)

    def run_sample(self, sample: Dict[str, Any]):
        ori_sample = copy.deepcopy(sample)
        messages = self.parse_input(ori_sample)

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        inputs.pop("token_type_ids", None)

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(inputs)
        else:
            pass

        return ori_sample


if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
