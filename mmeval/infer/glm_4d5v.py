import copy
import os
import re
from pathlib import Path

import torch
from PIL import Image

from transformers import (
    AutoProcessor,
    Glm4vForConditionalGeneration,
    Glm4vMoeForConditionalGeneration,
)

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import (
    parse_args,
    parse_model_kwargs,
    parse_gen_kwargs,
)

_PLACEHOLDER_PATTERN = re.compile(r"(<(?:image|video)>)")
_VIDEO_EXTENSIONS = {
    ".avi",
    ".gif",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}


def _normalize_media_list(media):
    if media is None:
        return []
    if isinstance(media, (str, os.PathLike, Image.Image, dict)):
        media_items = [media]
    elif isinstance(media, (list, tuple)):
        media_items = list(media)
    else:
        raise TypeError(
            "Unsupported media container for GLM-4.5V inference: "
            f"{type(media).__name__}"
        )

    normalized = []
    for item in media_items:
        if item is None:
            continue

        media_type = None
        payload = item

        if isinstance(item, dict):
            media_type = item.get("type") or item.get("media_type")
            for key in ("image", "video", "value", "url", "path"):
                if item.get(key) is not None:
                    payload = item[key]
                    if key in {"image", "video"} and not media_type:
                        media_type = key
                    break

        if payload is None:
            raise TypeError("Missing media payload for GLM-4.5V inference")

        if isinstance(payload, os.PathLike):
            payload = os.fspath(payload)

        if isinstance(payload, Image.Image):
            resolved_type = "image"
        else:
            resolved_type = None
            if isinstance(media_type, str):
                lowered = media_type.lower()
                if "video" in lowered:
                    resolved_type = "video"
                elif any(tag in lowered for tag in ("image", "img", "photo", "pic")):
                    resolved_type = "image"

            if resolved_type is None and isinstance(payload, str):
                suffix = Path(payload).suffix.lower()
                resolved_type = "video" if suffix in _VIDEO_EXTENSIONS else "image"

            if resolved_type is None:
                resolved_type = "image"

        normalized.append({"type": resolved_type, "value": payload})

    return normalized


def _as_chat_content(entry):
    payload = entry["value"]
    if entry["type"] == "image" and isinstance(payload, Image.Image):
        return {"type": "image", "image": payload}
    return {"type": entry["type"], "url": payload}


class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.default_model_kwargs = {"torch_dtype": "auto", "device_map": "auto"}
        self.default_gen_kwargs = {
            "max_new_tokens": 8192,
            "temperature": 1.0,
            "repetition_penalty": 1.1,
            "top_p": 1e-5,
            "top_k": 1,
        }
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)

    def load_model(self, args):
        self.processor = AutoProcessor.from_pretrained(args.model_name_or_path)

        model_cls = (
            Glm4vMoeForConditionalGeneration
            if "4.5" in args.model_name_or_path
            else Glm4vForConditionalGeneration
        )

        self.model = model_cls.from_pretrained(
            args.model_name_or_path,
            **self.model_kwargs,
        ).eval()

    def parse_input(self, sample: dict):
        prompt = sample.get("prompt", "")
        normalized_media = _normalize_media_list(copy.deepcopy(sample.get("media")))

        messages = [{"role": "user", "content": []}]
        content = messages[0]["content"]

        if _PLACEHOLDER_PATTERN.search(prompt):
            media_queue = list(normalized_media)
            for chunk in _PLACEHOLDER_PATTERN.split(prompt):
                if not chunk:
                    continue
                if chunk == constants.image:
                    if not media_queue:
                        raise ValueError(
                            "Encountered <image> token without corresponding media entry."
                        )
                    entry = media_queue.pop(0)
                    if entry["type"] != "image":
                        raise ValueError(
                            "Expected image media for <image> token but received "
                            f"{entry['type']}"
                        )
                    content.append(_as_chat_content(entry))
                elif chunk == constants.video:
                    if not media_queue:
                        raise ValueError(
                            "Encountered <video> token without corresponding media entry."
                        )
                    entry = media_queue.pop(0)
                    if entry["type"] != "video":
                        raise ValueError(
                            "Expected video media for <video> token but received "
                            f"{entry['type']}"
                        )
                    content.append(_as_chat_content(entry))
                else:
                    content.append({"type": "text", "text": chunk})

            if media_queue:
                raise ValueError(
                    "Unused media entries remain after parsing the prompt; check that "
                    "media placeholders and media inputs align."
                )
        else:
            for entry in normalized_media:
                content.append(_as_chat_content(entry))
            if prompt:
                content.append({"type": "text", "text": prompt})

        if not content and prompt:
            content.append({"type": "text", "text": prompt})

        return messages

    def _generate_response(self, inputs):
        generation_kwargs = dict(self.gen_kwargs)
        temperature = generation_kwargs.get("temperature")
        generation_kwargs["do_sample"] = bool(temperature and temperature > 0)
        if temperature is not None and temperature <= 0:
            generation_kwargs.pop("temperature", None)

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        decoded = self.processor.decode(
            generated_ids[0][prompt_len:-1], skip_special_tokens=False
        )
        match = re.search(r"<answer>(.*?)</answer>", decoded, re.DOTALL)
        return match.group(1).strip() if match else decoded

    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        messages = self.parse_input(ori_sample)

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
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
