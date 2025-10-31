import copy
import re
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from qwen_vl_utils import process_vision_info

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import (
    parse_args,
    parse_gen_kwargs,
    parse_model_kwargs,
)
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens

PLACEHOLDER_PATTERN = re.compile(r"(<(?:image|video)>)")


def _split_media(media: Any) -> List[Any]:
    if isinstance(media, (list, tuple)):
        return list(media)
    if media is None:
        return []
    return [media]


def _next_media_payload(media_items: List[Any], placeholder: str) -> Optional[Dict[str, Any]]:
    if not media_items:
        return None

    payload_type = "image" if placeholder == constants.image else "video"
    key = "image" if payload_type == "image" else "video"
    media = media_items.pop(0)

    if isinstance(media, dict):
        result = copy.deepcopy(media)
        value = result.get(key)
        if value is None:
            value = result.pop("value", None)
        if value is None:
            value = result.get("path")
        if value is None:
            return None
        result.setdefault("type", payload_type)
        result[key] = value
        return result

    if (
        isinstance(media, (tuple, list))
        and len(media) == 2
        and isinstance(media[1], dict)
    ):
        value, extra = media
        result = copy.deepcopy(extra)
        result.update({"type": payload_type, key: value})
        return result

    return {"type": payload_type, key: media}


def _expand_text(text: str, media_items: List[Any]) -> List[Dict[str, Any]]:
    pieces: List[Dict[str, Any]] = []
    for chunk in PLACEHOLDER_PATTERN.split(text or ""):
        if not chunk:
            continue
        if chunk in {constants.image, constants.video}:
            payload = _next_media_payload(media_items, chunk)
            if payload is None:
                pieces.append({"type": "text", "text": chunk})
            else:
                pieces.append(payload)
        else:
            pieces.append({"type": "text", "text": chunk})
    return pieces


class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype_name = getattr(args, "dtype", None) or "auto"
        self.torch_dtype = dtype_name if dtype_name == "auto" else getattr(torch, dtype_name)
        self.model_kwargs = parse_model_kwargs(args, {"device_map": "auto"})
        self.gen_kwargs = parse_gen_kwargs(args, {"max_new_tokens": 128})

        super().__init__(args)

    def load_model(self, args):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            torch_dtype=self.torch_dtype,
            **self.model_kwargs,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(args.model_name_or_path)
        self.tokenizer = self.processor.tokenizer

    def parse_input(self, sample: dict) -> List[Dict[str, Any]]:
        media_queue = copy.deepcopy(_split_media(sample.get("media")))

        if "messages" in sample:
            raw_messages = copy.deepcopy(sample["messages"])
        else:
            raw_messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": sample.get("prompt", "")}],
                }
            ]

        normalized: List[Dict[str, Any]] = []

        for message in raw_messages:
            content: List[Dict[str, Any]] = []
            for item in message.get("content", []):
                item_type = item.get("type")
                if item_type == "text":
                    content.extend(_expand_text(item.get("text", ""), media_queue))
                elif item_type in {"image", "video"}:
                    key = "image" if item_type == "image" else "video"
                    resolved = copy.deepcopy(item)
                    if resolved.get(key) is None:
                        placeholder = constants.image if item_type == "image" else constants.video
                        resolved = _next_media_payload(media_queue, placeholder)
                    if resolved is None:
                        fallback = constants.image if item_type == "image" else constants.video
                        content.append({"type": "text", "text": fallback})
                    else:
                        content.append(resolved)
                else:
                    content.append(copy.deepcopy(item))

            normalized.append({"role": message.get("role", "user"), "content": content})

        if not normalized:
            normalized = [
                {
                    "role": "user",
                    "content": _expand_text("", media_queue),
                }
            ]

        return normalized

    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        messages = self.parse_input(sample)

        chat_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        has_vision = any(
            isinstance(item, dict)
            and item.get("type") in {"image", "video"}
            for message in messages
            for item in message.get("content", [])
        )

        if has_vision:
            image_patch = getattr(self.processor.image_processor, "patch_size", 16)
            if isinstance(image_patch, (tuple, list)):
                image_patch = image_patch[0]
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=image_patch,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
        else:
            image_inputs = None
            video_inputs = None
            video_kwargs = None

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(
                chat_prompt, image_inputs, video_inputs, video_kwargs
            )
        else:
            ori_sample.update(
                self._score_choices(
                    chat_prompt, image_inputs, video_inputs, video_kwargs, sample
                )
            )

        return ori_sample

    def _generate_response(
        self,
        chat_prompt: str,
        image_inputs: Optional[List[Any]],
        video_inputs: Optional[List[Any]],
        video_kwargs: Optional[Dict[str, Any]],
    ):
        inputs = self._prepare_inputs(
            chat_prompt,
            image_inputs,
            video_inputs,
            video_kwargs,
        )

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                **self.gen_kwargs,
            )

        input_ids = inputs["input_ids"]
        generated_ids_trimmed = [
            output[len(src) :]
            for src, output in zip(input_ids, generated_ids)
        ]

        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return response

    def _score_choices(
        self,
        chat_prompt: str,
        image_inputs: Optional[List[Any]],
        video_inputs: Optional[List[Any]],
        video_kwargs: Optional[Dict[str, Any]],
        sample: dict,
    ):
        choices = sample.get("choices", [])
        if not choices:
            return {"response": "", "score": []}

        prompt_encoded = self._prepare_inputs(
            chat_prompt,
            image_inputs,
            video_inputs,
            video_kwargs,
        )

        full_encoded = [
            self._prepare_inputs(
                chat_prompt + choice,
                image_inputs,
                video_inputs,
                video_kwargs,
            )
            for choice in choices
        ]

        target_toks = target_tokens(self.tokenizer, choices)
        model_device = getattr(self.model, "device", self.device)
        if isinstance(model_device, str):
            model_device = torch.device(model_device)

        scorer = IncrementalLMScorer(
            self.model,
            model_device,
            tokenizer=self.tokenizer,
        )
        scores = scorer.conditional_score(target_toks, full_encoded, prompt_encoded)
        best_idx = max(range(len(scores)), key=lambda idx: scores[idx])

        return {"score": scores, "response": choices[best_idx]}

    def _prepare_inputs(
        self,
        text: str,
        image_inputs: Optional[List[Any]],
        video_inputs: Optional[List[Any]],
        video_kwargs: Optional[Dict[str, Any]],
    ):
        processor_kwargs: Dict[str, Any] = {
            "text": [text],
            "return_tensors": "pt",
            "do_resize": False,
        }

        if image_inputs is not None:
            processor_kwargs["images"] = image_inputs

        if video_inputs is not None:
            if video_inputs and isinstance(video_inputs[0], tuple):
                videos, metadata = zip(*video_inputs)
                processor_kwargs["videos"] = list(videos)
                processor_kwargs["video_metadata"] = list(metadata)
            else:
                processor_kwargs["videos"] = video_inputs

        if video_kwargs:
            processor_kwargs.update(video_kwargs)

        inputs = self.processor(**processor_kwargs)

        model_device = getattr(self.model, "device", self.device)
        if isinstance(model_device, str):
            model_device = torch.device(model_device)

        return inputs.to(model_device)


if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
