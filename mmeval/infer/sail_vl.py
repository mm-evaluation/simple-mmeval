import copy
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens

try:  # pragma: no cover - optional dependency for Qwen-style helpers
    from qwen_vl_utils import process_vision_info as qwen_process_vision_info  # type: ignore
except Exception:  # pragma: no cover - graceful fallback when helper is absent
    qwen_process_vision_info = None


def _load_image(image: Any) -> Any:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    return image


def _fallback_process_vision_info(messages: List[Dict[str, Any]]) -> Tuple[List[Any], List[Any]]:
    image_inputs: List[Any] = []
    video_inputs: List[Any] = []

    for message in messages:
        for content in message.get("content", []):
            if content.get("type") == "image" and "image" in content:
                image_inputs.append(_load_image(content["image"]))
            elif content.get("type") == "video" and "video" in content:
                video_inputs.append(content["video"])

    return image_inputs, video_inputs


def process_vision_info(messages: List[Dict[str, Any]]) -> Tuple[List[Any], List[Any]]:
    if qwen_process_vision_info is not None:
        return qwen_process_vision_info(messages)
    return _fallback_process_vision_info(messages)


class TaskRunner(Task):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, args):
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
        )

        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path,
                trust_remote_code=True,
            )
        except Exception:
            tokenizer = getattr(self.processor, "tokenizer", None)
        self.tokenizer = tokenizer

    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        messages = self.parse_input(sample)

        text = self._apply_chat_template(messages)
        image_inputs, video_inputs = process_vision_info(messages)

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(text, image_inputs, video_inputs)
        else:
            ori_sample.update(self._score_choices(text, image_inputs, video_inputs, sample))

        return ori_sample

    def _prepare_inputs(self, text: Any, image_inputs: List[Any], video_inputs: List[Any]):
        processor_kwargs: Dict[str, Any] = {
            "return_tensors": "pt",
        }

        if isinstance(text, list):
            processor_kwargs["text"] = text
        else:
            processor_kwargs["text"] = [text]

        if image_inputs:
            processor_kwargs["images"] = image_inputs
        if video_inputs:
            processor_kwargs["videos"] = video_inputs

        inputs = self.processor(**processor_kwargs)
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        return inputs

    def _generate_response(self, text, image_inputs, video_inputs):
        inputs = self._prepare_inputs(text, image_inputs, video_inputs)
        generation_kwargs = {"max_new_tokens": getattr(self.args, "max_new_tokens", 512)}
        generated_ids = self.model.generate(**inputs, **generation_kwargs)

        input_lengths = [len(in_ids) for in_ids in inputs.input_ids]
        generated_ids_trimmed = [
            output_ids[input_length:]
            for output_ids, input_length in zip(generated_ids, input_lengths)
        ]

        tokenizer = getattr(self.processor, "tokenizer", None)
        decoder = tokenizer or self.tokenizer

        if decoder is None:
            raise RuntimeError("Tokenizer is required for decoding SAIL-VL outputs.")

        output_text = decoder.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return output_text

    def _score_choices(self, text, image_inputs, video_inputs, sample):
        contents = sample.get("choices")
        if not contents:
            raise ValueError("Scoring requested but no choices were provided.")

        prompt_encoded = self._prepare_inputs(text, image_inputs, video_inputs)

        full = [text + choice for choice in contents]
        full_encoded = [
            self._prepare_inputs(choice_text, image_inputs, video_inputs)
            for choice_text in full
        ]

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is required for scoring SAIL-VL choices.")

        target_toks = target_tokens(self.tokenizer, contents)
        scorer = IncrementalLMScorer(self.model, self.device, tokenizer=self.tokenizer)
        scores = scorer.conditional_score(target_toks, full_encoded, prompt_encoded)

        return {
            "score": scores,
            "response": contents[int(np.argmax(scores))],
        }

    def parse_input(self, sample: dict):
        question = sample["prompt"]
        q_chunks = re.split(r'(<(?:image|video)>)', question)
        images = copy.deepcopy(sample["media"])

        messages = [
            {
                "role": "user",
                "content": [],
            }
        ]

        for chunk in q_chunks:
            if not chunk.strip():
                continue

            if any(placeholder in chunk for placeholder in constants.all):
                assert chunk == constants.image, f"Unsupported placeholder {chunk}"
                media_file = images.pop(0)
                messages[0]["content"].append(
                    {
                        "type": "image",
                        "image": media_file,
                    }
                )
            else:
                messages[0]["content"].append(
                    {
                        "type": "text",
                        "text": chunk,
                    }
                )
        return messages

    def _apply_chat_template(self, messages: List[Dict[str, Any]]) -> str:
        processor_apply = getattr(self.processor, "apply_chat_template", None)
        if callable(processor_apply):
            return processor_apply(messages, tokenize=False, add_generation_prompt=True)

        tokenizer_apply = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(tokenizer_apply):
            return tokenizer_apply(messages, tokenize=False, add_generation_prompt=True)

        parts: List[str] = []
        for message in messages:
            role = message.get("role", "user")
            parts.append(f"{role}: ")
            for content in message.get("content", []):
                if content.get("type") == "text":
                    parts.append(content.get("text", ""))
            parts.append("\n")
        parts.append("assistant: ")
        return "".join(parts)


if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
