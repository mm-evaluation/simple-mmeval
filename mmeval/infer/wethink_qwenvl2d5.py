import copy
import re
from typing import List

import numpy as np
import torch
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

from qwen_vl_utils import process_vision_info

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_gen_kwargs, parse_model_kwargs
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens


SYSTEM_PROMPT = (
    "You are WeThink-Qwen2.5VL, a helpful assistant with excellent multimodal "
    "reasoning skills. First reason carefully inside <think> </think> tags and "
    "then provide the final answer inside <answer> </answer> tags."
)

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def _resolve_dtype(dtype, device: torch.device):
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        lowered = dtype.strip().lower()
        if lowered in {"auto", "none", ""}:
            return "auto"
        if lowered in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if lowered in {"float16", "fp16", "half"}:
            return torch.float16
        if lowered in {"float32", "fp32"}:
            return torch.float32
    return torch.bfloat16 if device.type == "cuda" else torch.float32


class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(args, "dtype", None) or "auto"
        self.default_model_kwargs = {"attn_implementation": "flash_attention_2"}
        self.default_gen_kwargs = {
            "max_new_tokens": 1024,
            "do_sample": False,
            "use_cache": True,
        }
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)

    def load_model(self, args):
        model_kwargs = dict(self.model_kwargs)
        if self.device.type == "cuda" and "device_map" not in model_kwargs:
            model_kwargs["device_map"] = "auto"

        torch_dtype = _resolve_dtype(self.dtype, self.device)

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch_dtype,
            **model_kwargs,
        )

        if "device_map" not in model_kwargs:
            self.model.to(self.device)

        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    def _build_messages(self, sample: dict) -> List[dict]:
        question = sample["prompt"]
        q_chunks = re.split(r'(<(?:image|video)>)', question)
        images = copy.deepcopy(sample["media"])

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [],
            },
        ]

        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue

            if chunk in constants.all:
                assert chunk == constants.image, f"Unsupported placeholder {chunk}"
                media_file = images.pop(0)
                messages[1]["content"].append(
                    {
                        "type": "image",
                        "image": media_file,
                    }
                )
            else:
                messages[1]["content"].append(
                    {
                        "type": "text",
                        "text": chunk,
                    }
                )

        return messages

    def _prepare_prompt(self, messages: List[dict]):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        return text, image_inputs, video_inputs

    def _generate_response(self, messages: List[dict]):
        text, image_inputs, video_inputs = self._prepare_prompt(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **self.gen_kwargs)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return output_text

    def _extract_final_answer(self, text: str) -> str:
        match = ANSWER_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        return text.strip()

    def _score_choices(self, text, image_inputs, video_inputs, sample):
        contents = sample.get("choices")
        assert contents, "choices are required when score_target is enabled"

        full = [text + content for content in contents]

        full_encoded = [
            self.processor(
                text=i,
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
            ).to(self.device)
            for i in full
        ]
        prompt_encoded = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        ).to(self.device)

        target_toks = target_tokens(self.tokenizer, contents)

        scorer = IncrementalLMScorer(self.model, self.device, tokenizer=self.tokenizer)
        scores = scorer.conditional_score(target_toks, full_encoded, prompt_encoded)

        best_idx = int(np.argmax(scores))

        return {
            "score": scores,
            "response": contents[best_idx],
        }

    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        messages = self._build_messages(sample)

        if not self.args.score_target:
            response = self._generate_response(messages)
            ori_sample["response"] = response
            ori_sample.setdefault("metadata", {})
            ori_sample["metadata"]["parsed_answer"] = self._extract_final_answer(response)
        else:
            text, image_inputs, video_inputs = self._prepare_prompt(messages)
            ori_sample.update(self._score_choices(text, image_inputs, video_inputs, sample))

        return ori_sample


if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
