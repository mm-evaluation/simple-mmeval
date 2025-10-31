import copy
import re
from collections import deque

import torch

from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs


class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.dtype = getattr(args, "dtype") or "auto"
        self.use_audio_in_video = True
        self.default_speaker = "Ethan"
        self.default_model_kwargs = {
            "device_map": "auto",
            "attn_implementation": "flash_attention_2",
        }
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, {})

        super().__init__(args)

    def load_model(self, args):
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            dtype=self.dtype,
            **self.model_kwargs,
        )

        self.processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_name_or_path)

        self.talker_enabled = True
        suffix = args.model_name_or_path.split("/")[-1]
        if any(tag in suffix for tag in ("Thinking", "Captioner")):
            if hasattr(self.model, "disable_talker"):
                self.model.disable_talker()
            self.talker_enabled = False

    def parse_input(self, sample: dict):
        prompt = sample["prompt"]
        chunks = [c for c in re.split(r"(<(?:image|video|audio)>)", prompt) if c]
        media_entries = sample.get("media")

        placeholder_to_type = {
            constants.image: "image",
            constants.video: "video",
            constants.audio: "audio",
        }

        def ensure_list(value):
            if value in (None, ""):
                return []
            if isinstance(value, list):
                return copy.deepcopy(value)
            return [copy.deepcopy(value)]

        if isinstance(media_entries, dict):
            for alias in ("items", "contents", "data"):
                container = media_entries.get(alias)
                if isinstance(container, list):
                    media_entries = container
                    break

        if isinstance(media_entries, dict):
            normalized = {
                placeholder: deque(
                    ensure_list(
                        media_entries.get(media_type)
                        or media_entries.get(f"{media_type}s")
                    )
                )
                for placeholder, media_type in placeholder_to_type.items()
            }

            def pop_media(token):
                queue = normalized[token]
                if not queue:
                    raise ValueError("Insufficient media provided for the prompt placeholders.")
                return queue.popleft()

            def assert_unused():
                leftovers = [len(queue) for queue in normalized.values()]
                if any(leftovers):
                    raise ValueError("Unused media items remain after parsing the prompt.")

        else:
            flat_media = deque(ensure_list(media_entries))

            def pop_media(_token):
                if not flat_media:
                    raise ValueError("Insufficient media provided for the prompt placeholders.")
                return flat_media.popleft()

            def assert_unused():
                if flat_media:
                    raise ValueError("Unused media items remain after parsing the prompt.")

        def build_media_content(token, item):
            media_type = placeholder_to_type[token]

            if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], dict):
                payload = copy.deepcopy(item[1])
                payload.setdefault("type", media_type)
                payload.setdefault(media_type, item[0])
                return payload

            if isinstance(item, dict):
                payload = copy.deepcopy(item)
                payload.setdefault("type", media_type)
                payload.setdefault(
                    media_type,
                    payload.get(media_type)
                    or payload.get("value")
                    or payload.get("path")
                    or payload.get("url")
                    or payload.get("data"),
                )
                return payload

            return {"type": media_type, media_type: item}

        contents = []
        for chunk in chunks:
            if chunk in constants.all:
                contents.append(build_media_content(chunk, pop_media(chunk)))
            else:
                contents.append({"type": "text", "text": chunk})

        assert_unused()

        if not contents:
            contents.append({"type": "text", "text": prompt})

        return [{"role": "user", "content": contents}]

    def _generate_response(self, inputs):
        inputs = inputs.to(self.model.device)

        gen_kwargs = {
            **self.gen_kwargs,
            "thinker_return_dict_in_generate": True,
            "use_audio_in_video": self.use_audio_in_video,
        }
        if self.talker_enabled:
            gen_kwargs.setdefault("speaker", self.default_speaker)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        sequences = getattr(outputs[0] if isinstance(outputs, tuple) else outputs, "sequences", outputs)
        generated_ids = sequences[:, inputs["input_ids"].shape[1]:]

        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return response

    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        messages = self.parse_input(ori_sample)

        chat_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        audios, images, videos = process_mm_info(messages, use_audio_in_video=self.use_audio_in_video)

        processor_kwargs = {
            "text": chat_text,
            "return_tensors": "pt",
            "padding": True,
            "use_audio_in_video": self.use_audio_in_video,
        }
        if audios is not None:
            processor_kwargs["audio"] = audios
        if images is not None:
            processor_kwargs["images"] = images
        if videos is not None:
            processor_kwargs["videos"] = videos

        inputs = self.processor(**processor_kwargs)

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(inputs)
        else:
            pass

        return ori_sample


if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
