import copy
import os
import re
from typing import Iterable, List, Sequence, Tuple

import torch
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoModel as HF_AutoModel
from transformers import AutoTokenizer as HF_AutoTokenizer

try:  # pragma: no cover
    from modelscope import AutoModel as MS_AutoModel
    from modelscope import AutoTokenizer as MS_AutoTokenizer
except ImportError:  # pragma: no cover
    MS_AutoModel = None
    MS_AutoTokenizer = None

from mmeval.infer.task import Task
from mmeval.utils.argparser import parse_args, parse_gen_kwargs, parse_model_kwargs

_TOKEN_PATTERN = re.compile(r"(<\|?image\|?>|<\|?video\|?>)", re.IGNORECASE)
_IMAGE_TOKENS = {"<image>", "<|image|>"}
_VIDEO_TOKENS = {"<video>", "<|video|>"}
_CANONICAL_IMAGE = "<|image|>"
_CANONICAL_VIDEO = "<|video|>"
_VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".mpg",
    ".mpeg",
    ".flv",
    ".wmv",
    ".m4v",
}
_MAX_FRAMES = 16


def _resolve_dtype(value, default):
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "auto":
            return default
        if not hasattr(torch, value):
            raise ValueError(f"Unsupported dtype specification: {value}")
        return getattr(torch, value)
    raise TypeError(f"Unsupported dtype specification: {value}")


def _uniform_sample(indices: Sequence[int], target_length: int) -> List[int]:
    if not indices:
        return []
    if len(indices) <= target_length:
        return list(indices)
    gap = len(indices) / target_length
    return [indices[min(int(i * gap + gap / 2), len(indices) - 1)] for i in range(target_length)]


def _load_image(source) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    with Image.open(source) as img:
        return img.convert("RGB")


def _load_video(path: str, max_frames: int = _MAX_FRAMES) -> List[Image.Image]:
    vr = VideoReader(path, ctx=cpu(0))
    total = len(vr)
    if total == 0:
        return []
    fps = vr.get_avg_fps()
    stride = max(int(round(fps)) or 1, 1)
    frame_indices = list(range(0, total, stride))
    if len(frame_indices) > max_frames:
        frame_indices = _uniform_sample(frame_indices, max_frames)
    frames = vr.get_batch(frame_indices).asnumpy()
    return [Image.fromarray(frame.astype("uint8")).convert("RGB") for frame in frames]


def _infer_media_type(value) -> str:
    if isinstance(value, dict):
        return (value.get("type") or "image").lower()
    if isinstance(value, str):
        extension = os.path.splitext(value)[1].lower()
        if extension in _VIDEO_EXTENSIONS:
            return "video"
    return "image"


def _flatten_media(media) -> List[Tuple[str, object]]:
    if media is None:
        return []
    if isinstance(media, (list, tuple)):
        flattened: List[Tuple[str, object]] = []
        for item in media:
            flattened.extend(_flatten_media(item))
        return flattened
    if isinstance(media, dict):
        media_type = (media.get("type") or "").lower()
        value = None
        for key in ("value", "path", "data", "content", "image", "video"):
            if key in media:
                value = media[key]
                break
        if isinstance(value, (list, tuple)):
            flattened: List[Tuple[str, object]] = []
            for item in value:
                flattened.extend(_flatten_media({"type": media_type, "value": item}))
            return flattened
        if value is None:
            return []
        if media_type not in {"image", "video"}:
            media_type = _infer_media_type(value)
        return [(media_type, value)]
    media_type = _infer_media_type(media)
    return [(media_type, media)]


def _pop_media(buckets, key: str, fallback: str):
    bucket = buckets.get(key, [])
    index = buckets.get(f"_{key}_index", 0)
    if index < len(bucket):
        buckets[f"_{key}_index"] = index + 1
        return bucket[index]
    if fallback:
        bucket = buckets.get(fallback, [])
        index = buckets.get(f"_{fallback}_index", 0)
        if index < len(bucket):
            buckets[f"_{fallback}_index"] = index + 1
            return bucket[index]
    return None


class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.model_id = args.model_name_or_path
        is_new = "241101" in (self.model_id or "")
        base_dtype = torch.bfloat16 if is_new else torch.float16
        default_attn = "flash_attention_2" if is_new else "sdpa"

        self.dtype = _resolve_dtype(getattr(args, "dtype", None), base_dtype)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.default_model_kwargs = {"attn_implementation": default_attn}
        self.default_gen_kwargs = {"max_new_tokens": 100, "do_sample": True, "top_k": 1}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)

    def load_model(self, args):
        self.tokenizer = None
        self.model = None

        if "241101" in (self.model_id or "") and MS_AutoModel is not None:
            try:  # pragma: no cover
                self.model = MS_AutoModel.from_pretrained(
                    self.model_id,
                    torch_dtype=self.dtype,
                    trust_remote_code=True,
                    **self.model_kwargs,
                )
                self.tokenizer = MS_AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            except Exception:  # pragma: no cover
                self.model = None
                self.tokenizer = None

        if self.model is None:
            self.model = HF_AutoModel.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                trust_remote_code=True,
                **self.model_kwargs,
            )
            self.tokenizer = HF_AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

        self.model = self.model.to(self.device).eval()
        self.processor = self.model.init_processor(self.tokenizer)

    def parse_input(self, sample: dict):
        prompt = sample.get("prompt") or sample.get("question") or ""
        media_entries = _flatten_media(sample.get("media"))
        buckets = {"image": [], "video": [], "generic": []}
        for media_type, value in media_entries:
            if media_type in ("image", "video"):
                buckets[media_type].append(value)
            else:
                buckets["generic"].append(value)
        buckets["_image_index"] = 0
        buckets["_video_index"] = 0
        buckets["_generic_index"] = 0

        images: List[object] = []
        videos: List[object] = []
        content_parts: List[str] = []

        for chunk in _TOKEN_PATTERN.split(prompt):
            token = chunk.lower()
            if token in _IMAGE_TOKENS:
                source = _pop_media(buckets, "image", "generic")
                if source is None:
                    raise ValueError("Found <image> token but media list is exhausted.")
                images.append(source)
                content_parts.append(_CANONICAL_IMAGE)
            elif token in _VIDEO_TOKENS:
                source = _pop_media(buckets, "video", "generic")
                if source is None:
                    raise ValueError("Found <video> token but media list is exhausted.")
                videos.append(source)
                content_parts.append(_CANONICAL_VIDEO)
            elif chunk:
                content_parts.append(chunk)

        user_content = "".join(content_parts)
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": ""},
        ]
        return messages, images, videos

    def _generate_response(self, inputs):
        with torch.inference_mode():
            outputs = self.model.generate(**inputs)
        if isinstance(outputs, torch.Tensor):
            outputs = outputs[0]
            return self.tokenizer.decode(outputs, skip_special_tokens=True)
        if isinstance(outputs, (list, tuple)):
            first = outputs[0]
            if isinstance(first, torch.Tensor):
                return self.tokenizer.decode(first, skip_special_tokens=True)
            if isinstance(first, str):
                return first
        return str(outputs)

    def _prepare_media(self, images: Iterable, videos: Iterable):
        image_inputs = [_load_image(img) for img in images]
        video_inputs = []
        for idx, video in enumerate(videos):
            if isinstance(video, (list, tuple)):
                frames = [_load_image(frame) for frame in video]
            else:
                frames = _load_video(video)
            if not frames:
                raise ValueError(f"Video payload at index {idx} produced no frames.")
            video_inputs.append(frames)
        return image_inputs, video_inputs

    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        messages, image_sources, video_sources = self.parse_input(ori_sample)
        image_inputs, video_inputs = self._prepare_media(image_sources, video_sources)

        processor_kwargs = {}
        if image_inputs:
            processor_kwargs["images"] = image_inputs
        if video_inputs:
            processor_kwargs["videos"] = video_inputs

        processor_inputs = self.processor(messages, **processor_kwargs)
        processor_inputs = processor_inputs.to(self.device)
        processor_inputs["tokenizer"] = self.tokenizer
        processor_inputs.setdefault("decode_text", True)
        for key, value in self.gen_kwargs.items():
            processor_inputs[key] = value

        if self.args.score_target:
            raise NotImplementedError("score_target is not supported for mPLUG-Owl3.")
        ori_sample["response"] = self._generate_response(processor_inputs)
        return ori_sample


if __name__ == "__main__":
    parsed_args = parse_args()
    runner = TaskRunner(parsed_args)
    runner.inference_dataset()
