import copy
import re
from typing import List, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import (
    parse_args,
    parse_gen_kwargs,
    parse_model_kwargs,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float, target_ratios: List[tuple], width: int, height: int, image_size: int
) -> tuple:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    *,
    min_num: int = 1,
    max_num: int = 10,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> List[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images: List[Image.Image] = []

    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images


def load_image(
    image_file,
    *,
    input_size: int = 448,
    min_num: int = 1,
    max_num: int = 10,
    use_thumbnail: bool = True,
) -> Tuple[torch.Tensor, int]:
    if isinstance(image_file, Image.Image):
        image = image_file.convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image,
        image_size=input_size,
        min_num=min_num,
        max_num=max_num,
        use_thumbnail=use_thumbnail,
    )
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values, pixel_values.size(0)


class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype_arg = getattr(args, "dtype", None)
        self.dtype = getattr(torch, dtype_arg, torch.bfloat16) if dtype_arg else torch.bfloat16
        self.image_size = 448
        self.min_tiles = 1
        self.max_tiles = 10
        self.use_thumbnail = True

        self.default_model_kwargs = {}
        self.default_gen_kwargs = {"max_new_tokens": 1024, "do_sample": True}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)
        super().__init__(args)

    def load_model(self, args):
        self.model = AutoModel.from_pretrained(
            args.model_name_or_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            **self.model_kwargs,
        ).eval()
        if torch.cuda.is_available():
            self.model = self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            use_fast=False,
        )

        config = getattr(self.model, "config", None)
        if config is not None:
            self.image_size = (
                getattr(config, "force_image_size", None)
                or getattr(getattr(config, "vision_config", None), "image_size", None)
                or self.image_size
            )
            self.min_tiles = getattr(config, "min_dynamic_patch", self.min_tiles)
            self.max_tiles = getattr(config, "max_dynamic_patch", self.max_tiles)
            self.use_thumbnail = getattr(config, "use_thumbnail", self.use_thumbnail)

    def parse_input(self, sample: dict):
        prompt = sample.get("prompt") or sample.get("question") or ""
        q_chunks = re.split(r"(<(?:image|video)>)", prompt)
        media = copy.deepcopy(sample.get("media", []))

        text_parts: List[str] = []
        pixel_batches: List[torch.Tensor] = []
        num_patches: List[int] = []

        for chunk in q_chunks:
            if chunk == "":
                continue
            if chunk == constants.image:
                if media:
                    media_file = media.pop(0)
                    pixels, patches = load_image(
                        media_file,
                        input_size=self.image_size,
                        min_num=self.min_tiles,
                        max_num=self.max_tiles,
                        use_thumbnail=self.use_thumbnail,
                    )
                    pixel_batches.append(pixels)
                    num_patches.append(patches)
                text_parts.append(constants.image)
            else:
                text_parts.append(chunk)

        while media:
            media_file = media.pop(0)
            pixels, patches = load_image(
                media_file,
                input_size=self.image_size,
                min_num=self.min_tiles,
                max_num=self.max_tiles,
                use_thumbnail=self.use_thumbnail,
            )
            pixel_batches.append(pixels)
            num_patches.append(patches)
            text_parts.append(constants.image)

        question = "".join(text_parts).strip()

        pixel_values = None
        if pixel_batches:
            pixel_values = torch.cat(pixel_batches, dim=0).to(self.dtype).to(self.device)

        return question, pixel_values, num_patches

    def _generate_response(
        self,
        prompt: str,
        pixel_values: torch.Tensor | None,
        num_patches_list: List[int],
    ) -> str:
        generation_config = copy.deepcopy(self.gen_kwargs)
        with torch.inference_mode():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                generation_config,
                num_patches_list=num_patches_list or None,
            )
        return response

    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        prompt, pixel_values, num_patches_list = self.parse_input(ori_sample)

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(
                prompt, pixel_values, num_patches_list
            )

        return ori_sample

    def _score_choices(self, sample, response):
        pass


if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
