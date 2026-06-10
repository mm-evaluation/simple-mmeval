"""minicpm_v_4d5 — MiniCPM-V-4_5.

HF: https://huggingface.co/openbmb/MiniCPM-V-4_5
GH: https://github.com/OpenBMB/MiniCPM-o
"""
import re
import copy

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs


class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.default_model_kwargs = {
            "attn_implementation": "sdpa",
        }
        self.default_gen_kwargs = {"max_new_tokens": 1024}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)

    def load_model(self, args):
        self.model = AutoModel.from_pretrained(
            args.model_name_or_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            **self.model_kwargs,
        ).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True,
        )

    def parse_input(self, message):
        question = message["prompt"]
        q_chunks = re.split(r'(<(?:image|video)>)', question)
        media_list = message.get('media', [])

        content = []
        text_parts = []
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
                content.append(img)
                media_idx += 1
            elif chunk == constants.video:
                raise NotImplementedError("minicpm_v_4d5 video input not implemented")
            else:
                text_parts.append(chunk)
        content.append("".join(text_parts).strip())
        return [{"role": "user", "content": content}]

    def run_sample(self, sample: dict):
        if self.args.score_target:
            raise NotImplementedError(
                "minicpm_v_4d5: score_target is not implemented yet"
            )

        ori_sample = copy.deepcopy(sample)
        message = sample["messages"][0]
        msgs = self.parse_input(message)

        response = self.model.chat(
            msgs=msgs,
            tokenizer=self.tokenizer,
            enable_thinking=False,
            stream=False,
            **self.gen_kwargs,
        )
        ori_sample["messages"].append({"role": "assistant", "response": response})
        return ori_sample


if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
