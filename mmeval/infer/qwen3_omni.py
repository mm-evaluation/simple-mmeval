import re
import copy
import torch
import numpy as np
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

from qwen_omni_utils import process_mm_info

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens


class TaskRunner(Task):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, args):
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_name_or_path)

    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        messages = self.parse_input(sample)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(text, images, videos)
        else:
            ori_sample.update(self._score_choices(text, images, videos, sample))

        return ori_sample

    def _generate_response(self, text, images, videos):
        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
            use_audio_in_video=False
        )
        inputs = inputs.to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=256, use_audio_in_video=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        return output_text

    def _score_choices(self, text, images, videos, sample):
        contents = sample.get("choices")
        full = [text + content for content in contents]

        full_encoded = [
            self.processor(text=i, images=images, videos=videos, return_tensors="pt", use_audio_in_video=False).to(
                self.device)
            for i in full
        ]
        prompt_encoded = self.processor(text=text, images=images, videos=videos, return_tensors="pt",
                                         use_audio_in_video=False).to(self.device)
        target_toks = target_tokens(self.processor.tokenizer, contents)

        scorer = IncrementalLMScorer(self.model, self.device, tokenizer=self.processor.tokenizer)
        scores = scorer.conditional_score(target_toks, full_encoded, prompt_encoded)

        return {
            "score": scores,
            "response": contents[np.argmax(scores)]
        }

    def parse_input(self, sample: dict):
        question = sample["prompt"]
        q_chunks = re.split(r'(<(?:image|video|audio)>)', question)
        media_list = copy.deepcopy(sample['media'])

        messages = [
            {
                "role": "user",
                "content": []
            }
        ]

        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue

            if any(p in chunk for p in constants.all):
                if chunk == constants.audio:
                    raise ValueError(
                        f"Audio input is not supported for Qwen3-Omni in this configuration. Only image and video are supported.")

                if chunk == constants.image:
                    media_file = media_list.pop(0)
                    messages[0]["content"].append(
                        {
                            "type": "image",
                            "image": media_file
                        }
                    )
                elif chunk == constants.video:
                    media_file = media_list.pop(0)
                    messages[0]["content"].append(
                        {
                            "type": "video",
                            "video": media_file
                        }
                    )
            else:
                messages[0]["content"].append(
                    {
                        "type": "text",
                        "text": chunk
                    }
                )

        return messages


if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
