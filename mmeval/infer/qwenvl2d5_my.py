import re
import copy

import torch
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens

SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer>...</answer> tags that directly addresses the question."
    "Example:<think>\nThis is my reasoning.\n</think>\n<answer>\nThis is my answer.\n</answer>.\n"
)

# Appended to the user turn so the eval prompt matches OPSD training
# (data_collator.py / data_collator_swa.py student message ends with this exact line).
USER_ANSWER_INSTRUCTION = (
    "\n\nPlease reason step by step, and put your final answer within <answer>...</answer>."
)

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if "AWQ" in args.model_name_or_path or "GPTQ" in args.model_name_or_path:
            self.dtype = torch.float16
        else:
            self.dtype = getattr(args, "dtype") or "auto"
            
        self.default_model_kwargs = {"attn_implementation":"flash_attention_2", "device_map": "auto"}
        self.default_gen_kwargs = {"max_new_tokens": 128}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)
        
    def load_model(self, args):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_name_or_path, torch_dtype=self.dtype, **self.model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        # Aligned with VLMEvalKit: do NOT pass min/max_pixels, so the processor's
        # defaults apply (min_pixels=3136, max_pixels=12845056 ≈ 12.8M). The previous
        # cap of max_pixels=1280*28*28 (~1MP) was ~12.8x lower and downscaled images.
        self.processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    def parse_input(self, message):
        question = message["prompt"]
        # placeholder <>, can be image, video, etc.
        q_chunks = re.split(r'(<(?:image|video)>)', question)
        media_list = message.get('media', [])

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT}
                ],
            },
            {
                "role": "user",
                "content": []
            }
        ]

        media_idx = 0
        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            if chunk == constants.image:
                media = media_list[media_idx]
                media_idx += 1
                messages[1]["content"].append(
                    {
                        "type": "image",
                        "image": media
                    }
                )       
            elif chunk == constants.video:
                media = media_list[media_idx]
                media_idx += 1
                messages[1]["content"].append(
                    {
                        "type": "video",
                        "video": media,
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    }
                )
            else:
                messages[1]["content"].append(
                    {
                        "type": "text",
                        "text": chunk
                    }
                )

        # Align with training: end the user turn with the same reasoning/answer-format
        # instruction the OPSD collators append after the problem.
        messages[1]["content"].append(
            {
                "type": "text",
                "text": USER_ANSWER_INSTRUCTION,
            }
        )

        return messages

    def _generate_response(self, inputs):
        generated_ids = self.model.generate(**inputs, **self.gen_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text

    def _score_choices(self, text, image_inputs, video_inputs, message):
        contents = message["choices"]
        full = [text + content for content in contents]

        full_encoded = [self.processor(text=i, images=image_inputs, videos=video_inputs, return_tensors="pt").to(self.device) for i in full]
        prompt_encoded = self.processor(text=text, images=image_inputs, videos=video_inputs, return_tensors="pt").to(self.device)
        target_toks = target_tokens(self.tokenizer, contents)

        scorer = IncrementalLMScorer(self.model, self.device, tokenizer=self.tokenizer)
        scores = scorer.conditional_score(target_toks, full_encoded, prompt_encoded)
        
        return {
            "score": scores,
            "response": contents[np.argmax(scores)]
        }

    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        message = sample["messages"][0]
        
        user_message = self.parse_input(message)
        
        text = self.processor.apply_chat_template(
            user_message, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(user_message, return_video_kwargs=True)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to(self.device)

        if not self.args.score_target:
            response = self._generate_response(inputs)
            ori_sample["messages"].append({"role": "assistant", "response": response})
        else:
            ori_sample.update(self._score_choices(text, image_inputs, video_inputs, message))

        return ori_sample


if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
