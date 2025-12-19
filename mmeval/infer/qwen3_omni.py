import re
import copy

import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs

USE_AUDIO_IN_VIDEO = False


class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(args, "dtype") or "auto"
        self.default_model_kwargs = {"attn_implementation": "flash_attention_2", "device_map": "auto"}
        self.default_gen_kwargs = {"max_new_tokens": 128}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)
        
    def load_model(self, args):
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            dtype=self.dtype,
            **self.model_kwargs
        )
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_name_or_path)

    def parse_input(self, msg):
        question = msg["prompt"]
        q_chunks = re.split(r'(<(?:image|video)>)', question)
        media_list = copy.deepcopy(msg["media"])

        conversation = [
            {
                "role": "user",
                "content": []
            }
        ]

        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            if chunk == constants.image:
                media = media_list.pop(0)
                conversation[0]["content"].append(
                    {
                        "type": "image",
                        "image": media
                    }
                )       
            elif chunk == constants.video:
                media = media_list.pop(0)
                conversation[0]["content"].append(
                    {
                        "type": "video",
                        "video": media
                    }
                )
            else:
                conversation[0]["content"].append(
                    {
                        "type": "text",
                        "text": chunk
                    }
                )

        return conversation

    def _generate_response(self, inputs):
        text_ids, audio = self.model.generate(
            **inputs, 
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
            thinker_return_dict_in_generate=True,
            **self.gen_kwargs
        )

        text = self.processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return text

    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        responses = []
        conversation_history = []  # Accumulate conversation history for multi-turn chat
        
        for msg in sample["messages"]:
            # Parse current user message and add to history
            user_message = self.parse_input(msg)
            conversation_history.extend(user_message)

            text = self.processor.apply_chat_template(
                conversation_history, 
                add_generation_prompt=True, 
                tokenize=False
            )
            audios, images, videos = process_mm_info(
                conversation_history, 
                use_audio_in_video=USE_AUDIO_IN_VIDEO
            )
            inputs = self.processor(
                text=text, 
                audio=audios, 
                images=images, 
                videos=videos, 
                return_tensors="pt", 
                padding=True, 
                use_audio_in_video=USE_AUDIO_IN_VIDEO
            )
            inputs = inputs.to(self.model.device).to(self.model.dtype)

            if not self.args.score_target:
                response = self._generate_response(inputs)
                responses.append(response)
                # Add assistant response to conversation history
                conversation_history.append({
                    "role": "assistant",
                    "content": response[0]
                })
            else:
                pass

        ori_sample["response"] = responses
        return ori_sample


if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()


