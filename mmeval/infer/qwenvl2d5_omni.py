import re
import copy

import torch
import numpy as np
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens

# set use audio in video
USE_AUDIO_IN_VIDEO = False

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(args, "dtype") or "auto"
        self.default_model_kwargs = {"attn_implementation":"flash_attention_2", "device_map": "auto"}
        self.default_gen_kwargs = {"max_new_tokens": 128}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)
        
    def load_model(self, args):
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(args.model_name_or_path, torch_dtype=self.dtype, **self.model_kwargs)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(args.model_name_or_path)

    def parse_input(self, msg):
        """Parse a single message into user message format (without system message)."""
        question = msg["prompt"]
        # placeholder <>, can be image, video, etc.
        q_chunks = re.split(r'(<(?:image|video)>)', question)
        media_list = copy.deepcopy(msg["media"])

        user_message = {
            "role": "user",
            "content": []
        }

        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            if chunk == constants.image:
                media = media_list.pop(0)
                user_message["content"].append(
                    {
                        "type": "image",
                        "image": media
                    }
                )       
            elif chunk == constants.video:
                media = media_list.pop(0)
                user_message["content"].append(
                    {
                        "type": "video",
                        "video": media,
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    }
                )
            else:
                user_message["content"].append(
                    {
                        "type": "text",
                        "text": chunk
                    }
                )

        return user_message

    def _generate_response(self, inputs):
        # Inference: Generation of the output text and audio
        text_ids, audio = self.model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, **self.gen_kwargs)

        text = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return text

    # def _score_choices(self, text, image_inputs, video_inputs, sample):
    #     contents = sample.get("choices")
    #     full = [text + content for content in contents]

    #     full_encoded = [self.processor(text=i, images=image_inputs, videos=video_inputs, return_tensors="pt").to(self.device) for i in full]
    #     prompt_encoded = self.processor(text=text, images=image_inputs, videos=video_inputs, return_tensors="pt").to(self.device)
    #     target_toks = target_tokens(self.tokenizer, contents)

    #     scorer = IncrementalLMScorer(self.model, self.device, tokenizer=self.tokenizer)
    #     scores = scorer.conditional_score(target_toks, full_encoded, prompt_encoded)
        
    #     return {
    #         "score": scores,
    #         "response": contents[np.argmax(scores)]
    #     }

    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        responses = []
        # Initialize conversation history with system message
        conversation_history = [{
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],
        }]
        
        for msg in sample["messages"]:
            # Parse current user message and add to history
            user_message = self.parse_input(msg)
            conversation_history.append(user_message)

            # Preparation for inference using full conversation history
            text = self.processor.apply_chat_template(conversation_history, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation_history, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
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
