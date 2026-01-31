import os
# Disable TorchDynamo before importing torch to avoid FailOnRecompileLimitHit error
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import re
import copy
import torch

from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.default_model_kwargs = {"device_map": "auto"}
        self.default_gen_kwargs = {"max_new_tokens": 100, "do_sample": False}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)
        
    def load_model(self, args):
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            args.model_name_or_path, **self.model_kwargs
        ).eval()
        self.processor = AutoProcessor.from_pretrained(args.model_name_or_path)
        
    def _parse_input(self, msg):
        """Parse a single message into user message format (without system message)."""
        prompt = msg["prompt"]
        # placeholder <>, can be image, video, etc.
        q_chunks = re.split(r'(<(?:image|video)>)', prompt)
        media = copy.deepcopy(msg["media"])

        user_message = {
            "role": "user",
            "content": []
        }

        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            if chunk == constants.image:
                media_file = media.pop(0)
                user_message["content"].append(
                    {
                        "type": "image",
                        "image": media_file
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

    def _generate_response(self, inputs, input_len):
        with torch.inference_mode():
            generation = self.model.generate(**inputs, **self.gen_kwargs)
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)

        return decoded
    
    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        responses = []
        # Initialize conversation history with system message
        conversation_history = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            }
        ]
        
        for msg in sample["messages"]:
            # Parse current user message and add to history
            user_message = self._parse_input(msg)
            conversation_history.append(user_message)

            inputs = self.processor.apply_chat_template(
                conversation_history, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(self.model.device, dtype=self.dtype)

            input_len = inputs["input_ids"].shape[-1]

            if not self.args.score_target:
                response = self._generate_response(inputs, input_len)
                responses.append(response)
                # Add assistant response to conversation history
                conversation_history.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": response}]
                })
            else:
                pass

        ori_sample["response"] = responses
        return ori_sample

    
if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
