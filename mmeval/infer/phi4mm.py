import re
import copy
import torch

from transformers import AutoModelForCausalLM, AutoProcessor

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.dtype = getattr(args, "dtype") or "auto"
        self.default_model_kwargs = {"device_map": "auto"}
        self.default_gen_kwargs = {"max_new_tokens": 100, "do_sample": False, "temperature": 0.0}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)

    def load_model(self, args):
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            _attn_implementation='eager',
            **self.model_kwargs
        ).eval()

        self.processor = AutoProcessor.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True
        )

    def _parse_input(self, msg, image_counter=1):
        """Parse a single message and return user message, images, and updated image counter."""
        prompt = msg["prompt"]
        # placeholder <>, can be image
        q_chunks = re.split(r'(<image>)', prompt)
        media = copy.deepcopy(msg["media"])

        # Build the prompt with image placeholders for Phi-4
        text_content = []
        placeholder_content = []
        images = []

        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            if chunk == constants.image:
                if media:
                    images.append(media.pop(0))
                    placeholder_content.append(f"<|image_{image_counter}|>")
                    image_counter += 1
            else:
                text_content.append(chunk)

        # Combine placeholders and text
        full_content = "".join(placeholder_content) + "\n" + "".join(text_content) if placeholder_content else "".join(text_content)

        user_message = {
            "role": "user",
            "content": full_content
        }

        return user_message, images, image_counter

    def _generate_response(self, conversation_history, all_images):
        """Generate response using full conversation history and all images."""
        # Apply chat template to full conversation history
        prompt = self.processor.tokenizer.apply_chat_template(
            conversation_history,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process with all images if available, otherwise just text
        if all_images:
            inputs = self.processor(prompt, all_images, return_tensors="pt").to(self.model.device)
        else:
            inputs = self.processor(prompt, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            generate_ids = self.model.generate(
                **inputs,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                **self.gen_kwargs
            )

            # Remove input tokens
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            response = self.processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

        return response

    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        responses = []
        conversation_history = []  # Accumulate conversation history for multi-turn chat
        all_images = []  # Collect all images across turns
        image_counter = 1  # Track unique image placeholders
        
        for msg in sample["messages"]:
            # Parse current user message with current image_counter
            user_message, new_images, image_counter = self._parse_input(msg, image_counter)
            conversation_history.append(user_message)
            all_images.extend(new_images)

            if not self.args.score_target:
                response = self._generate_response(conversation_history, all_images)
                responses.append(response)
                # Add assistant response to conversation history
                conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
            else:
                pass

        ori_sample["response"] = responses
        return ori_sample


if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()