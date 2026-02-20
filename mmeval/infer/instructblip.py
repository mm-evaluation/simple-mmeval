
import copy
import logging
import torch

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

logger = logging.getLogger(__name__)

# Q-Former max position embeddings limit
QFORMER_MAX_LENGTH = 512

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.default_gen_kwargs = {
            "do_sample": False,
            "num_beams": 5,
            "max_new_tokens": 256,
            "min_length": 1,
            "top_p": 0.9,
            "repetition_penalty": 1.5,
            "length_penalty": 1.0,
            "temperature": 1.0
        }
        self.model_kwargs = parse_model_kwargs(args)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)

    def load_model(self, args):
        self.model = InstructBlipForConditionalGeneration.from_pretrained(args.model_name_or_path, **self.model_kwargs)
        self.processor = InstructBlipProcessor.from_pretrained(args.model_name_or_path)
        self.model.to(self.device)
        
    def _parse_input(self, message: dict):
        prompt = message["prompt"]
        prompt = prompt.replace("<image>", "")
        media_list = message.get('media', [])
        image = media_list[0] if media_list else None

        return prompt, image

    def _generate_response(self, inputs):
        outputs = self.model.generate(
                **inputs,
                **self.gen_kwargs
        )

        if not self.model.language_model.config.is_encoder_decoder:
            outputs = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
            ]
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

        return generated_text
    
    def run_sample(self, sample: dict):
        message = sample["messages"][0]
        ori_sample = copy.deepcopy(sample)
        prompt, image = self._parse_input(message)
        
        if image is None:
            raise ValueError("InstructBLIP requires an image input, but no image was provided in the sample")
        
        # Pre-truncate text if needed, accounting for special tokens that processor will add
        num_special = self.processor.qformer_tokenizer.num_special_tokens_to_add()
        max_text_tokens = QFORMER_MAX_LENGTH - num_special
        
        # Encode without special tokens to get pure text token count
        token_ids = self.processor.qformer_tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(token_ids) > max_text_tokens:
            logger.warning(
                f"Input text has {len(token_ids)} tokens, exceeds Q-Former limit of {max_text_tokens} "
                f"(reserving {num_special} for special tokens). Truncating."
            )
            truncated_ids = token_ids[:max_text_tokens]
            prompt = self.processor.qformer_tokenizer.decode(truncated_ids, skip_special_tokens=True)
        
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        if not self.args.score_target:
            response = self._generate_response(inputs)
            ori_sample["messages"].append({"role": "assistant", "response": response})
        else:
            pass

        return ori_sample

    
if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
