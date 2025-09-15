
import re
import copy
import torch
import numpy as np
from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_gen_kwargs
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens

# Import InternLM-XComposer model, tokenizer, and config

# Dynamically import modules from a folder with a hyphen in its name
import importlib.util
import sys
import os
from PIL import Image

BASE_MODEL_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    '..', 'models', 'internlm', 'internlm-xcomposer-7b'
))

# Create a package alias for the model folder so relative imports work
import types
PKG_ALIAS = 'internlm_xcomposer_7b_pkg'
if PKG_ALIAS not in sys.modules:
    pkg = types.ModuleType(PKG_ALIAS)
    pkg.__path__ = [BASE_MODEL_DIR]
    sys.modules[PKG_ALIAS] = pkg

def dynamic_import(module_filename, attr_name):
    module_name = os.path.splitext(module_filename)[0]
    fqmn = f"{PKG_ALIAS}.{module_name}"
    module_path = os.path.join(BASE_MODEL_DIR, module_filename)
    spec = importlib.util.spec_from_file_location(fqmn, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[fqmn] = module
    spec.loader.exec_module(module)
    return getattr(module, attr_name)

InternLMXComposerForCausalLM = dynamic_import('modeling_InternLM_XComposer.py', 'InternLMXComposerForCausalLM')
InternLMXComposerTokenizer = dynamic_import('tokenization_InternLM_XComposer.py', 'InternLMXComposerTokenizer')
InternLMXComposerConfig = dynamic_import('configuration_InternLM_XComposer.py', 'InternLMXComposerConfig')

def process_vision_info(messages):
    # Extract image paths from messages for InternLM-XComposer
    image_inputs = []
    video_inputs = []
    for msg in messages:
        for content in msg["content"]:
            if content["type"] == "image":
                image_inputs.append(content["image"])
            elif content["type"] == "video":
                video_inputs.append(content["video"])
    return image_inputs, video_inputs


class TaskRunner(Task):
    def __init__(self, args):
        # Initialize fields needed by load_model BEFORE calling parent __init__
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(args)

    def load_model(self, args):
        # Load config, tokenizer, and model
        self.config = InternLMXComposerConfig.from_pretrained(args.model_name_or_path)
        # Respect current hardware: fall back to CPU if CUDA isn't available/usable
        self.config.device = 'cuda' if (self.device != torch.device('cpu')) else 'cpu'
        self.tokenizer = InternLMXComposerTokenizer.from_pretrained(args.model_name_or_path)
        self.model = InternLMXComposerForCausalLM.from_pretrained(args.model_name_or_path, config=self.config).to(self.device)
        # Attach tokenizer to model for generate/chat paths inside the model class
        try:
            self.model.tokenizer = self.tokenizer
        except Exception:
            pass
        # Use the model's own vision preprocessor to ensure consistency and avoid torchvision
        self.vis_processor = self.model.vis_processor
        # Generation args from CLI (fallback to a reasonable default)
        self.gen_kwargs = parse_gen_kwargs(args, default_kwargs={"max_new_tokens": 128})

    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        messages = self.parse_input(sample)
        text = self._build_prompt(messages)
        image_inputs, _ = process_vision_info(messages)
        image_tensor = self._process_image(image_inputs[0]) if image_inputs else None

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(text, image_tensor)
        else:
            ori_sample.update(self._score_choices(text, image_tensor, sample))
        return ori_sample

    def _build_prompt(self, messages):
        # Build prompt string for InternLM-XComposer
        prompt = ""
        for msg in messages:
            for content in msg["content"]:
                if content["type"] == "text":
                    prompt += content["text"]
                elif content["type"] == "image":
                    prompt += "<ImageHere>"
        return prompt

    def _process_image(self, img_or_path):
        # Accept either a PIL Image or a string path
        if isinstance(img_or_path, Image.Image):
            image = img_or_path if img_or_path.mode == "RGB" else img_or_path.convert("RGB")
        else:
            image = Image.open(img_or_path).convert("RGB")
        # Debug: ensure vis_processor is callable
        if self.vis_processor is None:
            raise RuntimeError("vis_processor is None")
        tensor = self.vis_processor(image)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    def _generate_response(self, text, image_tensor):
        # Generate response using InternLM-XComposer
        with torch.no_grad():
            try:
                output = self.model.generate(text, image=image_tensor, **self.gen_kwargs)
            except Exception as e:
                raise RuntimeError(f"model.generate failed: {e}")
        return output.strip()

    def _score_choices(self, text, image_tensor, sample):
        contents = sample.get("choices")
        full_prompts = [text + content for content in contents]
        # Tokenize prompts
        full_encoded = [self.tokenizer(i, return_tensors="pt").to(self.device) for i in full_prompts]
        prompt_encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
        target_toks = target_tokens(self.tokenizer, contents)
        scorer = IncrementalLMScorer(self.model, self.device, tokenizer=self.tokenizer)
        scores = scorer.conditional_score(target_toks, full_encoded, prompt_encoded)
        return {
            "score": scores,
            "response": contents[np.argmax(scores)]
        }

    def parse_input(self, sample:dict):
        question = sample["prompt"]
        q_chunks = re.split(r'(<(?:image|video)>)', question)
        images = copy.deepcopy(sample['media'])
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
                assert chunk == constants.image, f"Unsupported placeholder {chunk}"
                media_file = images.pop(0)
                messages[0]["content"].append({"type": "image", "image": media_file})
            else:
                messages[0]["content"].append({"type": "text", "text": chunk})
        return messages

if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
