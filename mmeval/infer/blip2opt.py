import re
import copy
import torch
import numpy as np
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoTokenizer
from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens
from PIL import Image

class TaskRunner(Task):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self, args):
        self.model = Blip2ForConditionalGeneration.from_pretrained(f"Salesforce/{args.model_name_or_path}", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(f"Salesforce/{args.model_name_or_path}")
        self.processor = Blip2Processor.from_pretrained(f"Salesforce/{args.model_name_or_path}")
    
    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)

        question, modality = self.parse_input(sample)
        
        # Instruction Format
        question = f"Question: {question} Answer:"
        
        image_file = sample["media"][0]
        raw_image = Image.open(image_file).convert('RGB')

        if not self.args.score_target:
            inputs = self.processor(raw_image, question, return_tensors="pt").to("cuda")
            out = self.model.generate(**inputs)
            response = self.processor.decode(out[0], skip_special_tokens=True).strip()
            response = response.replace(question, "").strip()
            ori_sample["response"] = response
        else:
            ori_sample.update(self._score_choices(question, raw_image, sample))

        return ori_sample

    def _score_choices(self, question, image_input, sample):
        contents = sample.get("choices")
        full = [question + content for content in contents]

        full_encoded = [self.processor(text=i, images=image_input, return_tensors="pt").to(self.device) for i in full]
        prompt_encoded = self.processor(text=question, images=image_input,return_tensors="pt").to(self.device)
        target_toks = target_tokens(self.tokenizer, contents)

        scorer = IncrementalLMScorer(self.model, self.device, tokenizer=self.tokenizer)
        scores = scorer.conditional_score(target_toks, full_encoded, prompt_encoded)
        
        return {
            "score": scores,
            "response": contents[np.argmax(scores)]
        }


    def parse_input(self, sample:dict):
        question = sample.get("question") or sample.get("prompt")
        placeholders = re.findall(r'<[^>]*>', question)
        assert len(placeholders) == 1
        
        placeholder = placeholders[0]
        if placeholder == constants.image:
            modality = "image"
        elif placeholder == constants.video:
            raise NotImplementedError("Blip2Opt does not support video input")
        else:
            raise ValueError(f"Unsupported placeholder: {placeholder}")

        question = question.replace(placeholder, "").strip()

        return question, modality

    

if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
