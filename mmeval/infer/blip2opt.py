import re
import copy
import torch
import numpy as np
import transformers
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoTokenizer
from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args
from PIL import Image

class TaskRunner(Task):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self, args):
        self.model = Blip2ForConditionalGeneration.from_pretrained(f"Salesforce/{args.model_name_or_path}", torch_dtype=torch.float16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(f"Salesforce/{args.model_name_or_path}")
        self.processor = Blip2Processor.from_pretrained(f"Salesforce/{args.model_name_or_path}")
    
    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)

        question, modality = self.parse_input(sample)
        image_file = sample["media"][0]
        raw_image = Image.open(image_file).convert('RGB')
        inputs = self.processor(raw_image, question, return_tensors="pt").to("cuda")

        if not self.args.score_target:
            out = self.model.generate(**inputs)
            response = self.processor.decode(out[0], skip_special_tokens=True).strip()
            ori_sample["response"] = response
        else:
            ori_sample.update(self._score_choices())

        return ori_sample

    def _score_choices(self):
        raise NotImplementedError("Score choices is not implemented")


    def parse_input(self, sample:dict):
        question = sample["prompt"]
        # extract placeholder
        placeholders = re.findall(r'<[^>]*>', question)
        #assert len(placeholders) == 1
        
        placeholder = placeholders[0]
        modality = "image" if placeholder == constants.image else "video"

        # remove the placeholder in the question
        question = question.replace(placeholder, "").strip()

        # Instruction Format
        question = f"Question: {question} Answer:"
        return question, modality

    

if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
