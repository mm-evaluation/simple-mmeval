import re
import copy
import torch
import transformers

from repo.model.videollama2 import model_init, mm_infer
from repo.model.videollama2.utils import disable_torch_init

from mmeval.infer.task import Task
from mmeval.utils import spec_tokens 
from mmeval.utils.argparser import ModelArguments, DataArguments, InferenceArguments

class TaskRunner(Task):
    def __init__(self, model_arguments, data_arguments, inference_arguments):
        super().__init__(model_arguments, data_arguments, inference_arguments)
        
    
    def load_model(self, args):
        self.model, self.processor, self.tokenizer = model_init(args.model_name_or_path)

    def run_sample(self, sample:dict):

        ori_sample = copy.deepcopy(sample)
        question, modality = self.parse_input(sample)
        image_file = sample["media_path"][0]

        media_tensor = self.processor[modality](image_file)
        output_text = mm_infer(media_tensor, question, model=self.model, tokenizer=self.tokenizer, do_sample=False, modal=modality).strip()

        ori_sample["response"] = output_text
        return ori_sample

    def parse_input(self, sample:dict):
        question = sample["question"]
        # extract placeholder
        placeholders = re.findall(r'<[^>]*>', question)
        assert len(placeholders) == 1, f"VideoLLaMA2 supports one image or video, but got {len(placeholder)}"
        
        placeholder = placeholders[0]
        modality = "image" if placeholder == spec_tokens.image else "video"

        # remove the placeholder in the question
        question = question.replace(placeholder, "").strip()

        return question, modality
        

    

if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, InferenceArguments))
    model_arguments, data_arguments, inference_arguments = parser.parse_args_into_dataclasses()
    

    model_evaluator = TaskRunner(model_arguments, data_arguments, inference_arguments)

    model_evaluator.inference_dataset()

        
