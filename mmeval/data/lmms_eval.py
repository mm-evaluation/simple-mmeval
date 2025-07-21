# Dataset class adapted for lmms-eval
# https://github.com/EvolvingLMMs-Lab/lmms-eval

from typing import Set, List, Dict, Any
import ast

from datasets import load_dataset
from mmeval.data.dataset import Dataset
from mmeval.utils import constants

class LMMSEvalDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.name = self.args.infile
        self.load_dataset()
    
    def load_dataset(self):
        self.data = load_dataset(self.name, split=self.args.split).select(range(2))
        if "id" not in self.data.column_names:
            self.data = self.data.add_column(name="id", column=list(range(len(self.data))))
        data = []
        for item in self.data:
            data.append(self._process_sample(item))
        self.data = data
    
    def _process_sample(self, item):
        instance = {}
        instance["id"] = item["id"]
        if "hint" in item:
            instance["pre-prompt"] = item["hint"]
        else:
            instance["pre-prompt"] = None
        instance["choices"] = self.find_options(item)
        instance["question"] = item["question"]
        instance["media"] = self.find_modality(item)
        instance["question"] = self.construct_prompt(instance)
        return instance

    def _load_raw_data(self, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError("LMMSEvalDataset does not support _load_raw_data")

    def parse_options(self, options):
        option_letters = [chr(ord("A") + i) for i in range(len(options))]
        choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
        return choices_str

    def construct_prompt(self, item):
        if item["pre-prompt"] is not None:
            question = f"{item['pre-prompt']}\n{item['question']}"
        else:
            question = item["question"]
        if item["choices"] is not None:
            parsed_options = self.parse_options(item["choices"])
            post_prompt = "\nAnswer the question using a single word or phrase."
            question = f"{question}\n{parsed_options}\n{post_prompt}"
        else:
            question = question
        return question
    
    def find_options(self, item):
        '''Find if the options are in the dataset'''
        if "options" in item and isinstance(item["options"], list):
            return item["options"]
        else:
            options = []
            for i in range(26):  # 26 letters in the alphabet
                if chr(ord("A") + i) in item and item[chr(ord("A") + i)] is not None and item[chr(ord("A") + i)] != "nan":
                    options.append(item[chr(ord("A") + i)])
            return options
    
    def find_modality(self, item):
        if "image" in item:
            return item["image"]
        elif "images" in item:
            return item["images"]
        elif "video" in item:
            return item["video"]
        elif "videos" in item:
            return item["videos"]
        else:
            return None

