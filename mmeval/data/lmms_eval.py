# Dataset class adapted for lmms-eval
# https://github.com/EvolvingLMMs-Lab/lmms-eval

from typing import Set, List, Dict, Any
import ast

from datasets import load_dataset
from mmeval.data.dataset import Dataset
from mmeval.data.dataset_map import DATASET_MAP
from mmeval.utils import constants

class LMMSEvalDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.mapping = DATASET_MAP["default"]
        self.name = self.args.infile
        for k, v in DATASET_MAP[self.name].items():
            self.mapping[k] = v
        self.load_dataset()
    
    def load_dataset(self):
        self.data = load_dataset(self.name, split=self.args.split)
        data = []
        for item in self.data:
            data.append(self._process_sample(item))
        self.data = data
    
    def _process_sample(self, item):
        instance = {}
        if isinstance(self.mapping["modality"], list):
            instance["modality"] = [item[k] for k in self.mapping["modality"] if item[k] is not None]
        else:
            instance["modality"] = item[self.mapping["modality"]]
        if isinstance(self.mapping["options"], list):
            instance["options"] = [item[k] for k in self.mapping["options"] if item[k] is not None and item[k] != "nan"]
        else:
            instance["options"] = item[self.mapping["options"]]
        if self.mapping["pre-prompt"] is not None:
            if self.mapping["pre-prompt"] in item:
                instance["pre-prompt"] = item[self.mapping["pre-prompt"]]
            else:
                instance["pre-prompt"] = self.mapping["pre-prompt"]
        if self.mapping["post-prompt"] is not None:
            if self.mapping["post-prompt"] in item:
                instance["post-prompt"] = item[self.mapping["post-prompt"]]
            else:
                instance["post-prompt"] = self.mapping["post-prompt"]
        if self.mapping["options-prompt"] is not None:
            if self.mapping["options-prompt"] in item:
                instance["options-prompt"] = item[self.mapping["options-prompt"]]
            else:
                instance["options-prompt"] = self.mapping["options-prompt"]
        instance["questions"] = self.construct_prompt(instance, item[self.mapping["question"]])
        if self.mapping["id"] is not None:
            instance["id"] = item[self.mapping["id"]]
        return instance

    def _load_raw_data(self, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError("LMMSEvalDataset does not support _load_raw_data")

    def parse_options(self, options):
        option_letters = [chr(ord("A") + i) for i in range(len(options))]
        choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
        return choices_str


    def construct_prompt(self, doc, question):
        if doc["options"] is not None:
            parsed_options = self.parse_options(doc["options"])
            if "options-prompt" in doc and doc["options-prompt"] is not None and doc["options-prompt"] != "nan":
                question = f"{question}{doc['options-prompt']}"
        if "pre-prompt" in doc and doc["pre-prompt"] is not None and doc["pre-prompt"] != "nan":
            question = f"{doc['pre-prompt']}\n{question}"
        if "post-prompt" in doc and doc["post-prompt"] is not None and doc["post-prompt"] != "nan":
            question = f"{question}\n{doc['post-prompt']}"
        if parsed_options is not None:
            question = f"{question}\n{parsed_options}"
        return question
