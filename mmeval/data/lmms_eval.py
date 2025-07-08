# Dataset class adapted for lmms-eval
# https://github.com/EvolvingLMMs-Lab/lmms-eval

from typing import Set

from mmeval.data.dataset import Dataset
from datasets import load_dataset

class LMMSEvalDataset(Dataset):
    def __init__(self, dataset_name: str, **kwargs):
        self.name = dataset_name
        self._ids = Set[str] = set()
        self.load_dataset()
    
    def load_dataset(self):
        self.dataset = load_dataset(self.name)
    
    def _process_sample(self, item):
        if hasattr(item, "image"):
            item.rename_column("image", "modality")
        elif hasattr(item, "video"):
            item.rename_column("video", "modality")
        if not isinstance(item["modality"], list):
            item["modality"] = [item["modality"]]
        
    def __getitem__(self, index):
        return self.process_sample(self.dataset[index])

