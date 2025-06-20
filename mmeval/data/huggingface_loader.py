import os
from abc import abstractmethod
from datasets import load_dataset
from typing import List, Dict, Any, Optional

from .dataset import Dataset
from .dataset_loader import BaseDatasetLoader

__all__ = [
    "HuggingFaceDatasetLoader", 
    "MMEDatasetLoader",
    "MMVetDatasetLoader", 
    "SeedBenchDatasetLoader",
]


class HuggingFaceDatasetLoader(BaseDatasetLoader):
    """Base class for HuggingFace dataset loaders."""

    def __init__(self, dataset_dir: Optional[str] = None, hf_home: Optional[str] = None):
        super().__init__()
        self.dataset_dir = dataset_dir
        # Resolve cache directory lazily so code works anywhere.
        self.hf_home = hf_home or os.path.join(os.getcwd(), "hf_home")
        os.makedirs(self.hf_home, exist_ok=True)

    def _load_raw_data(self, sample_mode: str = "all", sample_number: Optional[int] = None, split: str = "all", **kwargs) -> List[Dict[str, Any]]:
        """Load raw data from HuggingFace hub with sampling applied.
        
        Parameters
        ----------
        sample_mode: str
            Sampling strategy: ``all``, ``first``, ``last``, ``random``
        sample_number: Optional[int]
            Number of samples for modes other than ``all``
        split: str
            Dataset split to load
        **kwargs
            Additional arguments
        
        Returns
        -------
        List[Dict[str, Any]]
            Raw data loaded from HuggingFace, already filtered according to sampling parameters
        """
        try:
            if split == "all":
                raw_dataset = load_dataset(self.dataset_dir, cache_dir=self.hf_home)
            else:
                raw_dataset = {split: load_dataset(self.dataset_dir, split=split, cache_dir=self.hf_home)}
        except Exception:
            # Fallback to loading all splits if specific split fails.
            raw_dataset = load_dataset(self.dataset_dir, cache_dir=self.hf_home)

        # Flatten all splits into a single list
        all_raw_data = []
        for split_name, split_data in raw_dataset.items():
            for item in split_data:
                item["split"] = split_name  # Add split info
                all_raw_data.append(item)
        
        # Apply sampling filter
        if sample_mode == "all":
            return all_raw_data
        else:
            # Get sample indices based on the full dataset length
            indices = self._get_sample_indices(len(all_raw_data), sample_mode, sample_number)
            # Return only the filtered samples
            return [all_raw_data[idx] for idx in indices]

    def load(
        self,
        split: str = "all",
        sample_mode: str = "all", 
        sample_number: Optional[int] = None,
    ) -> Dataset:
        """Load HuggingFace dataset with split support."""
        return super().load(sample_mode=sample_mode, sample_number=sample_number, split=split)

    def _process_sample(self, item: Dict[str, Any], split_name: str) -> Dict[str, Any]:
        """Use the original split name from the data."""
        original_split = item.pop("split", split_name)
        return self._process_hf_sample(item, original_split)

    @abstractmethod 
    def _process_hf_sample(self, item: Dict[str, Any], split_name: str) -> Dict[str, Any]:
        """Process HuggingFace sample - to be implemented by subclasses."""


# -------------------------------------------------------------------------
# HuggingFace dataset loaders  
# -------------------------------------------------------------------------

class MMEDatasetLoader(HuggingFaceDatasetLoader):
    """Loader for the MME dataset."""

    def __init__(self, dataset_dir: Optional[str] = None, hf_home: Optional[str] = None):
        # Set default dataset_dir if not provided
        if dataset_dir is None:
            dataset_dir = "lmms-lab/MME"
        super().__init__(dataset_dir, hf_home)

    def _get_dataset_name(self) -> str:
        """Return the dataset name for this loader."""
        return "MME"

    def _process_hf_sample(self, item: Dict[str, Any], split_name: str) -> Dict[str, Any]:
        """Convert a raw dataset sample into the common schema.
        
        Output format:
            {"id": str, "media": List[PIL.Image], "prompt": str, "question": str, 
             "answer": str, "category": str, "split": str}
        """
        # Ensure media is a list.
        media = item["image"] if isinstance(item["image"], list) else [item["image"]]

        # Prompt = <image> tokens + question.
        prompt = f"{'<image>' * len(media)}\n{item.get('question', '')}"

        sample = {
            "id": item["question_id"],
            "media": media,
            "prompt": prompt,
            **item,
            "split": split_name,
        }

        # Remove duplicates.
        for old_field in ("image"):
            sample.pop(old_field, None)
        return sample


class MMVetDatasetLoader(HuggingFaceDatasetLoader):
    """Loader for the MM-Vet dataset."""

    def __init__(self, dataset_dir: Optional[str] = None, hf_home: Optional[str] = None):
        # Set default dataset_dir if not provided
        if dataset_dir is None:
            dataset_dir = "lmms-lab/MMVet"
        super().__init__(dataset_dir, hf_home)

    def _get_dataset_name(self) -> str:
        """Return the dataset name for this loader."""
        return "MM-Vet"

    def _process_hf_sample(self, item: Dict[str, Any], split_name: str) -> Dict[str, Any]:
        """Convert a raw dataset sample into the common schema.
        
        Output format:
            {"id": str, "media": List[PIL.Image], "prompt": str, "question": str, 
             "answer": str, "image_source": str, "capability": list, "split": str}
        """
        media = item["image"] if isinstance(item["image"], list) else [item["image"]]
        prompt = f"{'<image>' * len(media)}\n{item.get('question', '')}"

        sample = {
            "id": item["question_id"],
            "media": media,
            "prompt": prompt,
            **item,
            "split": split_name,
        }
        for old_field in ("image"):
            sample.pop(old_field, None)
        return sample


class SeedBenchDatasetLoader(HuggingFaceDatasetLoader):
    """Loader for the SEED-Bench dataset (multiple-choice)."""

    def __init__(self, dataset_dir: Optional[str] = None, hf_home: Optional[str] = None):
        # Set default dataset_dir if not provided
        if dataset_dir is None:
            dataset_dir = "lmms-lab/SEED-Bench"
        super().__init__(dataset_dir, hf_home)

    def _get_dataset_name(self) -> str:
        """Return the dataset name for this loader."""
        return "SEED-Bench"

    def _process_hf_sample(self, item: Dict[str, Any], split_name: str) -> Dict[str, Any]:
        """Convert a raw dataset sample into the common schema.
        
        Output format:
            {"id": str, "media": List[PIL.Image], "prompt": str, "question": str, 
             "answer": str, "choice_a": str, "choice_b": str, "choice_c": str, 
             "choice_d": str, "data_type": str, "question_type_id": int, 
             "data_id": str, "split": str}
        """
        media = item["image"] if isinstance(item["image"], list) else [item["image"]]
        num_images = len(media)
        image_tokens = "<image>" * num_images

        # Build multiple-choice options.
        choices = (
            f"\nA. {item.get('choice_a', '')}"
            f"\nB. {item.get('choice_b', '')}"
            f"\nC. {item.get('choice_c', '')}"
            f"\nD. {item.get('choice_d', '')}"
        )
        prompt = f"{image_tokens}\n{item.get('question', '')}{choices}"

        sample = {
            "id": item["question_id"],
            "media": media,
            "prompt": prompt,
            **item,
            "split": split_name,
        }
        for old_field in ("image"):
            sample.pop(old_field, None)
        return sample 