import os
import json
import random
from datasets import load_dataset
from typing import List, Dict, Any, Optional


class SeedBenchDatasetLoader:
    """
    SEED-Bench dataset loader with sampling functionality.
    
    Supports multiple sampling modes: all, first, last, random.
    Ensures media field is always a list of PIL Images.
    """
    
    def __init__(self, dataset_dir: str = "lmms-lab/SEED-Bench", hf_home: str = "/MLLM_Eval/hf_home"):
        """
        Initialize SEED-Bench dataset loader.
        
        Args:
            dataset_dir: HF dataset identifier
            hf_home: HF cache directory
        """
        self.dataset_dir = dataset_dir
        self.hf_home = hf_home
        # Ensure cache directory exists
        os.makedirs(self.hf_home, exist_ok=True)
    
    def _validate_sampling_params(self, sample_mode: str, sample_number: Optional[int]) -> None:
        """Validate sampling parameters."""
        if sample_mode != "all" and sample_number is None:
            raise ValueError(f"sample_number must be specified when sample_mode is '{sample_mode}'")
        
        if sample_mode not in ["all", "first", "last", "random"]:
            raise ValueError(f"sample_mode must be one of: all, first, last, random")
    
    def _get_sample_indices(self, data_length: int, sample_mode: str, sample_number: Optional[int]) -> List[int]:
        """Get indices for sampling based on mode and count."""
        if sample_mode == "all":
            return list(range(data_length))
        elif sample_mode == "first":
            return list(range(min(sample_number, data_length)))
        elif sample_mode == "last":
            start_idx = max(0, data_length - sample_number)
            return list(range(start_idx, data_length))
        elif sample_mode == "random":
            selected_indices = random.sample(range(data_length), min(sample_number, data_length))
            return sorted(selected_indices)  # Sort for consistent processing order
    
    def _process_sample(self, item: Dict[str, Any], split_name: str) -> Dict[str, Any]:
        """Process a single sample with field mappings."""
        # Handle media field - ensure it's always a list
        image_data = item["image"]
        media = image_data if isinstance(image_data, list) else [image_data]
        
        # Apply field mappings
        sample = {
            "id": item["question_id"],  # Rename question_id -> id
            "media": media,             # Ensure media is always a list
            **item,                     # Copy all original fields
            "split": split_name         # Add split info
        }
        
        # Remove renamed fields to avoid duplication
        for old_field in ["question_id", "image"]:
            sample.pop(old_field, None)
        
        return sample
    
    def load(self, split: str = "all", sample_mode: str = "all", sample_number: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load SEED-Bench dataset with sampling options.
        
        Args:
            split: Dataset split ("all" for all splits, or specific split name)
            sample_mode: Sampling mode ("all", "first", "last", "random")
            sample_number: Number of samples to take (required if sample_mode != "all")
            
        Returns:
            List of standardized samples
            
        Output format:
            [{"id": str, "media": List[PIL.Image], "question": str, "answer": str, "choice_a": str, "choice_b": str, "choice_c": str, "choice_d": str, "data_type": str, "question_type_id": int, "data_id": str, "split": str}, ...]
        """
        # Validate sampling parameters
        self._validate_sampling_params(sample_mode, sample_number)
        
        try:
            # Load dataset
            if split == "all":
                dataset = load_dataset(self.dataset_dir, cache_dir=self.hf_home)
                print(f"Loaded all splits: {list(dataset.keys())}")
            else:
                dataset = {split: load_dataset(self.dataset_dir, split=split, cache_dir=self.hf_home)}
                print(f"Loaded split: {split}")
        except Exception as e:
            print(f"Failed to load split '{split}': {e}")
            # Fallback to all splits
            dataset = load_dataset(self.dataset_dir, cache_dir=self.hf_home)
            print(f"Fallback - loaded all splits: {list(dataset.keys())}")
        
        samples = []
        
        # Process each split
        for split_name, split_data in dataset.items():
            print(f"Processing {split_name}: {len(split_data)} samples")
            
            # Get sample indices
            selected_indices = self._get_sample_indices(len(split_data), sample_mode, sample_number)
            print(f"Sampling: {sample_mode} mode, selected {len(selected_indices)} samples")
            
            # Process only selected samples
            for idx in selected_indices:
                sample = self._process_sample(split_data[idx], split_name)
                samples.append(sample)
        
        print(f"✓ Processed {len(samples)} SEED-Bench samples")
        return samples


# Backward compatibility function
def load_seed_bench_dataset(dataset_dir: str = "lmms-lab/SEED-Bench", split: str = "all", 
                           hf_home: str = "/MLLM_Eval/hf_home",
                           sample_mode: str = "all", sample_number: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load SEED-Bench dataset (backward compatibility wrapper).
    """
    loader = SeedBenchDatasetLoader(dataset_dir, hf_home)
    return loader.load(split, sample_mode, sample_number)