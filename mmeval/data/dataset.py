from typing import List, Dict, Any

__all__ = ["Dataset"]


class Dataset:
    """A unified dataset class for storing and accessing processed samples.
    
    This class provides:
    1. Storage for processed dataset samples
    2. Standard iteration helpers (__iter__, __getitem__, __len__)
    3. Dataset metadata (name)
    4. Uniform interface across all dataset types
    """
    
    def __init__(self, dataset_name: str):
        """Initialize the dataset.
        
        Parameters
        ----------
        dataset_name: str
            Name/identifier for this dataset
        """
        self.dataset_name = dataset_name
        self.data: List[Dict[str, Any]] = []
    
    def add_sample(self, sample: Dict[str, Any]) -> None:
        """Add a processed sample to the dataset."""
        self.data.append(sample)
    
    def add_samples(self, samples: List[Dict[str, Any]]) -> None:
        """Add multiple processed samples to the dataset."""
        self.data.extend(samples)
    
    def clear(self) -> None:
        """Clear all samples from the dataset."""
        self.data.clear()
    
    # ----------------------------- Properties ---------------------------
    @property
    def name(self) -> str:
        """Return the dataset name."""
        return self.dataset_name
    
    # ----------------------------- Iteration helpers -------------------
    def __iter__(self):
        """Iterate over dataset samples."""
        return iter(self.data)
    
    def __next__(self):
        """Get next sample."""
        return next(self.data) 

    def __getitem__(self, index):
        """Get sample by index."""
        return self.data[index]

    def __len__(self):
        """Get number of samples in dataset."""
        return len(self.data)
    
    # ----------------------------- String representation ---------------
    def __repr__(self):
        return f"Dataset(name='{self.dataset_name}', samples={len(self.data)})"
    
    def __str__(self):
        return f"{self.dataset_name} dataset with {len(self.data)} samples" 