import random
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set

from .dataset import Dataset

__all__ = [
    "BaseDatasetLoader",
]


class BaseDatasetLoader(ABC):
    """Abstract base class for all dataset loaders.

    This base class provides:
    1. Common sampling logic (``all``, ``first``, ``last``, ``random``)
    2. Dataset creation and management
    3. Unified interface across all dataset types
    4. ID uniqueness enforcement

    Sub-classes must implement data loading and processing methods.
    """

    def __init__(self):
        # Dataset will be created during load
        self.dataset: Optional[Dataset] = None
        # Track used IDs to ensure uniqueness
        self._used_ids: Set[str] = set()

    # ---------------------------------------------------------------------
    # Static helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _validate_sampling_params(sample_mode: str, sample_number: Optional[int]):
        if sample_mode != "all" and sample_number is None:
            raise ValueError("sample_number must be specified when sample_mode is not 'all'")
        if sample_mode not in {"all", "first", "last", "random"}:
            raise ValueError("sample_mode must be one of: all, first, last, random")

    @staticmethod
    def _get_sample_indices(data_length: int, sample_mode: str, sample_number: Optional[int]) -> List[int]:
        if sample_mode == "all":
            return list(range(data_length))
        if sample_mode == "first":
            return list(range(min(sample_number, data_length)))
        if sample_mode == "last":
            start_idx = max(0, data_length - sample_number)
            return list(range(start_idx, data_length))
        # random
        return sorted(random.sample(range(data_length), min(sample_number, data_length)))

    def _ensure_unique_id(self, original_id: str) -> str:
        """Ensure ID uniqueness by adding suffix if needed.
        
        The first occurrence keeps the original ID unchanged.
        Subsequent duplicates get suffixes: _01, _02, etc.
        
        Parameters
        ----------
        original_id: str
            The original ID that might be duplicate
            
        Returns
        -------
        str
            A unique ID, with suffix added if necessary
        """
        if original_id not in self._used_ids:
            self._used_ids.add(original_id)
            return original_id
        
        # This is a duplicate - find the next available suffix
        counter = 1
        while True:
            unique_id = f"{original_id}_{counter:02d}"
            if unique_id not in self._used_ids:
                self._used_ids.add(unique_id)
                return unique_id
            counter += 1

    # ---------------------------------------------------------------------
    # Sub-class contract
    # ---------------------------------------------------------------------
    @abstractmethod
    def _load_raw_data(self, sample_mode: str = "all", sample_number: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        """Load raw data from the data source with optional sampling applied.
        
        Parameters
        ----------
        sample_mode: str
            Sampling strategy: ``all``, ``first``, ``last``, ``random``
        sample_number: Optional[int]
            Number of samples for modes other than ``all``
        **kwargs
            Additional arguments specific to the loader type
            
        Returns
        -------
        List[Dict[str, Any]]
            Raw data, already filtered according to sampling parameters
        """

    @abstractmethod
    def _process_sample(self, item: Dict[str, Any], split_name: str) -> Dict[str, Any]:
        """Convert a raw dataset sample into the common schema."""

    @abstractmethod
    def _get_dataset_name(self) -> str:
        """Return the dataset name for this loader."""

    # ---------------------------------------------------------------------
    # API surface
    # ---------------------------------------------------------------------
    def load(
        self,
        sample_mode: str = "all",
        sample_number: Optional[int] = None,
        **kwargs
    ) -> Dataset:
        """Load the dataset and apply sampling.

        Parameters
        ----------
        sample_mode: str
            Sampling strategy: ``all``, ``first``, ``last``, ``random``.
        sample_number: Optional[int]
            Number of samples for modes other than ``all``.
        **kwargs
            Additional arguments specific to the loader type.
            
        Returns
        -------
        Dataset
            The loaded and processed dataset
        """
        # Validate args first
        self._validate_sampling_params(sample_mode, sample_number)

        # Create dataset instance
        dataset_name = self._get_dataset_name()
        self.dataset = Dataset(dataset_name)

        # Reset used IDs for this load
        self._used_ids.clear()

        # Load raw data with sampling applied
        raw_data = self._load_raw_data(sample_mode=sample_mode, sample_number=sample_number, **kwargs)

        # Process all samples (filtering already applied in _load_raw_data)
        split_name = kwargs.get('split', 'default')
        for item in raw_data:
            processed = self._process_sample(item, split_name)
            # Ensure ID uniqueness
            if 'id' in processed:
                processed['id'] = self._ensure_unique_id(processed['id'])
            self.dataset.add_sample(processed)

        return self.dataset
