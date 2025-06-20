import json
import os
from typing import List, Dict, Any, Optional

from .dataset_loader import BaseDatasetLoader

__all__ = [
    "LocalDatasetLoader"
]


class LocalDatasetLoader(BaseDatasetLoader):
    """Loader for local JSON datasets.
    
    This loader extends BaseDatasetLoader to handle local JSON files
    with support for image directories and path resolution.
    """
    
    def __init__(self, data_file: Optional[str] = None, img_dir: Optional[str] = None):
        super().__init__()
        self.data_file = data_file
        self.img_dir = img_dir

    def _load_raw_data(self, sample_mode: str = "all", sample_number: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        """Load raw data from local JSON file with sampling applied.
        
        Parameters
        ----------
        sample_mode: str
            Sampling strategy: ``all``, ``first``, ``last``, ``random``
        sample_number: Optional[int]
            Number of samples for modes other than ``all``
        **kwargs
            Additional arguments (ignored for local loader)
        
        Returns
        -------
        List[Dict[str, Any]]
            Raw data loaded from JSON file, already filtered according to sampling parameters
            
        Raises
        ------
        FileNotFoundError
            If the data file doesn't exist
        """
        if not self.data_file or not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
            
        with open(self.data_file, 'r') as f:
            all_raw_data = json.load(f)
        
        # Apply sampling filter
        if sample_mode == "all":
            return all_raw_data
        else:
            # Get sample indices based on the full dataset length
            indices = self._get_sample_indices(len(all_raw_data), sample_mode, sample_number)
            # Return only the filtered samples
            return [all_raw_data[idx] for idx in indices]

    def _get_dataset_name(self) -> str:
        """Return the dataset name for this loader."""
        if self.data_file:
            return f"local@{os.path.basename(self.data_file)}"
        return "local@unknown"
    
    def _process_sample(self, item: Dict[str, Any], split_name: str) -> Dict[str, Any]:
        """Convert a raw local dataset sample into the common schema.
        
        This preserves the original LocalJSONDataset behavior:
        - media_path field gets converted to full paths using img_dir
        - question field is used as prompt 
        - All original fields are preserved
        
        Parameters
        ----------
        item: Dict[str, Any]
            Raw sample from the JSON file
        split_name: str
            Split name (typically "local" for local datasets)
            
        Returns
        -------
        Dict[str, Any]
            Processed sample in the common schema format
            
        Output format:
            {"id": str, "media": List[str], "prompt": str, "media_path": List[str], 
             "question": str, "split": str, ...}
        """
        # Handle media_path field - convert to full paths (original behavior)
        media_path = item.get("media_path", [])
        if self.img_dir:
            full_paths = [os.path.join(self.img_dir, f) for f in media_path]
        else:
            full_paths = media_path[:]
            
        # Use question as prompt (preserving original field structure)
        prompt = item.get("question", "")
        
        # Create processed sample - preserve ALL original fields
        sample = dict(item)  # Start with all original fields
        sample.update({
            "id": item.get("id", ""),
            "media": full_paths,  # New unified field
            "prompt": prompt,     # New unified field
            "media_path": full_paths,  # Update original field with full paths
            "split": split_name
        })
        
        return sample 