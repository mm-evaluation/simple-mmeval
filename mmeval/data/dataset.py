from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set, Tuple, Union

from PIL import Image

__all__ = ["Dataset"]


class Dataset(ABC):
    """A unified dataset class for loading, processing, and storing dataset samples.
    
    This class provides:
    1. Automatic data loading during initialization
    2. Storage for processed dataset samples
    3. Standard iteration helpers (__iter__, __getitem__, __len__)
    4. ID uniqueness enforcement
    5. Image resizing utilities
    6. Circular data preparation placeholder
    """
    
    def __init__(self, dataset_name: str, **kwargs):
        """Initialize the dataset and automatically load data.
        
        Parameters
        ----------
        dataset_name : str
            Name/identifier for this dataset
        **kwargs
            Additional arguments for data loading
        """
        self.name = dataset_name
        self.data: List[Dict[str, Any]] = []
        self._ids: Set[str] = set()
        
        # Automatically load data during initialization
        self._load_and_process_data(**kwargs)

    def _ensure_unique_id(self, original_id: str) -> str:
        """Ensure ID uniqueness by adding suffix if needed.
        
        Parameters
        ----------
        original_id : str
            The original ID that might be duplicate
            
        Returns
        -------
        str
            A unique ID, with suffix added if necessary
        """
        if original_id not in self._ids:
            self._ids.add(original_id)
            return original_id
        
        # Find the next available suffix for duplicates
        counter = 1
        while True:
            unique_id = f"{original_id}_{counter:02d}"
            if unique_id not in self._ids:
                self._ids.add(unique_id)
                return unique_id
            counter += 1

    def _load_and_process_data(self, **kwargs) -> None:
        """Load and process data into the dataset.

        Parameters
        ----------
        **kwargs
            Additional arguments specific to the dataset type
        """
        # Load and process raw data
        raw_data = self._load_raw_data(**kwargs)
        
        for item in raw_data:
            processed = self._process_sample(item)
            # Ensure ID uniqueness
            if 'id' in processed:
                processed['id'] = self._ensure_unique_id(processed['id'])
            self.data.append(processed)

    def resize_image(
        self, 
        image: Image.Image, 
        size: Union[int, Tuple[int, int]], 
        padding: bool = True,
        fill_color: str = "black"
    ) -> Image.Image:
        """Resize a PIL Image object with optional padding.

        Parameters
        ----------
        image : Image.Image
            The PIL Image object to resize
        size : Union[int, Tuple[int, int]]
            Target size. If int, resize to (size, size). If tuple, resize to (width, height)
        padding : bool, default=True
            If True, maintain aspect ratio and pad to target size.
            If False, directly resize to target size.
        fill_color : str, default="black"
            Fill color for padding. Common colors: "black", "white", "red", "green", "blue", "gray"

        Returns
        -------
        Image.Image
            The resized PIL Image object
        """
        if isinstance(size, int):
            target_size = (size, size)
        else:
            target_size = size

        if not padding:
            # Direct resize to target size
            return image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Resize with padding to maintain aspect ratio
        original_width, original_height = image.size
        target_width, target_height = target_size
        
        # Calculate scale to fit within target size
        scale = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create new image with target size and fill color
        padded_image = Image.new(image.mode, target_size, fill_color)
        
        # Calculate position to center the resized image
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        # Paste the resized image onto the padded image
        padded_image.paste(resized_image, (x_offset, y_offset))
        
        return padded_image

    def prepare_circular_data(self, **kwargs) -> Any:
        """Prepare dataset for circular evaluation by generating circular variants.
        
        Parameters
        ----------
        **kwargs
            Additional arguments for circular data preparation configuration
            
        Returns
        -------
        Any
            Results from circular data preparation (implementation dependent)
        """
        # Placeholder for circular data preparation implementation
        raise NotImplementedError("Circular data preparation not yet implemented")

    # Abstract methods that subclasses must implement
    @abstractmethod
    def _load_raw_data(self, **kwargs) -> List[Dict[str, Any]]:
        """Load raw data from the data source.
        
        Parameters
        ----------
        **kwargs
            Additional arguments specific to the dataset type
            
        Returns
        -------
        List[Dict[str, Any]]
            Raw data items
        """

    @abstractmethod
    def _process_sample(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a raw dataset sample into the common schema.
        
        Parameters
        ----------
        item : Dict[str, Any]
            Raw data item
            
        Returns
        -------
        Dict[str, Any]
            Processed data item following the common schema
        """

    # Iteration helpers
    def __iter__(self):
        """Iterate over dataset samples."""
        return iter(self.data)

    def __getitem__(self, index):
        """Get sample by index."""
        return self.data[index]

    def __len__(self):
        """Get number of samples in dataset."""
        return len(self.data)
    
    def __repr__(self):
        return f"Dataset(name='{self.name}', samples={len(self.data)})"
    
    def __str__(self):
        return f"{self.name} dataset with {len(self.data)} samples" 