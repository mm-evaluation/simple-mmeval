from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, TYPE_CHECKING
import transformers


if TYPE_CHECKING:
    import transformers

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    
    # inference hyper-parameters
    max_new_tokens: int = field(default=512, metadata={"help": "maximum number of new tokens to generate."})
    temperature: float = field(default=0.0, metadata={"help": "temperature for sampling."})
    top_p: float = field(default=None, metadata={"help": "top-p for sampling."})
    top_k: int = field(default=None, metadata={"help": "top-k for sampling."})
    repetition_penalty: float = field(default=None, metadata={"help": "repetition penalty for sampling."})
    

@dataclass
class DataArguments:
    dataset: str = field(default=None,
                           metadata={"help": "name of the dataset."})
    dataset_dir: Optional[str] = field(default=None,
                           metadata={"help": "HF dataset identifier or path to dataset directory."})
    hf_home: Optional[str] = field(default=None,
                           metadata={"help": "HF cache directory (defaults to work_dir/hf_home if not specified)."})
    split: Optional[str] = field(default="all",
                           metadata={"help": "Dataset split to load ('all' for all splits, or specific split name)."})
    sample_mode: Optional[str] = field(default="all",
                           metadata={"help": "Sampling mode: 'all', 'first', 'last', or 'random'."})
    sample_number: Optional[int] = field(default=None,
                           metadata={"help": "Number of samples to take (required if sample_mode != 'all')."})
    infile: Optional[str]= field(default=None,
                           metadata={"help": "input file."})
    img_dir: Optional[str] = field(default=None,
                           metadata={"help": "image directory."})

@dataclass
class InferenceArguments:
    save_freq: int = field(default=3, metadata={"help": "save frequency for cache."})
    max_retry: int = field(default=3, metadata={"help": "maximum number of retries for entire dataset."}) 
    max_retry_sample: int = field(default=3, metadata={"help": "maximum number of retries for one inference sample."}) 
    out_dir: str = field(default=None,
                           metadata={"help": "output directory."})
    
    
   