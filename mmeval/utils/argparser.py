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
    infile: Optional[str]= field(default=None,
                           metadata={"help": "input file."})
    img_dir: Optional[str] = field(default=None,
                           metadata={"help": "image directory."})
    split: Optional[str] = field(default=None,
                           metadata={"help": "split of the dataset."})
@dataclass
class InferenceArguments:
    save_freq: int = field(default=3, metadata={"help": "save frequency for cache."})
    max_retry: int = field(default=3, metadata={"help": "maximum number of retries for entire dataset."}) 
    max_retry_sample: int = field(default=3, metadata={"help": "maximum number of retries for one inference sample."}) 
    out_dir: str = field(default=None,
                           metadata={"help": "output directory."})
    
    
@dataclass
class ExperimentArguments:
    gpu_per_parallel: int = field(default=1, metadata={"help": "number of gpus per task"})
    parallel_per_task: int = field(default=4, metadata={"help": "number of parallel tasks."}) 
    rank: int = field(default=-1, metadata={"help": "rank for parallel inference"}) 


def parse_args():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, InferenceArguments, ExperimentArguments))
    args = parser.parse_args()
    return args