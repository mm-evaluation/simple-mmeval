from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, TYPE_CHECKING
import transformers


if TYPE_CHECKING:
    import transformers

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)

    # model hyper-parameters
    dtype: str = field(default=None, metadata={"help": "precision for model."})
    low_cpu_mem_usage: bool = field(default=None, metadata={"help": "whether to use low cpu memory usage."})
    use_flash_attn: bool = field(default=None, metadata={"help": "whether to use flash attention."})
    use_flash_attention_2: bool = field(default=None, metadata={"help": "whether to use flash attention 2."})
    
    # inference hyper-parameters
    max_new_tokens: int = field(default=None, metadata={"help": "maximum number of new tokens to generate."})
    max_length: int = field(default=None, metadata={"help": " maximum length the generated tokens can have."})
    min_length: int = field(default=None, metadata={"help": "minimum length of the sequence to be generated."})
    do_sample: bool = field(default=None, metadata={"help": "whether or not to use sampling."})
    num_beams: int = field(default=None, metadata={"help": "number of beams for beam search."})
    temperature: float = field(default=None, metadata={"help": "temperature for sampling."})
    top_k: int = field(default=None, metadata={"help": "top-k for sampling."})
    top_p: float = field(default=None, metadata={"help": "top-p for sampling."})
    repetition_penalty: float = field(default=None, metadata={"help": "repetition penalty for sampling."})
    length_penalty: float = field(default=None, metadata={"help": "length penalty for sampling."})

    # cache hyper-parameters
    use_cache: bool = field(default=None, metadata={"help": "whether to use cache to speed up decoding."})

@dataclass
class DataArguments:
    dataset: str = field(default=None,
                           metadata={"help": "name of the dataset."})
    infile: Optional[str]= field(default=None,
                           metadata={"help": "input file."})
    img_dir: Optional[str] = field(default=None,
                           metadata={"help": "image directory."})
@dataclass
class InferenceArguments:
    save_freq: int = field(default=3, metadata={"help": "save frequency for cache."})
    max_retry: int = field(default=1, metadata={"help": "maximum number of retries for entire dataset."}) 
    max_retry_sample: int = field(default=1, metadata={"help": "maximum number of retries for one inference sample."}) 
    out_dir: str = field(default=None,
                           metadata={"help": "output directory."})
    score_target: bool = field(default=False, metadata={"help": "whether to output scores for each choice."})
    resume: bool = field(default=True, metadata={"help": "whether to resume from cache."})
    
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