import argparse
import inspect
import os
import sys
import warnings
from dataclasses import dataclass, field, fields
from typing import Dict, Optional, Sequence, get_args

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    model_series: Optional[str] = field(default=None)

    # parameters for model
    low_cpu_mem_usage: Optional[bool] = field(default=None, metadata={"help": "Tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model."})
    attn_implementation: Optional[str] = field(default=None, metadata={"help": "The attention implementation to use in the model (if relevant)."})

    # parameters for model inference
    dtype: Optional[str] = field(default=None, metadata={"help": "Override the default torch.dtype and load the model under a specific dtype."})
    min_pixels: Optional[int] = field(default=None, metadata={"help": "Image processor min_pixels override (Qwen-VL family); None keeps the runner's default."})
    max_pixels: Optional[int] = field(default=None, metadata={"help": "Image processor max_pixels override (Qwen-VL family); None keeps the runner's default."})
    device_map: Optional[str] = field(default=None, metadata={"help": "A map that specifies where each submodule should go."})
    
    # parameters that control the length of the output
    max_length: Optional[int] = field(default=None, metadata={"help": "The maximum length the generated tokens can have."})
    max_new_tokens: Optional[int] = field(default=None, metadata={"help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."})
    min_length: Optional[int] = field(default=None, metadata={"help": "The minimum length of the sequence to be generated."})
    min_new_tokens: Optional[int] = field(default=None, metadata={"help": "The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt."})
    early_stopping: Optional[bool] = field(default=None, metadata={"help": "Controls the stopping condition for beam-based methods, like beam-search."})
    max_time: Optional[float] = field(default=None, metadata={"help": "The maximum amount of time you allow the computation to run for in seconds."})
    
    # parameters that control the generation strategy used
    do_sample: Optional[bool] = field(default=None, metadata={"help": "Whether or not to use sampling ; use greedy decoding otherwise."})
    num_beams: Optional[int] = field(default=None, metadata={"help": "Number of beams for beam search. 1 means no beam search."})

    # parameters that control the cache
    use_cache: Optional[bool] = field(default=None, metadata={"help": "Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding."})
    cache_implementation: Optional[str] = field(default=None, metadata={"help": "Name of the cache class that will be instantiated in generate."})

    # parameters for manipulation of the model output logits
    temperature: Optional[float] = field(default=None, metadata={"help": "The value used to module the next token probabilities."})
    top_k: Optional[int] = field(default=None, metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k-filtering."})
    top_p: Optional[float] = field(default=None, metadata={"help": "If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation."})
    min_p: Optional[float] = field(default=None, metadata={"help": "Minimum token probability, which will be scaled by the probability of the most likely token."})
    diversity_penalty: Optional[float] = field(default=None, metadata={"help": "This value is subtracted from a beam’s score if it generates a token same as any beam from other group at a particular time."})
    repetition_penalty: Optional[float] = field(default=None, metadata={"help": "The parameter for repetition penalty. 1.0 means no penalty."})
    length_penalty: Optional[float] = field(default=None, metadata={"help": "Exponential penalty to the length that is used with beam-based generation."})

@dataclass
class DataArguments:
    dataset: Optional[str] = field(default=None,
                           metadata={"help": "name of the dataset."})
    split: Optional[str] = field(default=None,
                           metadata={"help": "split of the dataset for huggingface."})
    subset: Optional[str] = field(default=None,
                           metadata={"help": "HF mm-eval subset key inside metadata.json (when multiple subsets exist)."})
    infile: Optional[str]= field(default=None,
                           metadata={"help": "input file."})
    img_dir: Optional[str] = field(default=None,
                           metadata={"help": "image directory."})
    template: Optional[str] = field(default=None,
                           metadata={"help": "template file path or template string."})
    circular: bool = field(default=False, 
                           metadata={"help": "whether to prepare data for circular evaluation."})
    resize: Optional[int] = field(default=None,
                           metadata={"help": "resize images to this pixel value."})
    sample_num: Optional[int] = field(default=None,
                           metadata={"help": "number of samples to run (default: all)."})
    sample_order: str = field(default="head",
                           metadata={"help": "sample selection order: head | tail | random (default: head)."})
    sample_seed: int = field(default=42,
                           metadata={"help": "random seed for sample_order=random (default: 42)."})

@dataclass
class InferenceArguments:
    save_freq: int = field(default=3, metadata={"help": "save frequency for cache."})
    max_retry: int = field(default=1, metadata={"help": "maximum number of retries for entire dataset."}) 
    max_retry_sample: int = field(default=1, metadata={"help": "maximum number of retries for one inference sample."}) 
    out_dir: Optional[str] = field(default=None,
                           metadata={"help": "output directory."})
    score_target: bool = field(default=False, metadata={"help": "whether to output scores for each choice."})
    resume: bool = field(default=True, metadata={"help": "whether to resume from cache."})
    
@dataclass
class ExperimentArguments:
    gpu_per_parallel: int = field(default=1, metadata={"help": "number of gpus per task"})
    parallel_per_task: int = field(default=4, metadata={"help": "number of parallel tasks."}) 
    rank: int = field(default=-1, metadata={"help": "rank for parallel inference"})
    no_conda: bool = field(default=False, metadata={"help": "use current python env instead of conda"})


@dataclass
class ScoreRuntimeArguments:
    # Score-mode CLI convention: every dest is prefixed `score_` (scoring
    # pipeline/IO/knobs) or `judge_` (the LLM-judge client group — already
    # score-mode-only and self-descriptive; `score_judge_*` would stutter).
    # No unprefixed or infer-colliding dests.
    score_out_dir: Optional[str] = field(default=None, metadata={"help": "output directory containing result.json files"})
    score_parallel_per_task: int = field(default=4, metadata={"help": "number of sample workers inside each result.json"})


@dataclass
class ScoreArguments:
    score_result_glob: str = field(default="**/result.json", metadata={"help": "glob pattern for result files in score mode"})
    score_output_name: str = field(default="score.json", metadata={"help": "output score json file name"})
    score_progress_bar: bool = field(default=True, metadata={"help": "show progress bar while scoring samples"})
    score_resume: bool = field(default=True, metadata={"help": "resume scoring from score tmp/final files when available"})
    score_save_freq: int = field(default=20, metadata={"help": "flush frequency for score resume tmp file"})
    score_debug: bool = field(default=False, metadata={"help": "write per-stage trace into score outputs"})
    score_pipeline: str = field(default="exact-match,rule-match", metadata={"help": "ordered scoring pipeline: comma-separated atomic stage names (exact-match, rule-match, llm-match, llm-judge, vqa-accuracy, anls)"})

    score_gt_field: str = field(default="answer", metadata={"help": "field name for ground-truth in result sample"})
    score_pred_field: str = field(default="messages[-1].response", metadata={"help": "field path for prediction in result sample"})
    score_question_type: str = field(default="auto", metadata={"help": "question type: auto|mcq|yes_no|numeric|open. auto = read the sample's question_type field (recognized values only), then infer from gt shape"})
    score_numeric_rel_tol: float = field(default=0.0, metadata={"help": "relative tolerance for numeric answers (0 = exact float equality)"})
    score_numeric_abs_tol: float = field(default=0.0, metadata={"help": "absolute tolerance for numeric answers (0 = off; DynaMath uses 0.001)"})
    score_string_match: str = field(default="exact", metadata={"help": "rule-chain text comparison: exact|contains|anls (contains = OCRBench substring protocol; anls = threshold ANLS)"})
    score_anls_threshold: float = field(default=0.5, metadata={"help": "ANLS threshold (string_match=anls and the anls grader)"})

    judge_provider: Optional[str] = field(default=None, metadata={"help": "llm judge provider: local (in-process open-weight model, framework-native loading) | openai | azure_openai"})
    judge_model: Optional[str] = field(default=None, metadata={"help": "llm judge model/deployment name"})
    # For the three knobs below, resolution order is: CLI flag > env var > default
    # (env names JUDGE_MAX_RETRY / JUDGE_MAX_CONCURRENCY / JUDGE_MAX_TOKENS).
    judge_max_retry: Optional[int] = field(default=None, metadata={"help": "llm judge max attempts per call (default: env JUDGE_MAX_RETRY or 3)"})
    judge_temperature: float = field(default=0.0, metadata={"help": "llm judge generation temperature"})
    judge_concurrency: Optional[int] = field(default=None, metadata={"help": "max concurrent llm judge calls, process-wide (default: env JUDGE_MAX_CONCURRENCY or 4)"})
    judge_max_tokens: Optional[int] = field(default=None, metadata={"help": "llm judge max completion tokens (default: env JUDGE_MAX_TOKENS or 2048)"})
    judge_include_reason: bool = field(default=False, metadata={"help": "include llm judge reason in output"})


ARGUMENT_DATACLASSES = (ModelArguments, DataArguments, InferenceArguments, ExperimentArguments, ScoreArguments)
INFER_ARGUMENT_DATACLASSES = (ModelArguments, DataArguments, InferenceArguments, ExperimentArguments)
SCORE_ARGUMENT_DATACLASSES = (ScoreRuntimeArguments, ScoreArguments)


BOOL_DEFAULTS = {
    f.name: f.default
    for dc in ARGUMENT_DATACLASSES
    for f in fields(dc)
    if (
        f.type is bool
        or (
            bool in get_args(f.type)
            and type(None) in get_args(f.type)
        )
    )
}

def _add_dataclass_arguments(parser, dataclasses, suppress_defaults=False):
    for dc in dataclasses:
        for f in fields(dc):
            tp = f.type
            type_args = getattr(tp, '__args__', None)
            if type_args and type(None) in type_args:
                tp = next(a for a in type_args if a is not type(None))
            help_text = f.metadata.get("help", "")
            cli_name = f.name.replace("_", "-")
            if tp is bool:
                if f.default is True:
                    parser.add_argument(f"--no-{cli_name}", f"--no_{f.name}",
                                        dest=f.name, action="store_false",
                                        default=argparse.SUPPRESS if suppress_defaults else True,
                                        help=help_text)
                elif f.default is False:
                    parser.add_argument(f"--{cli_name}", f"--{f.name}",
                                        dest=f.name, action="store_true",
                                        default=argparse.SUPPRESS if suppress_defaults else False,
                                        help=help_text)
                else:
                    parser.add_argument(f"--{cli_name}", f"--{f.name}",
                                        dest=f.name, action="store_true",
                                        default=argparse.SUPPRESS if suppress_defaults else None,
                                        help=help_text)
                    parser.add_argument(f"--no-{cli_name}", f"--no_{f.name}",
                                        dest=f.name, action="store_false",
                                        **({"default": argparse.SUPPRESS} if suppress_defaults else {}))
            else:
                parser.add_argument(f"--{cli_name}", f"--{f.name}",
                                    dest=f.name, type=tp,
                                    default=argparse.SUPPRESS if suppress_defaults else f.default,
                                    help=help_text)


def _track_explicit_flags(args, argv):
    """Record which score-mode dests the user explicitly provided (shadow parse
    with suppressed defaults). The scorer's per-knob resolution needs this:
    explicit CLI flag > dataset_meta from result.json > built-in default."""
    shadow = argparse.ArgumentParser()
    _add_dataclass_arguments(shadow, SCORE_ARGUMENT_DATACLASSES, suppress_defaults=True)
    args._explicit_flags = frozenset(vars(shadow.parse_known_args(argv)[0]).keys())
    return args


def parse_args(mode: str = "auto"):
    mode = (mode or "auto").strip().lower()
    if mode == "auto":
        script_name = os.path.basename(sys.argv[0])
        mode = "score" if script_name == "score.py" else "infer"

    if mode == "infer":
        dataclasses = INFER_ARGUMENT_DATACLASSES
    elif mode == "score":
        dataclasses = SCORE_ARGUMENT_DATACLASSES
    else:
        raise ValueError(f"Unsupported parse mode: {mode}. Expected infer|score|auto.")

    parser = argparse.ArgumentParser()
    _add_dataclass_arguments(parser, dataclasses)
    args = parser.parse_args()
    if mode == "score":
        args = _track_explicit_flags(args, sys.argv[1:])
    return args
def parse_model_kwargs(args, default_kwargs=None):
    default_kwargs = default_kwargs or {}
    model_kwargs = {}

    for key in (
        "low_cpu_mem_usage", "attn_implementation", "device_map"
    ):
        value = getattr(args, key, None)
        if value is None:
            value = default_kwargs.get(key)
        if value is not None:
            model_kwargs[key] = value

    return model_kwargs

def parse_gen_kwargs(args, default_kwargs=None):
    default_kwargs = default_kwargs or {}
    gen_kwargs = {}

    for key in (
        "max_length", "max_new_tokens",
        "min_length", "min_new_tokens",
        "early_stopping", "max_time",
        "do_sample", "num_beams",
        "use_cache", "cache_implementation",
        "temperature", "top_k", "top_p", "min_p",
        "diversity_penalty", "repetition_penalty", "length_penalty",
    ):
        value = getattr(args, key, None)
        if value is None:
            value = default_kwargs.get(key)
        if value is not None:
            gen_kwargs[key] = value

    return gen_kwargs


# Default parameter name mapping (Transformers -> OpenAI-compatible API)
GEN_KWARGS_MAPPING = {
    "max_new_tokens": "max_completion_tokens",
}


def filter_gen_kwargs(gen_kwargs, api_method, mapping=None):
    """Filter and map gen_kwargs to match target API method signature."""
    if mapping is None:
        mapping = GEN_KWARGS_MAPPING

    mapped_all = {mapping.get(key, key): value for key, value in gen_kwargs.items()}
    try:
        sig = inspect.signature(api_method)
    except (TypeError, ValueError):
        warnings.warn("Could not inspect target API signature, all generation kwargs will be passed through.")
        return mapped_all

    # api_method accepts **kwargs, so all mapped parameters are supported
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return mapped_all

    supported = set(sig.parameters.keys())

    filtered_kwargs = {}
    unsupported_kwargs = []

    for key, value in gen_kwargs.items():
        mapped_key = mapping.get(key, key)
        if mapped_key in supported:
            filtered_kwargs[mapped_key] = value
        else:
            unsupported_kwargs.append(key)

    if unsupported_kwargs:
        warnings.warn(f"Unsupported generation parameters will be ignored: {unsupported_kwargs}")

    return filtered_kwargs

