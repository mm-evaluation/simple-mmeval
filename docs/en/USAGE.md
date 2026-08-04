# Usage Guide

This document covers how to run evaluations with Simple-MMEval, including quick start examples, the full command-line reference, prompt template configuration, caching and resume behavior, and output format. For general overview, see the main [README](../../README.md).

---

- [Quick Start](#quick-start)
- [Command-Line Reference](#command-line-reference)
- [Prompt Template System](#prompt-template-system)
- [Resume and Caching](#resume-and-caching)
- [Output Format](#output-format)

---

## Quick Start

All examples assume you are in the project root with `PYTHONPATH` set:

```bash
export PYTHONPATH=./:$PYTHONPATH
```

### Example 1: Local JSON dataset

Evaluate a model on a local JSON file containing image/video samples:

```bash
python mmeval/run.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset local@json \
    --infile tests/samples/multi_image_video_interleave.json \
    --img_dir tests/media/448 \
    --out_dir work_dirs/local_test \
    --gpu_per_parallel 1 \
    --parallel_per_task 1
```

### Example 2: HuggingFace dataset

Run evaluation on a HuggingFace-hosted dataset with built-in prompt templates:

```bash
python mmeval/run.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset mmeval_hf@mm-eval/MMBench-V11 \
    --subset en \
    --split test \
    --out_dir work_dirs/mmbench_test \
    --gpu_per_parallel 1 \
    --parallel_per_task 8
```

### Example 3: VLMEvalKit dataset

Use any of 100+ datasets from VLMEvalKit (auto-downloaded):

```bash
# By dataset name (auto-downloads TSV from HuggingFace)
python mmeval/run.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset evalkit@MM-Math \
    --out_dir work_dirs/mm_math_test \
    --gpu_per_parallel 1 \
    --parallel_per_task 8

# By direct TSV URL
python mmeval/run.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset https://huggingface.co/datasets/mm-eval/VLMEvalKit/resolve/main/3DSRBench.tsv \
    --out_dir work_dirs/3dsrbench_test \
    --gpu_per_parallel 1 \
    --parallel_per_task 8

# By local TSV path
python mmeval/run.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset datasets/MM-Math.tsv \
    --out_dir work_dirs/mm_math_local \
    --gpu_per_parallel 1 \
    --parallel_per_task 8
```

### Example 4: Text-only (no media)

```bash
python mmeval/run.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset local@json \
    --infile tests/samples/no_media.json \
    --out_dir work_dirs/text_test \
    --gpu_per_parallel 1 \
    --parallel_per_task 1
```

### Example 5: Multi-GPU parallel inference

Distribute inference across multiple GPUs with automatic data sharding:

```bash
python mmeval/run.py \
    --model_name_or_path Qwen/Qwen2.5-VL-72B-Instruct \
    --dataset mmeval_hf@mm-eval/MMBench-V11 \
    --subset en \
    --split test \
    --out_dir work_dirs/mmbench_72b \
    --gpu_per_parallel 4 \
    --parallel_per_task 2
```

This allocates 4 GPUs per worker and runs 2 parallel workers (requiring 8 GPUs total).

### Example 6: Run a subset of samples

Run a quick deterministic smoke test on 20 randomly selected samples:

```bash
python mmeval/run.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset mmeval_hf@mm-eval/MMBench-V11 \
    --subset en \
    --split test \
    --out_dir work_dirs/mmbench_smoke \
    --gpu_per_parallel 1 \
    --parallel_per_task 4 \
    --sample_num 20 \
    --sample_order random \
    --sample_seed 42
```

---

## Command-Line Reference

All arguments are passed to `python mmeval/run.py` and organized into the following groups.

### Model Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_name_or_path` | str | -- | HuggingFace model ID, local path, or API model name |
| `--dtype` | str | None | Override default dtype (e.g., `float16`, `bfloat16`) |
| `--device_map` | str | None | Device placement strategy (e.g., `auto`) |
| `--attn_implementation` | str | None | Attention implementation (e.g., `flash_attention_2`) |
| `--low_cpu_mem_usage` | bool | None | Reduce CPU memory usage during model loading |

### Data Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | -- | Dataset specifier (see [Supported Datasets](SUPPORTED.md#supported-datasets)) |
| `--infile` | str | None | Input file path (required for `local@json`) |
| `--img_dir` | str | None | Base directory for resolving relative media paths |
| `--split` | str | None | Dataset split (for HuggingFace datasets, e.g., `test`, `dev`) |
| `--template` | str | None | Path to a Jinja2 template file or a template string |
| `--resize` | int | None | Resize all images to this pixel size (e.g., `448`) |
| `--sample_num` | int | None | Number of samples to run (default: all) |
| `--sample_order` | str | head | Sample selection order: `head`, `tail`, or `random` |
| `--sample_seed` | int | 42 | Seed for `--sample_order random` |

### Inference Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--out_dir` | str | -- | Output directory for results and cache |
| `--no-resume` | flag | off | Disable default resume behavior; when set, do not early-exit even if `result.json` already exists |
| `--save_freq` | int | 3 | Number of results to buffer before flushing to SQLite |
| `--max_retry` | int | 1 | Maximum number of full-dataset retry passes |
| `--max_retry_sample` | int | 1 | Maximum retries per sample within a pass |

### Generation Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max_new_tokens` | int | None | Maximum tokens to generate |
| `--max_length` | int | None | Maximum total sequence length |
| `--min_new_tokens` | int | None | Minimum tokens to generate |
| `--min_length` | int | None | Minimum total sequence length |
| `--temperature` | float | None | Sampling temperature |
| `--top_k` | int | None | Top-k sampling |
| `--top_p` | float | None | Nucleus sampling threshold |
| `--min_p` | float | None | Minimum token probability (scaled by top token probability) |
| `--do_sample` | bool | None | Enable sampling (vs. greedy decoding) |
| `--num_beams` | int | None | Beam search width |
| `--early_stopping` | bool | None | Stopping condition for beam-based methods |
| `--max_time` | float | None | Maximum generation time in seconds |
| `--use_cache` | bool | None | Use KV cache to speed up decoding |
| `--cache_implementation` | str | None | Cache class for `generate` (e.g., `quantized`) |
| `--repetition_penalty` | float | None | Repetition penalty (1.0 = no penalty) |
| `--diversity_penalty` | float | None | Diversity penalty for group beam search |
| `--length_penalty` | float | None | Exponential length penalty for beam search |

### Experiment Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--gpu_per_parallel` | int | 1 | Number of GPUs allocated to each parallel worker |
| `--parallel_per_task` | int | 4 | Number of parallel workers to spawn |

---

## Batch Inference

`mmeval/run.py` evaluates **one** model on **one** dataset. To sweep many
models over many datasets in a single command, use `mmeval/batch_infer.py`. It is
a thin scheduler on top of `run.py`: for each model it estimates peak VRAM,
decides how many GPUs a shard needs and how many shards can run in parallel, then
calls `run.py` once per dataset.

```bash
# Two models on one HuggingFace benchmark, GPUs allocated automatically
python mmeval/batch_infer.py \
    --models Qwen2.5-VL-7B-Instruct InternVL3-8B \
    --datasets mmeval_hf@mm-eval/MMBench-en-V11:test \
    --out-dir work_dirs/batch
```

Results are written to `<out-dir>/<dataset>/<model>/result.json`, the same
per-run layout `run.py` produces.

### Selecting models

Choose models by name or by series (the two are mutually exclusive); pass `all`
to select everything. Selection is then narrowed by filters:

```bash
# Every model in two series, newest first
python mmeval/batch_infer.py --model-series qwenvl2d5 internvl3 --sort-by newest ...

# All instruct-tuned, non-quantized local models that support video
python mmeval/batch_infer.py --models all --modalities multi_video_interleave ...
```

| Filter | Default | Description |
|--------|---------|-------------|
| `--model-type` | `non-pretrain` | `pretrain`, `instruct`, `reasoning`, `non-pretrain`, or `all` |
| `--quantization` | `false` | `true`, `false`, or `all` |
| `--api` | `false` | `false` = local only, `true` = API only, `all` = both |
| `--modalities` | -- | keep models supporting at least one listed modality |
| `--sort-by` | registry order | `newest`, `oldest`, `largest`, `smallest` |

Preview the resolved selection without running anything:

```bash
python mmeval/batch_infer.py --model-series qwenvl2d5 --list-models
```

### Specifying datasets

`--datasets` accepts one or more specs in the form `<type>@<value>`:

| Spec | Expands to |
|------|------------|
| `mmeval_hf@org/name:split` | `--dataset mmeval_hf@org/name --split split` |
| `evalkit@Name` | `--dataset evalkit@Name` |
| `local@/path/file.json[:/img_dir]` | `--dataset local@json --infile … [--img_dir …]` |
| `tsv@name-or-path-or-url` | `--dataset name-or-path-or-url` |

### GPU allocation

Each model's VRAM is estimated from its size and architecture (recorded in
`mmeval/model_metadata.json`). A model is given the fewest GPUs that fit it, and
the remaining GPUs run additional shards in parallel; a model that cannot fit on
the available GPUs is **skipped** (reported in the summary). Override the
detected per-GPU VRAM — or disable specific GPUs — with `--gpu-memory` (values in
GiB, in visible-GPU order, `0` to disable):

```bash
# 80GiB on GPU0/GPU1, GPU2 disabled, 40GiB on GPU3
python mmeval/batch_infer.py ... --gpu-memory 80 80 0 40
```

API models need no GPU; control their concurrency with `--api-parallel`.

### Forwarding generation/inference options

Any flag `batch_infer` does not recognize is forwarded verbatim to every `run.py`
invocation, so all [generation](#generation-arguments) and
[inference](#inference-arguments) arguments work unchanged:

```bash
python mmeval/batch_infer.py --models all --datasets evalkit@MMBench_dev_en \
    --out-dir work_dirs/batch --max_new_tokens 256 --sample_num 50 --circular
```

(The per-task `--model_name_or_path`, `--dataset`, `--out_dir`,
`--gpu_per_parallel`, and `--parallel_per_task` are managed by `batch_infer` and
cannot be forwarded.) Add `--save-log` to write a JSON run log to `--out-dir`.

### Maintaining the model metadata

`mmeval/model_metadata.json` is a companion to `mmeval/registry.py`: `registry.py`
remains the source of truth for which models exist and which environment runs
them, while this file adds the size/architecture/modality fields the scheduler
needs. It is a JSON object keyed by model name; for example:

```json
"Qwen2.5-VL-7B-Instruct": {"hf_path": "Qwen/Qwen2.5-VL-7B-Instruct", "series": "qwenvl2d5",
  "model_type": "instruct", "api_model": false, "quantization": false,
  "modalities": ["multi_image_video_interleave", "text"], "parameter_count": 8292166656,
  "num_hidden_layers": 28, "hidden_size": 3584, "num_attention_heads": 28,
  "num_key_value_heads": 4, "max_position_embeddings": 128000, "required_gpu_count": 1}
```

After adding models to `registry.py`, refresh the auto-derived fields (parameter
count, architecture, etc.) and check the two stay in sync with:

```bash
HF_TOKEN=hf_xxx python scripts/update_model_metadata.py
```

---

## Prompt Template System

Simple-MMEval uses [Jinja2](https://jinja.palletsprojects.com/) templates to construct prompts from structured sample data. Templates are resolved in the following priority order:

1. **User template** (`--template` argument) -- highest priority
2. **Dataset template** -- provided by the dataset itself (e.g., HuggingFace `jinja_template` in metadata)
3. **Default template** ([`mmeval/data/default_template.txt`](mmeval/data/default_template.txt)) -- fallback

The default template renders `question`, `options`, and `hint` fields:

```jinja2
{{ question }}{% if options %}
Options:
{% for k, v in options.items() %}{{ k }}. {{ v }}{% if not loop.last %}
{% endif %}{% endfor %}{% endif %}{% if hint %}
Hint: {{ hint }}{% endif %}
```

**Example**: Given a sample with `question`, `options`, and `hint`, the default template produces:

```
What is shown in the image?
Options:
A. A truck
B. A dog
C. A boat
D. A plane
Hint: Please choose the correct option.
```

To use a custom template, either pass a file path or an inline string:

```bash
# File path
--template path/to/my_template.txt

# Inline string
--template "Answer the question: {{ question }}"
```

If a sample's message already contains a `prompt` field, it is used directly and template rendering is skipped for that message.

---

## Resume and Caching

Simple-MMEval writes the final output to `result.json`. During execution, it also uses an SQLite key-value store (`cache.db`) as a temporary, multi-process-safe cache.

**How it works:**

- During an in-progress run, each completed sample is written to `cache.db` in the output directory
- Results are buffered in memory and flushed every `--save_freq` samples (default: 3)
- If a run is interrupted, restarting with the same `--out_dir` reuses existing `cache.db` entries and skips already completed samples
- The cache is shared across all parallel workers using SQLite WAL mode with retry logic for lock contention
- After all workers finish successfully, `run.py` merges cached entries into final `result.json` and deletes `cache.db`

```bash
# Resume an interrupted run (default behavior)
python mmeval/run.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset local@json \
    --infile tests/samples/multi_image_video_interleave.json \
    --img_dir tests/media/448 \
    --out_dir work_dirs/resume_test \
    --gpu_per_parallel 1 \
    --parallel_per_task 8
```

Resume is enabled by default. In `run.py`, this currently controls one behavior: if `result.json` already exists in `--out_dir`, the runner exits immediately. Use `--no-resume` (or `--no_resume`) to force a re-run in that case.

`result.json` is the completed-task artifact. `cache.db` is a temporary file used for incremental persistence and resume; it normally exists only while a run is in progress (or after an interrupted run).

To start fully from scratch, use a new `--out_dir` or clean the existing output directory before running.

---

## Output Format

After inference completes, results are saved to `<out_dir>/result.json` -- a JSON array sorted by `eval-id`. Each entry contains the original sample fields plus an appended assistant message with the model's response:

```json
[
  {
    "id": 0,
    "media": ["truck.png"],
    "messages": [
      {
        "role": "user",
        "question": "<image> Please provide a detailed description of the contents shown in the image.",
        "prompt": "<image> Please provide a detailed description of the contents shown in the image."
      },
      {
        "role": "assistant",
        "response": ["The image shows a dark gray GMC Sierra pickup truck ..."]
      }
    ],
    "eval-id": 0
  }
]
```

Key output fields:

| Field | Description |
|-------|-------------|
| `eval-id` | Framework-assigned integer index (used for sharding and caching) |
| `messages[-1].role` | Always `"assistant"` for the model's response |
| `messages[-1].response` | Model output text as returned by the model backend |
| `media` | Original media paths (PIL Image objects are replaced with `"Image Object"` placeholder) |
