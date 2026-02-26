# Contributing to Simple-MMEval

This guide covers how to extend Simple-MMEval with new models and new datasets. For general usage, see the main [README](../../README.md).

---

- [Adding a New Model](#adding-a-new-model)
- [Adding a New Dataset](#adding-a-new-dataset)

---

## Adding a New Model

### Step 1: Create the inference script

Create `mmeval/infer/<model_series>.py` inheriting from the base `Task` class:

```python
import copy
from mmeval.infer.task import Task
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.default_model_kwargs = {}
        self.default_gen_kwargs = {"max_new_tokens": 128}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)
        super().__init__(args)

    def load_model(self, args):
        # Load your model and processor here
        ...

    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        message = sample["messages"][0]

        # Parse input: extract prompt text and media from message
        # message["prompt"] contains the rendered prompt with <image>/<video> placeholders
        # message["media"] contains loaded PIL Images / video paths
        ...

        response = ...  # generate text
        ori_sample["messages"].append({"role": "assistant", "response": response})

        return ori_sample

if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
```

**Key methods to implement:**

| Method | Purpose |
|--------|---------|
| `load_model(self, args)` | Load your model and processor |
| `run_sample(self, sample)` | Run inference on one sample and return the result |
| `parse_input(self, message)` | Parse the message into your model's expected input format |
| `_generate_response(...)` | Separate method for response generation |

For models requiring special tokens, implement a `parse_input` method to convert the framework's generic `<image>` / `<video>` placeholders into model-specific tokens:

```python
def parse_input(self, message: dict):
    prompt = message["prompt"]
    media_list = message.get("media", [])

    # Replace generic placeholders with model-specific tokens
    prompt = prompt.replace("<image>", "<|image|>")
    prompt = prompt.replace("<video>", "<|video|>")

    return prompt, media_list
```

### Step 2: Register in `registry.py`

Add entries to both `series_mapping` and `series_infer_env_mapping` in [`mmeval/registry.py`](mmeval/registry.py):

```python
# In series_mapping — map series key to list of model names:
"my_model": ["MyModel-7B", "MyModel-13B"],

# In series_infer_env_mapping — map series key to env and script:
"my_model": {
    "env": os.path.join(env_dir, "my_model"),
    "infer_file": "my_model.py",
},
```

The model name in `series_mapping` must match the last path component of `--model_name_or_path` (e.g., `Qwen2.5-VL-3B-Instruct` for `Qwen/Qwen2.5-VL-3B-Instruct`).

### Step 3: Create and export the conda environment

```bash
conda create -p envs/my_model python=3.10 -y
conda activate envs/my_model
pip install <your-model-dependencies>
pip freeze > env_files/my_model_requirements.txt
```

> **Tip**: Only install essential packages to keep the environment lightweight.

### Step 4: Test

Run the modality test to verify your implementation:

```bash
export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --model_name_or_path MyModel-7B \
    --dataset local@json \
    --infile tests/samples/single_image_start.json \
    --img_dir tests/media/448 \
    --out_dir work_dirs/my_model_test \
    --gpu_per_parallel 1 \
    --parallel_per_task 1
```

Available test sample files in `tests/samples/`:

| Test File | What it tests |
|-----------|---------------|
| `single_image_start.json` | Single image at the start of the prompt |
| `single_video_start.json` | Single video at the start of the prompt |
| `multi_image_start.json` | Multiple images at the start of the prompt |
| `multi_image_interleave.json` | Multiple images interleaved in the prompt |
| `multi_video_interleave.json` | Multiple videos interleaved in the prompt |
| `multi_image_video_interleave.json` | Mixed images and videos interleaved (most comprehensive) |
| `no_media.json` | Text-only, no media |

Verify that `work_dirs/my_model_test/result.json` is generated and contains valid responses.

---

## Adding a New Dataset

This section explains how to convert a vision-language benchmark into the [mm-eval HuggingFace dataset format](https://huggingface.co/mm-eval) and publish it for use with `--dataset mmeval_hf@mm-eval/YourDataset`.

### Step 1: Find the source and define the prompt layout

1. **Identify the source of truth**
   - Official HuggingFace dataset (e.g., `lmms-lab/MMBench`), or
   - Official file (TSV, JSON, CSV, etc.) from the benchmark authors.

2. **Read the original paper / docs**
   - Note how each key (question, choices, hint, image, etc.) is used in the full prompt.
   - Check if the prompt differs by question type (e.g., multiple-choice vs. open-ended). This informs your Jinja template(s) and possibly per-split metadata.

3. **Define a column mapping**
   - Map the source's column/field names to a canonical set you'll use in `messages` and in the Jinja template (e.g., `hint`, `question`, `options`, `choices`, `answer`, `image`).

### Step 2: Parse and normalize to the required schema

Parse the source and produce rows with these columns in the **default** HF config:

| Column | Description |
|--------|-------------|
| `media` | List of one or more PIL Images per row (HF `Image()`-compatible). At inference time, these are interleaved with the prompt using `<image>` placeholders. |
| `messages` | A single **JSON string**: a list of message objects. Each object must contain all keys your Jinja template expects (e.g., `role`, `question`, `answer`, `options`, `choices`, `hint`). |
| `id` | Unique example ID (string) for tracking and evaluation. |

**Example row (MMBench-style):**

- `media`: one PIL Image
- `messages`: `'[{"role":"user","question":"...","answer":"","options":{"A":"...","B":"..."},"choices":["A","B"],"hint":"..."}]'`
- `id`: `"243"`

Guidelines:

- **Multi-turn**: `messages` is a list of turns. The template is rendered with the first (or appropriate) message's fields.
- **Validation**: Filter out rows where image decode fails or media is missing.
- **Size**: Keep metadata small; store images as PIL/Image in `media`, not as base64 strings.

### Step 3: How the prompt is built (interleave + Jinja)

The final prompt seen by the model is constructed in two stages:

1. **Template rendering** -- The split's Jinja template (stored in the `metadata` config) is rendered with the context from the `messages` entry.
2. **Media interleave** -- Each `<image>` placeholder in the rendered string is replaced by the runner with the corresponding image from `media`, in order.

**Single-image example:**

- Template: `"<image>{% if hint %}Hint: {{ hint }}\n{% endif %}Question: {{ question }}\n..."`
- `messages[0]`: `{ "hint": "See figure.", "question": "What is in the image?" }`
- `media`: `[<PIL Image>]`
- Rendered: `"<image>Hint: See figure.\nQuestion: What is in the image?\n..."` -- the runner replaces `<image>` with the image.

**Multi-image example:**

- Template: `"<image>Image 1. <image>Image 2. Question: {{ question }}"`
- `media`: `[<PIL Image A>, <PIL Image B>]`
- Rendered: `"<image>Image 1. <image>Image 2. Question: ..."` -- first `<image>` is replaced with Image A, second with Image B.

Template variables must match keys present in the message objects (e.g., `hint`, `question`, `options`). See the [Jinja2 documentation](https://jinja.palletsprojects.com/) for template syntax.

### Step 4: Push to your HF account and test

1. **Push the dataset to your HuggingFace account** (not the `mm-eval` org yet):

   - **Default config**: `DatasetDict` with splits (e.g., `test`, `dev`), columns `media`, `messages`, `id`. Use `datasets.Image()` for `media`.

     ```python
     dataset.push_to_hub("YOUR_USERNAME/YOUR_DATASET", private=False)
     ```

   - **Metadata config**: Same splits, config name `metadata`. One row per split with columns `jinja_template`, `version`, `metadata` (JSON string of your column mapping).

     ```python
     metadata_ds.push_to_hub("YOUR_USERNAME/YOUR_DATASET", config_name="metadata")
     ```

2. **Test with Simple-MMEval**:

   ```bash
   export PYTHONPATH=./:$PYTHONPATH

   python mmeval/run.py \
       --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
       --dataset mmeval_hf@YOUR_USERNAME/YOUR_DATASET \
       --split test \
       --out_dir work_dirs/my_dataset_test \
       --gpu_per_parallel 1 \
       --parallel_per_task 1
   ```

   Confirm that the evaluation runs correctly and produces expected results.

### Step 5: Request official publish under mm-eval org

1. **Share with maintainers**:
   - Your HuggingFace repo URL.
   - The test command and result (e.g., screenshot of a successful run).

2. **After review**, maintainers will:
   - Grant access to the **mm-eval** HuggingFace org.
   - You (or they) upload the dataset to `mm-eval/BENCHMARK_NAME`.

3. **Main repo updates** (via PR):
   - Add a test script if applicable.
   - Update the HuggingFace dataset table in the main [README](../../README.md).

### Reference: conversion script example

The script `notebooks/mmeval_push_mmbench_tsv_v11.py` demonstrates the full pipeline for MMBench:

1. Reads a **TSV** file with columns: index, question, hint, A–D, image (base64), split.
2. Uses a **column mapping** to map source columns to canonical keys; defines a **Jinja template** with `<image>`, `hint`, `question`, `options`.
3. Per row: decodes base64 image to PIL, builds one message object, serializes to JSON string. Output row = `media` (PIL Image), `messages` (JSON string), `id`.
4. Filters rows without a valid image.
5. Builds a `DatasetDict` per split, casts `media` to `Image()`, pushes the **default** config; then pushes the **metadata** config with `jinja_template`, `version`, and `metadata` per split.

Use this script as a reference when converting other benchmarks. Keep the same **default + metadata** layout and the **media / messages / id** contract so Simple-MMEval can load your dataset with `--dataset mmeval_hf@mm-eval/YourDataset`.
