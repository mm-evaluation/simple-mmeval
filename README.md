# Simple-MMEval

**A Self-Contained, Scalable Framework for Multimodal Model Evaluation**

**50 model series** (HuggingFace models + commercial APIs) | **3 dataset backends** | **Multi-GPU parallel scheduling**

[![contributors](https://img.shields.io/github/contributors/mm-evaluation/simple-mmeval)](https://github.com/mm-evaluation/simple-mmeval/graphs/contributors) [![forks](https://img.shields.io/github/forks/mm-evaluation/simple-mmeval)](https://github.com/mm-evaluation/simple-mmeval/network/members) [![stars](https://img.shields.io/github/stars/mm-evaluation/simple-mmeval)](https://github.com/mm-evaluation/simple-mmeval/stargazers) [![issues](https://img.shields.io/github/issues/mm-evaluation/simple-mmeval)](https://github.com/mm-evaluation/simple-mmeval/issues) [![license](https://img.shields.io/github/license/mm-evaluation/simple-mmeval)](LICENSE)

---

📦 [Installation](#installation) | 🚀 [Quick Start](#quick-start) | 📊 [Supported Models and Datasets](#supported-models-and-datasets) | 🤝 [Contributing](#contributing) | 📎 [Citation](#citation)

## Key Features

- **Simple** -- Set up a conda environment and run inference with a single command; no complex configuration or pipeline assembly required
- **Self-contained** -- Each model series has its own complete inference implementation in a single file and its own conda environment, eliminating dependency conflicts
- **Scalable** -- Automatic multi-GPU parallel scheduling with worker-based data sharding; lazy loading for memory-efficient processing of large datasets
- **Modular** -- Pluggable dataset loaders (local JSON, HuggingFace, VLMEvalKit TSV), Jinja2 prompt templates, and a registry-based model system that can be extended independently

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/mm-evaluation/simple-mmeval.git
cd simple-mmeval
```

### 2. Set up model environments

Each model series uses its own isolated conda environment. Pre-exported requirement files are in `env_files/`.

```bash
# Create a model-specific environment (example: Qwen2.5-VL)
conda create -p envs/qwenvl python=3.10 -y
conda activate envs/qwenvl
pip install -r env_files/qwenvl_requirements.txt
```

> **Note**: Some requirement files contain commented-out dependencies (e.g., `# flash-attn==2.7.3`) that require manual installation. Check the top of each requirement file and install these packages separately if needed:
> ```bash
> pip install flash-attn==2.7.3 --no-build-isolation
> ```

### 3. Configure environment directory (optional)

Set `ENV_DIR` to the parent directory that contains all per-model environment folders. The framework resolves each environment as `$ENV_DIR/<env_name>` (e.g., `$ENV_DIR/qwenvl`, `$ENV_DIR/gemma3`). If not set, conda environments are resolved by name.

```bash
# Example: if your environments are at /data/envs/qwenvl, /data/envs/gemma3, ...
export ENV_DIR=/data/envs
```

For API models (OpenAI, Gemini, Claude, etc.), set the corresponding API keys in your environment or `.env` file:

```bash
export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=...
export ANTHROPIC_API_KEY=...
```

## Quick Start

Run Qwen2.5-VL-3B on a local image/video test set using a single GPU:

```bash
export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset local@json \
    --infile tests/samples/multi_image_video_interleave.json \
    --img_dir tests/media/448 \
    --out_dir work_dirs/local_test \
    --gpu_per_parallel 1 \
    --parallel_per_task 1
```

For more examples (HuggingFace datasets, VLMEvalKit benchmarks, multi-GPU parallel inference, etc.), command-line reference, prompt templates, caching, and output format, see the **[Usage Guide](docs/en/USAGE.md)**.

## Supported Models and Datasets

Simple-MMEval supports **50 model series** (44 open source + 6 API providers) and **3 dataset backends** (local JSON, HuggingFace, VLMEvalKit 100+ benchmarks).

See the complete list of all supported models and datasets in **[SUPPORTED.md](docs/en/SUPPORTED.md)**.

## Contributing

Simple-MMEval is designed to be easily extended with new models and datasets.

- **Add a new model** -- implement a single Python file inheriting from `Task`, register it in `registry.py`, create a conda environment, and test
- **Add a new dataset** -- convert a benchmark to the mm-eval HuggingFace format (`media` / `messages` / `id` + Jinja template), push to HuggingFace, and test with `--dataset mmeval_hf@`

See the full step-by-step guide in **[CONTRIBUTING.md](docs/en/CONTRIBUTING.md)**.

## Citation

If you find Simple-MMEval useful in your research, please consider citing:

```bibtex
@misc{zhao2026simplemmeval,
  title        = {Simple-MMEval: A Self-Contained, Scalable Framework for Multimodal Model Evaluation},
  author       = {Zhao, Tianwei and Li, Yijiang and Yang, Snorf and Wang, Bingyang},
  year         = {2026},
  url          = {https://github.com/mm-evaluation/simple-mmeval},
  note         = {GitHub repository}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Contact

For questions, suggestions, or collaboration inquiries, please reach out to [simple.mmeval@gmail.com](mailto:simple.mmeval@gmail.com).
