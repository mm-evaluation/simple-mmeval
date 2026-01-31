# install_envs.sh

Batch creates/updates conda environments from requirements files.

## Prerequisites

- Conda installed and available in PATH
- Bash 4.0+

## Usage

```bash
./install_envs.sh
```

Or control parallel build jobs (useful for packages like flash-attn):

```bash
MAX_JOBS=4 ./install_envs.sh
```

## How It Works

1. Scans `env_files/` for files matching `*_requirements.txt`
2. For each file (e.g., `qwen_requirements.txt`):
   - Creates conda env named `qwen` at `/data/ztw/mmeval_envs/qwen`
   - Installs packages from the requirements file

## Requirements File Format

Standard pip requirements format. **Header comments** (before any blank line) are also installed as packages:

```txt
# flash-attn==2.5.0 --no-build-isolation
# some-special-pkg>=1.0

torch>=2.0
transformers
```

The commented lines at the top will be installed separately after the main requirements.

## Configuration

Edit these variables in the script if needed:

| Variable | Default | Description |
|----------|---------|-------------|
| `REQS_DIR` | `/data/ztw/simple-mmeval-infer-data/env_files` | Requirements files location |
| `CONDA_ENVS_DIR` | `/data/ztw/mmeval_envs` | Where envs are created |
| `PYTHON_VERSION` | `3.10` | Python version for new envs |
| `MAX_JOBS` | `2` | Parallel build jobs |

## Logs

All output is logged to `environment_setup.log` in the current directory.
