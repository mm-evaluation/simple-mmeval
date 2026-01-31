# install_envs.sh

Automatically creates and updates conda environments from `*_requirements.txt` files.

## Prerequisites

- Conda installed and available in PATH

## Configuration

Edit these variables in the script as needed:

| Variable | Default | Description |
|----------|---------|-------------|
| `REQS_DIR` | `/data/ztw/simple-mmeval-infer-data/env_files` | Directory containing requirements files |
| `CONDA_ENVS_DIR` | `/data/ztw/mmeval_envs` | Directory where conda environments are created |
| `PYTHON_VERSION` | `3.10` | Python version for new environments |

You can also set `MAX_JOBS` environment variable (default: 2) to control parallel build jobs.

## Usage

```bash
# Run directly
./install_envs.sh
```

## How It Works

1. Scans `REQS_DIR` for all `*_requirements.txt` files
2. For each file, creates a conda environment named after the file prefix
   - Example: `foo_requirements.txt` → environment named `foo`
   - If the environment already exists, skips creation and updates packages only
3. Installs packages listed in the requirements file
4. **Special feature**: Header comments (contiguous `#` lines at the top) are also treated as pip requirements

## Output

- All logs are written to `environment_setup.log` in the current directory
- Environment paths: `CONDA_ENVS_DIR/<env_name>`

