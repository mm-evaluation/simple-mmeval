"""
Extract selected metadata for every model listed in model_information.json
and save it to model_metadata.json.

Required package:
    pip install huggingface_hub
"""

import json
from pathlib import Path
from huggingface_hub import HfApi

INPUT_FILE  = "model_information.json"
OUTPUT_FILE = "model_metadata.json"

api = HfApi()                     # authenticated account not required for public models

# ----------------------------------------------------------------------
# 1. Load the list of models you already have
# ----------------------------------------------------------------------
with open(INPUT_FILE, "r", encoding="utf-8") as fp:
    raw_models = json.load(fp)    # {"Alias": {"model_path": "author/model_name"}, ...}

result = {}

# ----------------------------------------------------------------------
# 2. Query Hub metadata for each model
# ----------------------------------------------------------------------
for alias, meta in raw_models.items():
    repo_id = meta["model_path"]          # e.g. "meta-llama/Llama-3-8B"
    try:
        info = api.model_info(repo_id)    # Hugging Face API call
    except Exception as err:
        print(f"[WARN] Skipping {repo_id}: {err}")
        continue

    card   = info.cardData or {}          # YAML front-matter from the model card

    # Flexible fall-backs for fields that may appear under different names
    dataset_used = (
        card.get("datasets") or
        card.get("dataset")  or
        card.get("dataset_name")
    )

    result[alias] = {
        "repo_id"       : repo_id,
        "created_at"    : getattr(info, "created_at", None)  and info.created_at.date().isoformat(),
        "last_modified" : getattr(info, "lastModified", None) and info.lastModified.date().isoformat(),
        "downloads"     : info.downloads,
        "parameter_size": getattr(info, "safetensors", None),
        "training_dataset"       : dataset_used,
    }

# ----------------------------------------------------------------------
# 3. Save the collected metadata
# ----------------------------------------------------------------------
Path(OUTPUT_FILE).write_text(json.dumps(result, indent=2, ensure_ascii=False))
print(f"✔ Saved metadata for {len(result)} models ➜ {OUTPUT_FILE}")
