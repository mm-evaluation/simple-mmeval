#!/usr/bin/env python3
"""Refresh auto-derivable fields in ``mmeval/model_metadata.json`` from HuggingFace.

The metadata file is a companion to ``mmeval/registry.py`` used by
``mmeval/batch_infer.py`` for VRAM-aware scheduling and model filtering. It is a
JSON object keyed by model name; each value carries the size/architecture and
modality fields the scheduler needs. Curated fields (hf_path, series, model_type,
api_model, quantization, modalities, required_gpu_count) are left untouched; only
the architecture/size fields fetched from HuggingFace are updated.

This script also cross-checks the file against ``mmeval/registry.py`` (the
framework's source of truth) and reports models present in one but not the other,
so the two stay in sync.

Usage:
    HF_TOKEN=hf_xxx python scripts/update_model_metadata.py
"""

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

METADATA_JSON = REPO_ROOT / "mmeval" / "model_metadata.json"

# Fields refreshed from HuggingFace; everything else is curated by hand.
AUTO_FIELDS = [
    "parameter_count", "created_at", "torch_dtype",
    "num_hidden_layers", "hidden_size", "num_attention_heads",
    "num_key_value_heads", "max_position_embeddings",
    "image_tokens", "max_dynamic_patch",
]

# Per-model field order for stable, readable output.
FIELD_ORDER = [
    "hf_path", "series", "model_type", "api_model", "quantization", "modalities",
    "parameter_count", "created_at", "torch_dtype",
    "num_hidden_layers", "hidden_size", "num_attention_heads",
    "num_key_value_heads", "max_position_embeddings",
    "image_tokens", "max_dynamic_patch", "required_gpu_count",
]


def load_metadata(path: Path = METADATA_JSON) -> dict:
    """Load the metadata file as a dict keyed by model name."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object keyed by model name.")
    return data


def write_metadata(data: dict, path: Path = METADATA_JSON) -> None:
    """Write the metadata file with one model per line (compact, diff-friendly)."""
    names = sorted(data)
    lines = ["{"]
    for i, name in enumerate(names):
        fields = data[name]
        ordered = {k: fields[k] for k in FIELD_ORDER if k in fields}
        # Preserve any unexpected extra keys so they are never silently dropped.
        ordered.update({k: v for k, v in fields.items() if k not in ordered})
        tail = "," if i < len(names) - 1 else ""
        lines.append(f"  {json.dumps(name, ensure_ascii=False)}: "
                     f"{json.dumps(ordered, ensure_ascii=False)}{tail}")
    lines.append("}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def check_against_registry(data: dict) -> None:
    """Warn about models that differ between the metadata file and registry.py."""
    try:
        from mmeval.registry import series_mapping
    except Exception as exc:  # pragma: no cover - registry import is optional here
        print(f"WARN: could not import mmeval.registry for cross-check: {exc}")
        return
    registry_models = {m for models in series_mapping.values() for m in models}
    meta_models = set(data)
    missing = sorted(registry_models - meta_models)
    stale = sorted(meta_models - registry_models)
    if missing:
        print(f"NOTE: {len(missing)} model(s) in registry.py but not in the metadata file "
              f"(add them with curated metadata): {', '.join(missing)}")
    if stale:
        print(f"NOTE: {len(stale)} model(s) in the metadata file but not in registry.py "
              f"(run.py cannot resolve these): {', '.join(stale)}")
    if not missing and not stale:
        print(f"Registry cross-check OK: {len(meta_models)} models in sync with registry.py.")


def _int_or_none(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_parameter_count(info) -> int | None:
    st = getattr(info, "safetensors", None)
    total = getattr(st, "total", None) if st is not None else None
    if isinstance(total, (int, float)):
        return int(total)
    card = getattr(info, "cardData", None) or {}
    for key in ("parameters", "parameter_count", "num_parameters"):
        if isinstance(card.get(key), (int, float)):
            return int(card[key])
    return None


def fetch_architecture(hf_path: str, token: str | None) -> dict:
    from huggingface_hub import hf_hub_download
    config_path = hf_hub_download(repo_id=hf_path, filename="config.json", token=token)
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    llm = config.get("llm_config") or {}
    text = config.get("text_config") or {}
    text_cfg = llm or text or config

    vision = config.get("vision_config")
    image_tokens = None
    if vision:
        size = _int_or_none(vision.get("image_size")) or 448
        patch = _int_or_none(vision.get("patch_size")) or 14
        image_tokens = (size // patch) ** 2

    return {
        "num_hidden_layers": _int_or_none(text_cfg.get("num_hidden_layers")),
        "hidden_size": _int_or_none(text_cfg.get("hidden_size")),
        "num_attention_heads": _int_or_none(text_cfg.get("num_attention_heads")),
        "num_key_value_heads": _int_or_none(text_cfg.get("num_key_value_heads")
                                            or text_cfg.get("num_kv_heads")),
        "max_position_embeddings": _int_or_none(text_cfg.get("max_position_embeddings")),
        "torch_dtype": llm.get("torch_dtype") or text.get("torch_dtype") or config.get("torch_dtype"),
        "image_tokens": image_tokens,
        "max_dynamic_patch": _int_or_none(config.get("max_dynamic_patch")),
    }


def fetch_metadata(hf_path: str, api, token: str | None, retries: int = 3) -> tuple[dict | None, str | None]:
    from huggingface_hub.utils import RepositoryNotFoundError
    for attempt in range(retries):
        try:
            info = api.model_info(hf_path)
            meta: dict = {}
            params = extract_parameter_count(info)
            if params is not None:
                meta["parameter_count"] = params
            created = getattr(info, "created_at", None)
            if created:
                meta["created_at"] = created.date().isoformat()
            try:
                meta.update({k: v for k, v in fetch_architecture(hf_path, token).items() if v is not None})
            except Exception as exc:
                print(f"(config.json failed: {exc}) ", end="")
            return meta, None
        except RepositoryNotFoundError:
            return None, "repository not found"
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            return None, f"{type(exc).__name__}: {exc}"
    return None, "max retries exceeded"


def main() -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub not installed. Run: pip install huggingface_hub")

    data = load_metadata()
    check_against_registry(data)

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    updated = 0
    failures: list[str] = []

    for i, (name, model) in enumerate(sorted(data.items()), 1):
        hf_path = (model.get("hf_path") or "").strip()
        if not hf_path:
            continue  # API models and local-only entries have no HF repo.
        print(f"[{i}/{len(data)}] {hf_path} ... ", end="", flush=True)
        meta, error = fetch_metadata(hf_path, api, token)
        if error:
            print(f"FAILED ({error})")
            failures.append(f"{name} ({error})")
            continue
        changed = False
        for field in AUTO_FIELDS:
            new = meta.get(field)
            if new is not None and new != model.get(field):
                model[field] = new
                changed = True
        print("UPDATED" if changed else "OK")
        updated += changed
        time.sleep(0.5)

    write_metadata(data)
    print(f"\nDone. Updated {updated} model(s) in {METADATA_JSON}")
    if failures:
        print(f"\nSkipped/failed ({len(failures)}):")
        for item in failures:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
