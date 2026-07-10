#!/usr/bin/env python3
"""Merge several per-split HF artifacts (each produced by ``convert.py --mode hf``
with a single split) into a single multi-split artifact ready for ``push_to_hf.py``.

Usage
-----
    python3 merge_splits.py \\
        --inputs <dir1> <dir2> ... \\
        --out <merged_dir>

Each input directory is expected to contain ``hf_dataset/`` (DatasetDict with
exactly one split, or a flat Dataset — not supported for merge: use DatasetDict
per split) and ``metadata.json``. The output contains the union of splits in
``hf_dataset/`` plus a merged ``metadata.json``.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from datasets import Dataset, DatasetDict, Image, Sequence, Value, load_from_disk

# Keep in sync with convert.py MODALITY_ORDER
MODALITY_ORDER = (
    "single_image_start", "single_video_start",
    "multi_image_start", "multi_image_interleave",
    "multi_video_interleave", "multi_image_video_interleave", "text",
)

REQUIRED_DEFAULT = [
    ("id", Value("string")),
    ("media", "Sequence(Image())"),
    ("messages", Value("string")),
]


def _check_features(label: str, split: str, features: Dict[str, Any], required: List) -> None:
    for name, expected in required:
        actual = features.get(name)
        if actual is None:
            raise ValueError(f"{label}/{split}: missing required column '{name}' "
                             f"(have: {list(features.keys())})")
        if expected == "Sequence(Image())":
            ok = isinstance(actual, Sequence) and isinstance(actual.feature, Image)
            if not ok:
                raise ValueError(f"{label}/{split}: column '{name}' must be Sequence(Image()), "
                                 f"got {actual!r}")
        elif actual != expected:
            raise ValueError(f"{label}/{split}: column '{name}' must be {expected!r}, "
                             f"got {actual!r}")


def _normalize_for_compare(obj: Any) -> Any:
    """Deepcopy metadata for structural equality (ignore merge-union fields)."""
    o = copy.deepcopy(obj)
    try:
        o.pop("release_date", None)
        subs = o.get("subsets") or {}
        for _sk, block in subs.items():
            if not isinstance(block, dict):
                continue
            block.pop("modalities", None)
            mfs = block.get("mapping_from_source") or {}
            src = mfs.get("source")
            if isinstance(src, dict):
                src.pop("url", None)
            media = mfs.get("media")
            if isinstance(media, dict):
                media.pop("min_items", None)
                media.pop("max_items", None)
    except (TypeError, AttributeError):
        pass
    return o


def _sort_modalities_union(mod_sets: List[Set[str]]) -> List[str]:
    union_list = list({x for s in mod_sets for x in s})
    seen: Set[str] = set()
    ordered: List[str] = []
    for tag in MODALITY_ORDER:
        if tag in union_list and tag not in seen:
            ordered.append(tag)
            seen.add(tag)
    for tag in union_list:
        if tag not in seen:
            ordered.append(tag)
            seen.add(tag)
    return ordered


def _resolve_subset_key(metas: List[Dict[str, Any]], subset: Optional[str]) -> str:
    if subset is not None:
        return subset
    key_sets: List[Set[str]] = []
    for m in metas:
        subs = m.get("subsets") or {}
        key_sets.append(set(subs.keys()))
    if not key_sets:
        raise ValueError("no subsets in metadata")
    first = key_sets[0]
    if len(first) != 1:
        raise ValueError(
            "auto --subset requires each input metadata to define exactly one subset key; "
            f"first input has {sorted(first)} — pass --subset explicitly."
        )
    name = next(iter(first))
    for i, ks in enumerate(key_sets):
        if ks != {name}:
            raise ValueError(
                "metadata subset keys differ across inputs for auto-detect; "
                f"expected only {name!r}, got {sorted(ks)} in input {i}. Pass --subset."
            )
    return name


def merge(input_dirs: List[Path], out_dir: Path, subset: Optional[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    default_dd: Dict[str, Any] = {}
    metas: List[Dict[str, Any]] = []

    for d in input_dirs:
        ds_obj = load_from_disk(str(d / "hf_dataset"))
        if isinstance(ds_obj, Dataset):
            raise ValueError(
                f"{d}: hf_dataset is a flat Dataset; merge_splits expects per-split "
                "DatasetDict artifacts (pass --split when running convert.py for each domain)."
            )
        for split_name, sub in ds_obj.items():
            if split_name in default_dd:
                raise ValueError(f"Duplicate split '{split_name}' in default config "
                                 f"across inputs (conflict at {d})")
            _check_features("default", split_name, sub.features, REQUIRED_DEFAULT)
            default_dd[split_name] = sub

        mp = d / "metadata.json"
        if not mp.exists():
            raise ValueError(f"{d}: missing metadata.json")
        with open(mp, encoding="utf-8") as f:
            metas.append(json.load(f))

    if not metas:
        raise ValueError("no metadata.json loaded")

    subset = _resolve_subset_key(metas, subset)

    base_meta = copy.deepcopy(metas[0])
    names = {m.get("name") for m in metas}
    if len(names) > 1:
        raise ValueError(f"metadata.json name mismatch across inputs: {sorted(names)}")

    for m in metas[1:]:
        a = _normalize_for_compare(base_meta)
        b = _normalize_for_compare(m)
        if a != b:
            raise ValueError(
                "metadata.json files differ in fields other than source.url / media min/max; "
                f"refusing to merge. Compare {input_dirs[0]}/metadata.json vs a later input."
            )

    # Merge source.url, media bounds, modalities, release_date
    all_url: Dict[str, str] = {}
    mins: List[int] = []
    maxs: List[int] = []
    mod_sets: List[set] = []
    dates: List[str] = []

    for m in metas:
        dates.append(str(m.get("release_date") or ""))
        blk = (m.get("subsets") or {}).get(subset)
        if not isinstance(blk, dict):
            raise ValueError(f"{m.get('name')}: subset {subset!r} missing in metadata.json")
        mfs = blk.get("mapping_from_source") or {}
        src = mfs.get("source") or {}
        url_dict = src.get("url") or {}
        if not isinstance(url_dict, dict):
            url_dict = {}
        for k, v in url_dict.items():
            if k in all_url and all_url[k] != v:
                raise ValueError(f"Conflicting source.url[{k!r}] across inputs")
            all_url[k] = v
        media = mfs.get("media") or {}
        if isinstance(media, dict):
            if "min_items" in media:
                mins.append(int(media["min_items"]))
            if "max_items" in media:
                maxs.append(int(media["max_items"]))
        mods = blk.get("modalities") or []
        mod_sets.append(set(mods) if isinstance(mods, list) else set())

    merged = copy.deepcopy(base_meta)
    merged_sub = ((merged.get("subsets") or {}).get(subset)) or {}
    mfs = copy.deepcopy((merged_sub.get("mapping_from_source") or {}))
    src = copy.deepcopy((mfs.get("source") or {}))
    src["url"] = all_url
    mfs["source"] = src
    if mins:
        media = dict(mfs.get("media") or {})
        media["min_items"] = min(mins)
        if maxs:
            media["max_items"] = max(maxs)
        mfs["media"] = media
    merged_mods = _sort_modalities_union(mod_sets)
    merged_sub["mapping_from_source"] = mfs
    merged_sub["modalities"] = merged_mods
    merged.setdefault("subsets", {})[subset] = merged_sub
    merged["release_date"] = max(d for d in dates if d) if any(dates) else merged.get("release_date")

    DatasetDict(default_dd).save_to_disk(str(out_dir / "hf_dataset"))
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"merged splits: {sorted(default_dd.keys())} -> {out_dir}")
    print("  rows per split: " + ", ".join(f"{k}={len(v)}" for k, v in default_dd.items()))


def merge_multi_subset(input_dirs: List[Path], out_dir: Path) -> None:
    """Multi-subset merge: each input may define a different subset.

    All (subset, split) pairs across the inputs are combined into one
    DatasetDict where each pair becomes a split named ``{subset}_{split}``
    (or just ``{subset}`` when the source split name == subset already).
    Each subset's metadata block points only to its own owned splits.

    Use this for multi-config benchmarks like CRPE (exist, relation) or
    MMBench (cc, cn, en) where the source carries N distinct configs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_dd: Dict[str, Any] = {}
    name: Optional[str] = None
    release_dates: List[str] = []
    subsets_meta: Dict[str, Dict[str, Any]] = {}

    for d in input_dirs:
        ds_obj = load_from_disk(str(d / "hf_dataset"))
        if isinstance(ds_obj, Dataset):
            raise ValueError(f"{d}: hf_dataset is a flat Dataset (DatasetDict required)")

        mp = d / "metadata.json"
        if not mp.exists():
            raise ValueError(f"{d}: missing metadata.json")
        with open(mp, encoding="utf-8") as f:
            m = json.load(f)
        if name is None:
            name = m.get("name")
        elif m.get("name") != name:
            raise ValueError(f"name mismatch: {name!r} vs {m.get('name')!r} in {d}")
        if m.get("release_date"):
            release_dates.append(str(m["release_date"]))

        subs = m.get("subsets") or {}
        if not subs:
            raise ValueError(f"{d}: metadata.json has no subsets")

        for subset_key, subset_block in subs.items():
            block = copy.deepcopy(subset_block)
            mfs = block.get("mapping_from_source") or {}
            src = mfs.get("source") or {}
            url_map = src.get("url") or {}
            owned_split_renames: Dict[str, str] = {}
            for split_name, sub in ds_obj.items():
                target = split_name if split_name == subset_key else f"{subset_key}_{split_name}"
                if target in combined_dd:
                    raise ValueError(f"split-name collision after multi-subset merge: {target!r}")
                _check_features("default", target, sub.features, REQUIRED_DEFAULT)
                combined_dd[target] = sub
                owned_split_renames[split_name] = target
            # Rewrite the subset's source.url to reference the renamed split keys
            new_url_map: Dict[str, str] = {}
            for split_name, target in owned_split_renames.items():
                # Prefer URL keyed by original split name; fall back to subset key
                u = url_map.get(split_name) or url_map.get(subset_key)
                if isinstance(url_map, dict) and isinstance(u, str):
                    new_url_map[target] = u
            if new_url_map:
                src = dict(src)
                src["url"] = new_url_map
                mfs = dict(mfs); mfs["source"] = src
                block["mapping_from_source"] = mfs
            if subset_key in subsets_meta:
                raise ValueError(f"subset key collision: {subset_key!r} appears in multiple inputs")
            subsets_meta[subset_key] = block

    merged = {
        "name": name,
        "release_date": (max(release_dates) if release_dates else None),
        "subsets": subsets_meta,
    }
    DatasetDict(combined_dd).save_to_disk(str(out_dir / "hf_dataset"))
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"merged (multi-subset): {len(combined_dd)} splits, "
          f"{len(subsets_meta)} subsets -> {out_dir}")
    print("  rows per split: " + ", ".join(f"{k}={len(v)}" for k, v in combined_dd.items()))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Per-split artifact directories (each has hf_dataset/ + metadata.json)")
    p.add_argument("--out", required=True, help="Output directory for the merged artifact")
    p.add_argument("--subset", default=None,
                   help="Subset key inside metadata.json (default: auto when all inputs share one subset)")
    p.add_argument("--multi-subset", action="store_true",
                   help="Combine inputs that each declare DIFFERENT subset keys. Each (subset, split) "
                        "becomes a HF split named '{subset}_{split}', and the merged metadata.json "
                        "contains one block per source subset with its own source.url mapping. Use for "
                        "multi-config benchmarks like CRPE (exist/relation) or MMBench (cc/cn/en).")
    args = p.parse_args()
    if args.multi_subset:
        if args.subset is not None:
            p.error("--subset and --multi-subset are mutually exclusive")
        merge_multi_subset([Path(d) for d in args.inputs], Path(args.out))
    else:
        merge([Path(d) for d in args.inputs], Path(args.out), args.subset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
