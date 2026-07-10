#!/usr/bin/env python3
"""Inspect a source dataset and print its schema + a sample row.

Examples
--------
    python3 inspect_source.py --hf lmms-lab/VizWiz-VQA --split val
    python3 inspect_source.py --json data.json --media-dir media/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


def short(value: Any, limit: int = 200) -> str:
    if value is None:
        return "None"
    if hasattr(value, "size") and hasattr(value, "mode"):  # PIL.Image
        return f"<PIL.Image mode={value.mode} size={value.size}>"
    s = repr(value)
    if len(s) > limit:
        s = s[:limit] + "…"
    return s


def describe_value(value: Any) -> str:
    t = type(value).__name__
    if isinstance(value, list):
        if not value:
            return "list[empty]"
        return f"list[{type(value[0]).__name__}] (len={len(value)})"
    if isinstance(value, dict):
        return f"dict (keys={list(value.keys())[:8]})"
    return t


def inspect_hf(repo_id: str, split: str, n: int = 1) -> None:
    from datasets import load_dataset

    print(f"# HF dataset: {repo_id} (split={split})")
    ds = load_dataset(repo_id, split=split, streaming=True)
    it = iter(ds)
    rows = []
    for _ in range(max(n, 1)):
        try:
            rows.append(next(it))
        except StopIteration:
            break
    if not rows:
        print("  (no rows)")
        return
    print(f"  columns: {list(rows[0].keys())}")
    for k in rows[0].keys():
        v = rows[0][k]
        print(f"    {k}: {describe_value(v)} -> {short(v, 160)}")
    if n > 1:
        print(f"\n# additional samples (truncated):")
        for i, row in enumerate(rows[1:], 1):
            print(f"  [{i}] " + ", ".join(f"{k}={short(row[k], 80)}" for k in rows[0].keys()))


def inspect_json(path: str, media_dir: str | None) -> None:
    print(f"# Local JSON: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print(f"  ERROR: top-level should be a list, got {type(data).__name__}")
        return
    print(f"  rows: {len(data)}")
    if not data:
        return
    sample = data[0]
    print(f"  top-level keys: {list(sample.keys())}")
    for k, v in sample.items():
        print(f"    {k}: {describe_value(v)} -> {short(v, 160)}")
    media = sample.get("media", [])
    if media_dir and isinstance(media, list) and media:
        first = media[0]
        if isinstance(first, str):
            full = os.path.join(media_dir, first)
            print(f"  first media path: {full} (exists={os.path.exists(full)})")


def main() -> int:
    p = argparse.ArgumentParser(description="Inspect a source dataset for dataset-upload.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf", help="HuggingFace dataset repo id (e.g. lmms-lab/VizWiz-VQA)")
    src.add_argument("--json", help="Local JSON file path")
    p.add_argument("--split", default=None, help="HF split (required for --hf)")
    p.add_argument("--media-dir", help="Local media directory (for --json)")
    p.add_argument("-n", type=int, default=1, help="Number of sample rows to peek (HF only)")
    args = p.parse_args()

    if args.hf:
        if not args.split:
            p.error("--hf requires --split")
        inspect_hf(args.hf, args.split, args.n)
    else:
        inspect_json(args.json, args.media_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
