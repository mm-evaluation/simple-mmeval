#!/usr/bin/env python3
"""Push an mm-eval artifact as a MULTI-CONFIG HuggingFace dataset.

Each ``subset`` declared in ``metadata.json`` becomes its own HF dataset
``config`` (BuilderConfig). Splits inside each config are derived by
stripping the ``<subset>_`` prefix from the merged DatasetDict's split names.

Example: for an artifact with subsets ``{en, cn}`` and DatasetDict splits
``{en_dev, en_test, cn_dev, cn_test}``, this pushes:

    repo/en/dev-*.parquet      repo/en/test-*.parquet
    repo/cn/dev-*.parquet      repo/cn/test-*.parquet

and writes a README YAML configs block referencing them.

This is the structural complement to the original push_to_hf.py (which uses
the legacy ``default`` config). New uploads should prefer this path so that
``load_dataset(repo, name=<subset>, split=<split>)`` works the way official
HF benchmarks expect.

Usage::

    python3 push_to_hf_multiconfig.py \\
        --artifact-dir <out> \\
        --repo-id mm-eval/<name> \\
        [--token <hf_token>] \\
        [--cleanup-artifact]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk
from huggingface_hub import HfApi


def _map_splits_to_subsets(splits: list[str], subsets: list[str]) -> dict[str, dict[str, str]]:
    """Return {subset: {hf_split_name: original_split_name}}.

    For each split name in the flat DatasetDict, decide which subset owns
    it. Decision rule:
      1. If split == subset, the subset owns it under the same name.
      2. If split startswith "{subset}_", the subset owns it under the
         remainder (e.g. "en_dev" -> "dev").
      3. If split endswith "_{subset}", the subset owns it (e.g. for the
         MathVerse `text_only` pattern: split "testmini_text_only" ->
         subset "text_only" owns it under "testmini").
      4. Otherwise unassigned (caller decides).

    Ambiguous case: when subsets overlap as prefixes (e.g. "en" / "english"),
    longest-match wins.
    """
    sorted_subsets = sorted(subsets, key=len, reverse=True)
    mapping: dict[str, dict[str, str]] = {s: {} for s in subsets}
    unassigned: list[str] = []
    for sp in splits:
        owner = None
        derived = None
        for s in sorted_subsets:
            if sp == s:
                owner, derived = s, sp
                break
            if sp.startswith(f"{s}_"):
                owner, derived = s, sp[len(s) + 1:]
                break
            if sp.endswith(f"_{s}"):
                owner, derived = s, sp[:-len(s) - 1]
                break
        if owner is None:
            unassigned.append(sp)
        else:
            mapping[owner][sp] = derived
    if unassigned:
        # Fall back: any unassigned split gets owned by the first subset under its
        # original name (only valid for single-subset metadata).
        first = subsets[0]
        for sp in unassigned:
            mapping[first][sp] = sp
    return mapping


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--private", action="store_true")
    p.add_argument("--cleanup-artifact", action="store_true")
    p.add_argument("--commit-message", default=None)
    args = p.parse_args()

    if not args.token:
        try:
            from huggingface_hub import get_token
            args.token = get_token()
        except Exception:
            pass
    if not args.token:
        print("ERROR: provide --token or set HF_TOKEN or run `huggingface-cli login`", file=sys.stderr)
        return 1

    art = Path(args.artifact_dir)
    ds_dir = art / "hf_dataset"
    meta_path = art / "metadata.json"
    if not ds_dir.exists():
        print(f"ERROR: {ds_dir} not found", file=sys.stderr); return 1
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found", file=sys.stderr); return 1

    dd = load_from_disk(str(ds_dir))
    if isinstance(dd, Dataset):
        # Flat Dataset - wrap into DatasetDict with single "train" split
        dd = DatasetDict({"train": dd})
    with open(meta_path) as f:
        meta = json.load(f)

    subsets = list(meta.get("subsets", {}).keys())
    if not subsets:
        print("ERROR: metadata.json has no subsets", file=sys.stderr); return 1

    splits = list(dd.keys())
    mapping = _map_splits_to_subsets(splits, subsets)

    print(f"[push] repo={args.repo_id}")
    print(f"[push] subsets={subsets}")
    for sub, sp_map in mapping.items():
        print(f"[push]   {sub}: {sp_map}")

    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset",
                    private=args.private, exist_ok=True)

    # Push one config per subset, using subset name as HF config_name
    for sub in subsets:
        sub_splits = mapping[sub]
        if not sub_splits:
            print(f"[push] WARN: subset {sub!r} has no splits assigned; skipping")
            continue
        sub_dd = DatasetDict({hf_split: dd[orig] for orig, hf_split in sub_splits.items()})
        msg = args.commit_message or f"Push {sub} config"
        sub_dd.push_to_hub(
            repo_id=args.repo_id,
            config_name=sub,
            set_default=(sub == subsets[0]),
            token=args.token,
            commit_message=msg,
        )
        print(f"[push] subset={sub} -> config={sub} splits={list(sub_dd.keys())}")

    # Upload metadata.json at root (overwrite)
    api.upload_file(
        path_or_fileobj=str(meta_path),
        path_in_repo="metadata.json",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Update metadata.json",
    )
    print(f"[push] metadata.json uploaded to repo root")

    if args.cleanup_artifact:
        shutil.rmtree(art, ignore_errors=True)
        print(f"[push] cleaned artifact dir {art}")
    print(f"\nDone. Run with simple-mmeval:")
    print(f"  python mmeval/run.py \\\\")
    print(f"      --model_name_or_path Qwen3-VL-2B-Instruct \\\\")
    print(f"      --dataset mmeval_hf@{args.repo_id} \\\\")
    print(f"      --subset {subsets[0]} \\\\")
    print(f"      --split <pick-one-of:{','.join(mapping[subsets[0]].values())}> \\\\")
    print(f"      --out_dir work_dirs/{Path(args.repo_id).name} \\\\")
    print(f"      --gpu_per_parallel 1 --parallel_per_task 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
