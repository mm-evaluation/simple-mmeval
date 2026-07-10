#!/usr/bin/env python3
"""Push a converted mm-eval HF dataset (the artifact produced by ``convert.py --mode hf``) to HuggingFace Hub.

The user only needs to supply the HF token and the target repo name; everything else
is read from the artifact directory.

Usage
-----
    python3 push_to_hf.py \\
        --artifact-dir /path/to/convert/output \\
        --repo-id <user>/<repo> \\
        --token <hf_token> \\
        [--private] \\
        [--cleanup-artifact]   # delete artifact-dir after a successful push

The artifact directory must contain:
    artifact-dir/
      hf_dataset/      # DatasetDict for the `default` config
      metadata.json    # manifest (top-level prompt_template + mapping_from_source)

Push behaviour:
- Always pushes ``hf_dataset/`` as the ``default`` config.
- If ``metadata.json`` is present, uploads it to the repo root.

After pushing, the script prints the exact ``simple-mmeval`` invocation to copy.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact-dir", required=True, help="Directory produced by convert.py --mode hf")
    p.add_argument("--repo-id", required=True, help="Target HF repo id, e.g. username/dataset-name")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                   help="HF token (or set HF_TOKEN env var)")
    p.add_argument("--private", action="store_true", help="Create the repo as private")
    p.add_argument("--no-create", action="store_true",
                   help="Don't try to create the repo (assume it already exists)")
    p.add_argument("--cleanup-artifact", action="store_true",
                   help="Delete --artifact-dir after a successful push to free local disk space. "
                        "Safe only when the dataset is already pushed and you no longer need the "
                        "local Arrow files (re-run convert.py to regenerate if needed).")
    args = p.parse_args()

    if not args.token:
        print("ERROR: provide --token or set HF_TOKEN", file=sys.stderr)
        return 2

    artifact = Path(args.artifact_dir)
    default_dir = artifact / "hf_dataset"
    metadata_json = artifact / "metadata.json"
    if not default_dir.exists():
        print(f"ERROR: expected {default_dir} to exist (run convert.py --mode hf first)",
              file=sys.stderr)
        return 2

    from datasets import Dataset, DatasetDict
    from datasets import load_from_disk
    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    if not args.no_create:
        api.create_repo(repo_id=args.repo_id, repo_type="dataset",
                        private=args.private, exist_ok=True)
        print(f"[hf] ensured repo exists: {args.repo_id} (private={args.private})")

    print(f"[hf] loading {default_dir}")
    default_dd = load_from_disk(str(default_dir))
    if isinstance(default_dd, DatasetDict):
        split_names = list(default_dd.keys())
        print(f"[hf] pushing default config: splits={split_names}")
        default_dd.push_to_hub(args.repo_id, config_name="default",
                               private=args.private, token=args.token)
        split = split_names[0]
    elif isinstance(default_dd, Dataset):
        print("[hf] pushing default config: flat Dataset (Hub split will be train)")
        default_dd.push_to_hub(args.repo_id, config_name="default",
                               private=args.private, token=args.token)
        split = "train"
    else:
        print(f"ERROR: unexpected object from load_from_disk: {type(default_dd)}", file=sys.stderr)
        return 2

    subset_names: list[str] = []
    if metadata_json.exists():
        print(f"[hf] uploading {metadata_json.name} to repo root")
        api.upload_file(
            path_or_fileobj=str(metadata_json),
            path_in_repo="metadata.json",
            repo_id=args.repo_id,
            repo_type="dataset",
            token=args.token,
            commit_message="Add manifest (metadata.json)",
        )
        import json as _json
        try:
            with open(metadata_json, encoding="utf-8") as _f:
                subset_names = list((_json.load(_f).get("subsets") or {}).keys())
        except Exception:
            subset_names = []
    else:
        print(f"[hf] note: no metadata.json in {artifact} — skipping manifest upload "
              f"(re-run convert.py to emit it)")

    print()
    print("Done. Run with simple-mmeval:")
    print("  python mmeval/run.py \\")
    print("      --model_name_or_path Qwen3-VL-2B-Instruct \\")
    print(f"      --dataset mmeval_hf@{args.repo_id} \\")
    if len(subset_names) > 1:
        print(f"      --subset {subset_names[0]} \\  # pick one of {subset_names}")
    print(f"      --split {split} \\")
    print(f"      --out_dir work_dirs/$(basename {args.repo_id}) \\")
    print("      --gpu_per_parallel 1 --parallel_per_task 1")

    if args.cleanup_artifact:
        print(f"\n[cleanup] deleting local artifact {artifact} ...")
        shutil.rmtree(str(artifact))
        print("[cleanup] done")

    return 0


if __name__ == "__main__":
    sys.exit(main())
