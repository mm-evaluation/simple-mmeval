#!/usr/bin/env python3
"""Post-conversion cleanup: remove intermediate artifacts, per-split dirs, preprocessed
files, and HF dataset cache entries after a dataset has been fully converted and pushed.

All deletions are printed before execution. Use --dry-run to preview without deleting.

Examples
--------

# After merging N per-split artifacts into merged/, delete the per-split dirs:
python3 cleanup.py --artifact-splits out/2022 out/2023 out/2024

# After merging+pushing, delete the merged artifact and pre-processed JSON:
python3 cleanup.py --merged-artifact out/merged --preprocessed /tmp/enem_2022.json

# Delete the HF download cache for a specific dataset repo (frees parquet files):
python3 cleanup.py --hf-cache maritaca-ai/enem ibm-granite/ChartNet

# Delete everything: per-split dirs, merged artifact, and HF download cache:
python3 cleanup.py \\
    --artifact-splits out/2022 out/2023 \\
    --merged-artifact out/merged \\
    --hf-cache maritaca-ai/enem \\
    --dry-run  # preview first, then re-run without --dry-run

# Clean up the per-dataset work dir created by convert.py --work-dir
# (contains HF download cache under hf/ and temp files under tmp/):
python3 cleanup.py --work-dir .tmp/conversions/enem_workdir
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List


def _du(path: Path) -> int:
    """Return total bytes used by a path (file or directory)."""
    if path.is_file():
        return path.stat().st_size
    total = 0
    try:
        for entry in os.scandir(path):
            p = Path(entry.path)
            if p.is_symlink():
                continue
            if p.is_dir():
                total += _du(p)
            else:
                total += p.stat().st_size
    except PermissionError:
        pass
    return total


def _human(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n_bytes) < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def _remove(path: Path, dry_run: bool) -> int:
    """Delete path (file or dir). Return bytes freed."""
    if not path.exists():
        return 0
    size = _du(path)
    if dry_run:
        print(f"  [dry-run] would remove {path}  ({_human(size)})")
        return 0
    print(f"  removing {path}  ({_human(size)})")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return size


def _remove_work_dir_protecting_smoke(path: Path, dry_run: bool) -> int:
    """Remove a work dir, but never touch the project's smoke-results tree.

    Smoke results are deliberately persisted under
    `<project-root>/.tmp/smoke_tests/<dataset>/` (legacy: `smoke_<dataset>/`
    or any `smoke_*` subdir under a work dir) so a human can inspect what
    the framework actually saw. A user running
    `cleanup.py --work-dir .tmp/conversions/<dataset>/` shouldn't lose them
    as a side effect of freeing the HF cache.
    """
    if not path.exists():
        return 0
    # Refuse to operate on the project's smoke-results root directly.
    if path.resolve().name == "smoke_tests":
        print(f"  refusing to remove smoke-results tree: {path} "
              "(pass --include-smoke-results to delete)")
        return 0
    # Legacy fallback: a per-dataset work dir that contains `smoke_*` subdirs
    # (old layout where smoke artifacts lived inside the work dir).
    smoke_children = [c for c in path.iterdir() if c.is_dir() and c.name.startswith("smoke_")]
    if not smoke_children:
        return _remove(path, dry_run)
    freed = 0
    for child in path.iterdir():
        if child.is_dir() and child.name.startswith("smoke_"):
            print(f"  preserving smoke results: {child}  ({_human(_du(child))}) "
                  "— pass --include-smoke-results to delete")
            continue
        freed += _remove(child, dry_run)
    return freed


def _hf_cache_paths(repo_id: str, cache_dir: Path) -> List[Path]:
    """Return candidate on-disk cache paths for a HuggingFace dataset repo_id.

    HuggingFace uses two conventions depending on library version and env vars:
      - Newer unified hub cache:  <hf_home>/hub/datasets--<org>--<repo>
      - Legacy datasets cache:    <hf_datasets_cache>/datasets--<org>--<repo>
                                  (also sometimes directly under cache_dir/)

    We probe all plausible locations and return whichever exists.
    """
    slug = repo_id.replace("/", "--")
    name = f"datasets--{slug}"
    candidates = [
        cache_dir / name,           # HF_DATASETS_CACHE or HF_HUB_CACHE / name
        cache_dir.parent / "hub" / name,    # if cache_dir is …/datasets
        cache_dir.parent / "datasets" / name,  # if cache_dir is …/hub
        cache_dir / "hub" / name,  # if cache_dir is the hf_home root
    ]
    return list(dict.fromkeys(candidates))  # deduplicate while preserving order


def cleanup(
    artifact_splits: List[str],
    merged_artifact: str | None,
    preprocessed: List[str],
    hf_cache: List[str],
    work_dirs: List[str],
    hf_cache_dir: Path,
    dry_run: bool,
    include_smoke_results: bool = False,
) -> int:
    freed = 0

    if artifact_splits:
        print("=== Per-split artifact dirs ===")
        for d in artifact_splits:
            p = Path(d)
            if not p.exists():
                print(f"  skip (not found): {p}")
                continue
            if not (p / "metadata.json").exists() and not (p / "hf_dataset").exists():
                print(f"  WARNING: {p} doesn't look like a convert.py artifact (no hf_dataset/ or metadata.json) — skipping")
                continue
            freed += _remove(p, dry_run)

    if merged_artifact:
        print("=== Merged artifact dir ===")
        p = Path(merged_artifact)
        if not p.exists():
            print(f"  skip (not found): {p}")
        elif not (p / "hf_dataset").exists() and not (p / "metadata.json").exists():
            print(f"  WARNING: {p} doesn't look like a convert.py artifact — skipping")
        else:
            freed += _remove(p, dry_run)

    if preprocessed:
        print("=== Preprocessed JSON / media dirs ===")
        for f in preprocessed:
            p = Path(f)
            if not p.exists():
                print(f"  skip (not found): {p}")
                continue
            freed += _remove(p, dry_run)

    if hf_cache:
        print("=== HF dataset cache ===")
        for repo_id in hf_cache:
            candidates = _hf_cache_paths(repo_id, hf_cache_dir)
            found = [c for c in candidates if c.exists()]
            if not found:
                print(f"  skip (no cache found for {repo_id!r}); tried:")
                for c in candidates:
                    print(f"    {c}")
                continue
            for cache_p in found:
                freed += _remove(cache_p, dry_run)

    if work_dirs:
        print("=== Whole work directories ===")
        for d in work_dirs:
            p = Path(d)
            if not p.exists():
                print(f"  skip (not found): {p}")
                continue
            if include_smoke_results:
                freed += _remove(p, dry_run)
            else:
                freed += _remove_work_dir_protecting_smoke(p, dry_run)

    total_label = "would free" if dry_run else "freed"
    print(f"\nTotal {total_label}: {_human(freed)}")
    if dry_run:
        print("Re-run without --dry-run to actually delete.")
    return 0


def _default_hf_cache() -> Path:
    env = os.environ.get("HF_DATASETS_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env:
        return Path(env)
    hf_home_env = os.environ.get("HF_HOME")
    if hf_home_env:
        return Path(hf_home_env) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--artifact-splits", nargs="+", default=[],
        help="Per-split artifact dirs to delete (each has hf_dataset/ + metadata.json). "
             "Safe to delete after merging into a merged/ dir.",
    )
    p.add_argument(
        "--merged-artifact", default=None,
        help="Merged artifact dir to delete (has hf_dataset/ + metadata.json). "
             "Safe to delete after push_to_hf.py succeeds.",
    )
    p.add_argument(
        "--preprocessed", nargs="+", default=[],
        help="Preprocessed JSON files or image dirs to delete (intermediate files "
             "created before convert.py). Can be deleted after the HF artifact is saved.",
    )
    p.add_argument(
        "--hf-cache", nargs="+", default=[], metavar="REPO_ID",
        help="HuggingFace dataset repo ids whose download cache to delete "
             "(e.g. ibm-granite/ChartNet). Safe after the HF artifact is pushed.",
    )
    p.add_argument(
        "--work-dir", nargs="+", dest="work_dirs", default=[],
        help="Entire work directories to remove (use for a full cleanup after push).",
    )
    p.add_argument(
        "--hf-cache-dir", default=None,
        help="Override HF dataset cache root (default: auto-detected from "
             "HF_DATASETS_CACHE / HF_HOME / ~/.cache/huggingface/hub).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be deleted without actually deleting anything.",
    )
    p.add_argument(
        "--include-smoke-results", action="store_true",
        help="Also delete `smoke_*` subdirectories under any --work-dir. "
             "By default smoke results are preserved (they're meant for human inspection).",
    )
    args = p.parse_args()

    if not (args.artifact_splits or args.merged_artifact or args.preprocessed
            or args.hf_cache or args.work_dirs):
        p.print_help()
        return 0

    hf_cache_dir = Path(args.hf_cache_dir) if args.hf_cache_dir else _default_hf_cache()
    return cleanup(
        artifact_splits=args.artifact_splits,
        merged_artifact=args.merged_artifact,
        preprocessed=args.preprocessed,
        hf_cache=args.hf_cache,
        work_dirs=args.work_dirs,
        hf_cache_dir=hf_cache_dir,
        dry_run=args.dry_run,
        include_smoke_results=args.include_smoke_results,
    )


if __name__ == "__main__":
    sys.exit(main())
