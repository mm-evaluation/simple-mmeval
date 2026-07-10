#!/usr/bin/env python3
"""Round-trip a converted artifact through Simple-MMEval's data loaders to verify it.

Usage
-----

# Validate a local artifact (must be the directory with data.json + media/)
python3 validate.py --simple-mmeval /path/to/simple-mmeval --local /path/to/out

# Validate an HF artifact on disk (no push required) — patches mmeval.data.mmeval_hf
python3 validate.py --simple-mmeval /path/to/simple-mmeval --hf /path/to/out --split val

# Standalone audit (no simple-mmeval needed) — checks metadata.json + placeholder counts
python3 validate.py --audit /path/to/out/hf_dataset --metadata /path/to/out/metadata.json

What it does
------------
- For local: instantiates ``LocalJSONDataset``, iterates a few rows, prints the
  rendered prompt and confirms the media files load as PIL Images.
- For HF: instantiates ``MMEvalHFDataset`` against the on-disk DatasetDict/Dataset
  and ``metadata.json`` at the artifact root.
- For --audit: no simple-mmeval required; checks (1) metadata.json has no local paths,
  (2) source.url keys match actual split names and no deprecated repo/links fields exist,
  (3) every row's placeholder count matches its media count using the Jinja template.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from unittest.mock import patch


def _build_jinja_env():
    from jinja2 import Environment
    env = Environment()
    env.globals.update({
        "zip": zip, "enumerate": enumerate, "len": len, "range": range,
        "list": list, "dict": dict, "str": str, "int": int, "float": float,
        "bool": bool, "sum": sum, "max": max, "min": min,
    })
    return env


VIDEO_MODALITIES = {"single_video_start", "multi_video_interleave", "multi_image_video_interleave"}

VIDEO_EXTS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv",
    ".mpeg", ".mpg", ".m4v", ".3gp", ".3g2", ".ts", ".mts", ".vob", ".gif",
}


def _is_video_ext(path: str) -> bool:
    ext = os.path.splitext(path.split("?")[0])[-1].lower()
    return ext in VIDEO_EXTS


def audit_video_metadata(subsets: dict, issues: list) -> None:
    """Check video-specific metadata requirements."""
    for sk, sv in subsets.items():
        mods = set(sv.get("modalities") or [])
        has_video = bool(mods & VIDEO_MODALITIES)
        if not has_video:
            continue
        vs = sv.get("video_storage")
        if vs is None:
            issues.append(
                f"subset={sk!r}: modalities include video tags {sorted(mods & VIDEO_MODALITIES)} "
                f"but no 'video_storage' block is present in metadata.json. "
                f"Add a video_storage block (see metadata-json.md)."
            )
            continue
        if not isinstance(vs, dict):
            issues.append(f"subset={sk!r}: video_storage must be an object, got {type(vs).__name__}")
            continue
        fmt = vs.get("format")
        if fmt not in ("files", "archives"):
            issues.append(
                f"subset={sk!r}: video_storage.format must be 'files' or 'archives', got {fmt!r}"
            )
        if not vs.get("media_root"):
            issues.append(f"subset={sk!r}: video_storage.media_root is missing or empty")
        if fmt == "archives":
            af = vs.get("archive_format")
            if af not in ("zip", "tar", "tar.gz"):
                issues.append(
                    f"subset={sk!r}: video_storage.archive_format must be 'zip'/'tar'/'tar.gz' "
                    f"when format='archives', got {af!r}"
                )
            afiles = vs.get("archive_files") or []
            if not afiles:
                issues.append(
                    f"subset={sk!r}: video_storage.archive_files is empty but format='archives'"
                )


def audit_video_local(artifact_dir: Path, metadata_path: Path, issues: list, max_check: int = 200) -> None:
    """Check that video files referenced in a local artifact actually exist."""
    data_json = artifact_dir / "data.json"
    if not data_json.exists():
        return
    media_dir_path = artifact_dir / "media"

    with open(metadata_path, encoding="utf-8") as f:
        meta = json.load(f)
    subsets = meta.get("subsets") or {}
    has_video = any(
        bool(set(sv.get("modalities") or []) & VIDEO_MODALITIES)
        for sv in subsets.values()
    )
    if not has_video:
        return

    with open(data_json, encoding="utf-8") as f:
        data = json.load(f)

    missing = 0
    empty = 0
    bad_ext = 0
    checked = 0
    for entry in data[:max_check]:
        for m in (entry.get("media") or []):
            if not isinstance(m, str):
                continue
            if not _is_video_ext(m):
                continue
            checked += 1
            full = media_dir_path / m
            if not full.exists():
                missing += 1
                if missing <= 3:
                    issues.append(f"video file missing: {full}")
            elif full.stat().st_size == 0:
                empty += 1
                if empty <= 3:
                    issues.append(f"video file empty (0 bytes): {full}")
            ext = os.path.splitext(m)[-1].lower()
            if ext not in VIDEO_EXTS:
                bad_ext += 1
    if missing > 3:
        issues.append(f"  ... {missing} total missing video files (showing first 3)")
    if empty > 3:
        issues.append(f"  ... {empty} total empty video files (showing first 3)")
    if checked > 0:
        print(f"  [video] checked {checked} video refs in first {min(len(data), max_check)} rows: "
              f"{missing} missing, {empty} empty, {bad_ext} bad extension")


def audit_artifact(hf_dataset_dir: Path, metadata_path: Path) -> int:
    """Standalone audit: metadata + placeholder/media count check.

    Does not require simple-mmeval to be installed.  Returns 0 if clean,
    1 if any issues found.
    """
    from datasets import load_from_disk

    VALID_MODALITIES = {
        "single_image_start", "single_video_start",
        "multi_image_start", "multi_image_interleave",
        "multi_video_interleave", "multi_image_video_interleave", "text",
    }
    VALID_TASK_TYPES = {"vqa", "multiple_choice_vqa", "captioning"}

    issues: list[str] = []

    # 1. metadata.json: load and check for local paths
    if not metadata_path.exists():
        print(f"ERROR: {metadata_path} not found", file=sys.stderr)
        return 2
    with open(metadata_path, encoding="utf-8") as f:
        meta = json.load(f)

    meta_text = json.dumps(meta, ensure_ascii=False)
    local_paths = re.findall(r'"/(?:raid|home|root|tmp)/[^"]+', meta_text)
    if local_paths:
        for lp in local_paths:
            issues.append(f"metadata.json: local path found: {lp!r}")

    # 2. metadata.json schema checks
    subsets = meta.get("subsets") or {}

    # Try to load the dataset once for split cross-checks
    try:
        loaded = load_from_disk(str(hf_dataset_dir))
        from datasets import DatasetDict
        actual_splits: set[str] = set(loaded.keys()) if isinstance(loaded, DatasetDict) else {"train"}
    except Exception:
        actual_splits = set()

    for sk, sv in subsets.items():
        # 2a. Modalities: must use the supported taxonomy
        mods = sv.get("modalities") or []
        bad_mods = set(mods) - VALID_MODALITIES
        if bad_mods:
            issues.append(
                f"subset={sk!r}: invalid modality tag(s) {sorted(bad_mods)}; "
                f"allowed: {sorted(VALID_MODALITIES)}"
            )

        # 2b. task_type: must be from the supported taxonomy
        tt = sv.get("task_type", "")
        if tt not in VALID_TASK_TYPES:
            issues.append(
                f"subset={sk!r}: invalid task_type {tt!r}; allowed: {sorted(VALID_TASK_TYPES)}"
            )

        mfs = sv.get("mapping_from_source") or {}
        src = mfs.get("source") or {}

        # 2c. Deprecated fields: 'repo' and 'links' must not be present
        if "repo" in src:
            issues.append(
                f"subset={sk!r}: source.repo is deprecated — remove it; "
                "provenance belongs in source.url"
            )
        if "links" in src:
            issues.append(
                f"subset={sk!r}: source.links is deprecated — merge into source.url dict"
            )

        # 2d. source.url must be a dict of split→URL
        url = src.get("url")
        if url is None:
            issues.append(f"subset={sk!r}: source.url is missing")
        elif not isinstance(url, dict):
            issues.append(
                f"subset={sk!r}: source.url must be a dict {{split: url, ...}}, "
                f"got {type(url).__name__}"
            )
        elif actual_splits:
            for url_key in url:
                if url_key not in actual_splits:
                    issues.append(
                        f"subset={sk!r}: source.url key {url_key!r} not in actual "
                        f"splits {sorted(actual_splits)}"
                    )

        # 2e. prompt_template_source must document prompt provenance
        pts = sv.get("prompt_template_source")
        valid_origins = {"official", "source_column", "fallback"}
        if not isinstance(pts, dict):
            issues.append(
                f"subset={sk!r}: prompt_template_source is missing or not an object"
            )
        else:
            origin = pts.get("origin")
            if origin not in valid_origins:
                issues.append(
                    f"subset={sk!r}: prompt_template_source.origin must be one of "
                    f"{sorted(valid_origins)}, got {origin!r}"
                )
            reference = pts.get("reference")
            if not isinstance(reference, str) or not reference.strip():
                issues.append(
                    f"subset={sk!r}: prompt_template_source.reference must be a non-empty string"
                )

    # 2f. Video-specific metadata checks
    audit_video_metadata(subsets, issues)

    # 2g. Video file existence checks (local artifacts only)
    local_artifact_dir = hf_dataset_dir.parent
    if (local_artifact_dir / "data.json").exists():
        audit_video_local(local_artifact_dir, metadata_path, issues)

    # 3. Load dataset and run placeholder/media check for each split
    try:
        loaded = load_from_disk(str(hf_dataset_dir))
    except Exception as e:
        issues.append(f"load_from_disk failed: {e}")
        _print_issues(issues)
        return 1

    from datasets import DatasetDict, Dataset
    split_ds: dict[str, Dataset] = {}
    if isinstance(loaded, DatasetDict):
        split_ds = dict(loaded)
    else:
        split_ds = {"train": loaded}

    env = _build_jinja_env()

    for split_name, ds in split_ds.items():
        # Pick the right subset template for this split.
        # Strategy: find the subset whose source.url contains this split name.
        tmpl_str = None
        for sk, sv in subsets.items():
            mfs = sv.get("mapping_from_source") or {}
            src = mfs.get("source") or {}
            url_dict = src.get("url") or {}
            if isinstance(url_dict, dict) and split_name in url_dict:
                tmpl_str = sv.get("prompt_template")
                break
        if tmpl_str is None and len(subsets) == 1:
            tmpl_str = next(iter(subsets.values())).get("prompt_template")
        if not tmpl_str:
            issues.append(f"Split '{split_name}': no template found in metadata.json")
            continue

        tmpl = env.from_string(tmpl_str)
        n_rows = len(ds)
        n_ph_mismatches = 0
        n_repeated_img1 = 0
        # Track row-id uniqueness so we surface accidental row duplication
        # (e.g., a split assembled by concatenating two copies of the same
        # Arrow file). 1k duplicates inside 2k rows is silent unless we check.
        # Additionally track per-source_id row counts when the message dict
        # carries a `source_id` field — that signals the converter applied
        # the composite-id convention `{source_id}_q{k}`.  Multiple rows
        # may share a source_id (that is the whole point of the convention),
        # but their row ids must be distinct.
        seen_ids: dict[str, int] = {}
        source_ids_seen: dict[str, list[str]] = {}
        for i, row in enumerate(ds):
            rid = row.get("id")
            if rid is not None:
                seen_ids[rid] = seen_ids.get(rid, 0) + 1
            try:
                msg0 = json.loads(row["messages"])[0]
            except Exception as e:
                issues.append(f"Split '{split_name}' row[{i}] id={rid}: messages JSON parse error: {e}")
                continue
            src_id = msg0.get("source_id") if isinstance(msg0, dict) else None
            if src_id:
                source_ids_seen.setdefault(src_id, []).append(str(rid))
            msg = msg0
            try:
                rendered = tmpl.render(**msg)
            except Exception as e:
                issues.append(f"Split '{split_name}' row[{i}] id={row.get('id')}: template error: {e}")
                continue
            n_ph  = len(re.findall(r"<(image|video)>", rendered))
            n_med = len(row["media"]) if row["media"] else 0
            if n_ph != n_med:
                n_ph_mismatches += 1
                if n_ph_mismatches <= 3:
                    issues.append(
                        f"Split '{split_name}' row[{i}] id={row.get('id')}: "
                        f"{n_ph} placeholder(s) != {n_med} media item(s)"
                    )
            # Check for <image> (plain) embedded in the question field while the
            # template already contributes <image> placeholders.  This pattern means
            # the converter treated a numbered figure reference (<image1>) as an
            # actual media placeholder, producing an off-by-one or duplicate count.
            # The correct pattern: the template owns <image>/<video> placeholders;
            # the question field should only contain numbered refs like <image1>.
            q = msg.get("question", "")
            tmpl_contributes_ph = bool(re.search(r"<(image|video)>", tmpl_str))
            if tmpl_contributes_ph and "<image>" in q:
                n_ph_mismatches += 1  # count as a structural issue
                if n_ph_mismatches <= 3:
                    issues.append(
                        f"Split '{split_name}' row[{i}] id={row.get('id')}: "
                        f"<image> found inside question text while template already "
                        f"provides placeholders — likely a mis-converted <image1> "
                        f"(numbered figure label). Restore the original <imageN> text."
                    )

            # Informational: count rows with <imageN> text refs (expected/harmless).
            if re.search(r"<image\d+>", q):
                n_repeated_img1 += 1

        if n_ph_mismatches > 3:
            issues.append(f"Split '{split_name}': {n_ph_mismatches} total issues (showing first 3)")

        # Surface duplicate row ids (counted across the full split). These
        # must be unique even when multiple rows legitimately share a
        # source_id (the composite-id convention assigns distinct
        # `{source_id}_q{k}` to each).
        dup_ids = {k: v for k, v in seen_ids.items() if v > 1}
        if dup_ids:
            n_dup = sum(v - 1 for v in dup_ids.values())
            issues.append(
                f"Split '{split_name}': {len(dup_ids)} duplicate id(s) "
                f"({n_dup} extra rows). First few: "
                f"{list(dup_ids.items())[:3]}"
            )

        # When the composite-id convention is in use (source_id present on
        # any row), require that rows sharing a source_id have distinct ids.
        if source_ids_seen:
            broken = {sid: ids for sid, ids in source_ids_seen.items()
                      if len(ids) > len(set(ids))}
            if broken:
                first = list(broken.items())[:3]
                issues.append(
                    f"Split '{split_name}': composite-id convention violated — "
                    f"{len(broken)} source_id(s) have collisions among row ids. "
                    f"First few: {first}"
                )
            n_multi_sid = sum(1 for ids in source_ids_seen.values() if len(ids) > 1)
            if n_multi_sid:
                print(f"  [{split_name}] composite-id convention active: "
                      f"{n_multi_sid}/{len(source_ids_seen)} source_id(s) span >1 row")

        warn_refs = f"  (info: {n_repeated_img1} rows with <imageN> text refs — OK)" if n_repeated_img1 else ""
        status = "OK" if not n_ph_mismatches and not dup_ids else "FAIL"
        print(f"  [{split_name}] {n_rows} rows, placeholder/media: {status}{warn_refs}")

    _print_issues(issues)
    return 1 if issues else 0


def _print_issues(issues: list[str]) -> None:
    if issues:
        print(f"\n{'='*60}")
        print(f"AUDIT: {len(issues)} issue(s) found:")
        for iss in issues:
            print(f"  - {iss}")
        print("="*60)
    else:
        print("\nAUDIT: no issues found.")


def validate_local(simple_mmeval: Path, artifact: Path, n: int) -> int:
    sys.path.insert(0, str(simple_mmeval))
    from mmeval.data.local import LocalJSONDataset
    ns = argparse.Namespace(
        infile=str(artifact / "data.json"),
        img_dir=str(artifact / "media"),
        parallel_per_task=1, rank=0, template=None, resize=None,
    )
    ds = LocalJSONDataset(ns)
    ds.setup_parallel()
    print(f"[local] global rows: {ds.global_length}")
    if ds.global_length == 0:
        print("[local] EMPTY (no rows)")
        return 1
    for i in range(min(n, ds.global_length)):
        s = ds[i]
        msg = s["messages"][0]
        media_imgs = msg.get("media", [])
        sizes = [getattr(m, "size", "video") for m in media_imgs]
        print(f"  [{i}] id={s.get('id')} eval-id={s.get('eval-id')} media={sizes} "
              f"prompt={msg['prompt'][:80]!r}")
    print("[local] OK")
    return 0


def validate_hf(simple_mmeval: Path, artifact: Path, split: str | None, subset: str | None, n: int) -> int:
    from datasets import Dataset, DatasetDict
    from datasets import load_from_disk

    sys.path.insert(0, str(simple_mmeval))
    from mmeval.data import mmeval_hf as mhf
    from mmeval.data.mmeval_hf import MMEvalHFDataset

    meta_path = artifact / "metadata.json"
    if not meta_path.exists():
        print(f"ERROR: missing {meta_path}", file=sys.stderr)
        return 2

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    subsets = meta.get("subsets") or {}
    if subset:
        subset_name = subset
        if subset_name not in subsets:
            print(f"ERROR: subset {subset_name!r} not in metadata {list(subsets)}", file=sys.stderr)
            return 2
    elif len(subsets) == 1:
        subset_name = next(iter(subsets))
    else:
        print(f"ERROR: metadata has multiple subsets {list(subsets)}; pass --subset", file=sys.stderr)
        return 2
    blk = subsets[subset_name]
    tmpl = (blk.get("prompt_template") or "")[:80]
    print(f"[hf] metadata subset={subset_name!r} prompt_template[:80]={tmpl!r}")

    loaded = load_from_disk(str(artifact / "hf_dataset"))

    resolved_split = split
    if resolved_split is None:
        if isinstance(loaded, DatasetDict):
            resolved_split = next(iter(loaded.keys()))
        else:
            resolved_split = "train"

    def fake_hf_hub_download(*_a, **_k):
        return str(meta_path)

    def fake_load_dataset(repo_id, name="default", split=None, **_kwargs):
        if name != "default":
            raise ValueError(f"unexpected config {name}")
        if isinstance(loaded, DatasetDict):
            if split is None:
                raise ValueError("split required for DatasetDict artifact")
            return loaded[split]
        return loaded

    with patch.object(mhf, "hf_hub_download", fake_hf_hub_download):
        with patch.object(mhf, "load_dataset", fake_load_dataset):
            ns = argparse.Namespace(
                dataset="mmeval_hf@local-disk",
                split=resolved_split,
                subset=subset_name,
                circular=False,
                resize=None,
                parallel_per_task=1,
                rank=0,
                template=None,
            )
            ds = MMEvalHFDataset(ns)
            ds.setup_parallel()
            print(f"[hf] global rows: {ds.global_length}")
            if ds.global_length == 0:
                print("[hf] EMPTY (no rows)")
                return 1
            for i in range(min(n, ds.global_length)):
                s = ds[i]
                msg = s["messages"][0]
                media_imgs = msg.get("media", [])
                sizes = [getattr(m, "size", "video") for m in media_imgs]
                print(f"  [{i}] id={s.get('id')} eval-id={s.get('eval-id')} media={sizes} "
                      f"prompt={msg['prompt'][:80]!r}")
    print("[hf] OK")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--simple-mmeval",
                   help="Path to a simple-mmeval checkout (so mmeval/ is importable). "
                        "Not required when using --audit.")
    p.add_argument("--local", help="Path to local artifact (with data.json + media/)")
    p.add_argument("--hf", help="Path to HF artifact (with hf_dataset/ + metadata.json)")
    p.add_argument("--split", default=None, help="Split name for on-disk DatasetDict (default: train for flat)")
    p.add_argument("--subset", default=None, help="Subset key inside metadata.json when multiple exist")
    p.add_argument("-n", type=int, default=2, help="Rows to print per artifact")
    p.add_argument("--audit", metavar="HF_DATASET_DIR",
                   help="Standalone audit mode (no simple-mmeval needed): checks metadata.json "
                        "schema (modalities, task_type, source.url), local-path leaks, deprecated "
                        "fields, and placeholder/media count for every row.")
    p.add_argument("--metadata", default=None,
                   help="metadata.json path for --audit (default: <HF_DATASET_DIR>/../metadata.json)")
    args = p.parse_args()

    if args.audit:
        hf_dir = Path(args.audit)
        meta_path = Path(args.metadata) if args.metadata else hf_dir.parent / "metadata.json"
        print(f"Auditing {hf_dir}  (metadata: {meta_path})")
        return audit_artifact(hf_dir, meta_path)

    if not args.local and not args.hf:
        print("ERROR: pass --local and/or --hf (or use --audit for standalone checks)", file=sys.stderr)
        return 2
    if not args.simple_mmeval:
        print("ERROR: --simple-mmeval is required for --local / --hf validation", file=sys.stderr)
        return 2
    smm = Path(args.simple_mmeval)
    rc = 0
    if args.local:
        rc |= validate_local(smm, Path(args.local), args.n)
    if args.hf:
        rc |= validate_hf(smm, Path(args.hf), args.split, args.subset, args.n)
    return rc


if __name__ == "__main__":
    sys.exit(main())
