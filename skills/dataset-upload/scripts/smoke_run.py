#!/usr/bin/env python3
"""End-to-end smoke test for a converted artifact through the real Simple-MMEval framework.

Sampling contract (applies to both modes)
-----------------------------------------

For EVERY (subset, split) the smoke samples ``--rows-per-split`` rows
**uniformly at random** via mmeval/run.py's own
``--sample_num / --sample_order random / --sample_seed`` flags — no
derived artifact, no row-selection shim. The default is 50; if a split
has fewer rows, mmeval caps to the split size and every row runs. The
smoke is green only when **every sampled row passes every fidelity
check** below.

Two modes
---------

  --local <dir>            Pre-push sanity check against the local artifact
                           (data.json + media/). Same sampler, same checks
                           as --hf, just one synthetic "split" (data.json).

  --hf <repo>              Post-push framework test against the pushed HF repo
                           via ``mmeval_hf@<repo>`` (+ ``--subset <name>``).
                           Authoritative gate: iterates EVERY (subset, split)
                           pair declared in the uploaded metadata.json (or one
                           subset when ``--subset`` is passed). Exercises the
                           actual MMEvalHFDataset loader, prompt rendering
                           from the uploaded metadata.json, and image bytes
                           round-trip through Arrow.

What gets exercised
-------------------

Both modes spawn ``mmeval/run.py`` exactly as a real eval would. mmeval's
own loader handles dataset/subset/split selection, prompt template
rendering, and media decode — so a non-zero exit catches any dataloader,
prompt-render, or media-load failure surfaced by the framework.

Outputs are persisted under ``<simple-mmeval>/.tmp/smoke_tests/<dataset-slug>/``
so they remain available for human inspection — ``cleanup.py`` skips the
``smoke_tests/`` tree unless ``--include-smoke-results`` is passed.

Post-run fidelity checks (per sampled row)
------------------------------------------

After each spawn, ``result.json`` is loaded and **every** row must satisfy:

  - ``id`` is present, non-empty, and unique within the result file.
  - ``messages[0].question`` (when present) is a string — surfaces source
    mapping bugs that overwrite question with a non-string.
  - ``messages[0].prompt`` is a non-empty string (proves the template
    rendered for this row).
  - The ``<image>`` / ``<video>`` placeholder count in the rendered prompt
    equals the sample-level ``media`` count (``res_handler`` strips
    per-message ``media`` but preserves the sample-level list; we count
    that — the most reliable signal that prompt and media list aligned).
  - When present, ``options`` is a dict, ``choices`` is a list, ``hint`` is
    a string, ``answer`` is a string or list (type drift here means the
    metadata.json mapping leaked a non-scalar through).
  - ``messages[1].role == "assistant"`` and its ``response`` is non-empty
    after flattening (mmeval may write str / list[str] / list[list[str]]).
  - Row count equals ``min(--rows-per-split, split_size)`` — partial
    result.json (e.g. silent post-load drop) is treated as a failure.

If anything fails the script exits non-zero, prints the offending row
ids, and leaves ``result.json`` + ``run.log`` + ``smoke_summary.json``
in the ``.tmp`` dir for follow-up.

GPU selection
-------------

By default, `nvidia-smi --query-gpu` is parsed and the GPU with the
lowest utilization that has enough free VRAM is bound to the child
process via `CUDA_VISIBLE_DEVICES`. Pass `--gpu N` to override, or
`--gpu -1` to disable GPU pinning entirely (e.g. for CPU-only smokes).

Usage
-----

    # post-push (recommended after push_to_hf.py succeeds):
    python3 smoke_run.py \\
        --simple-mmeval /path/to/simple-mmeval \\
        --hf <user>/<repo>                 \\
        --python "$MMEVAL_SMOKE_PYTHON"        # path to a python env with torch + transformers

    # pre-push (same 50/split contract as --hf, scoped to the local artifact):
    python3 smoke_run.py \\
        --simple-mmeval /path/to/simple-mmeval \\
        --local /path/to/converted/artifact    \\
        --python "$MMEVAL_SMOKE_PYTHON"

The --python flag also reads its default from the MMEVAL_SMOKE_PYTHON env
var. Falls back to `python3` on PATH if neither is provided. The interpreter
needs `torch`, `transformers` (recent enough for the chosen model series),
and `datasets`. No absolute paths baked into this script.

Override ``--rows-per-split N`` only when you specifically want a tighter
or looser sample (e.g. ``--rows-per-split 10`` for a quick local
iteration, or to match a known small split size).

Exits non-zero on any spawn failure, missing/empty result.json, or any
fidelity-check violation across any (subset, split).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"  # full HF repo id; mmeval/registry.py resolves the series from the final path component
# A smoke run samples this many rows per (subset, split). If a split has fewer
# rows, mmeval's --sample_num is capped at the split size and every row runs.
DEFAULT_ROWS_PER_SPLIT = 50
DEFAULT_SEED = 42
DEFAULT_MIN_FREE_MB = 8000


# ---------- GPU picker ----------------------------------------------------- #


def pick_gpu(min_free_mb: int = DEFAULT_MIN_FREE_MB) -> Optional[str]:
    """Return the index (as a string) of the least-utilized GPU with enough free VRAM.

    Returns None if nvidia-smi is unavailable or no GPU qualifies — caller
    decides whether to fall back to whatever the env already has set.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

    candidates: List[Tuple[int, int, int]] = []  # (util, free_mb, idx)
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            idx, free_mb, util = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        if free_mb >= min_free_mb:
            candidates.append((util, -free_mb, idx))  # prefer low util, then high free

    if not candidates:
        return None
    candidates.sort()
    chosen = candidates[0][2]
    print(f"[smoke] auto-picked GPU {chosen} (util={candidates[0][0]}%, "
          f"free={-candidates[0][1]} MiB; tried min_free={min_free_mb} MiB)")
    return str(chosen)


# ---------- HF discovery --------------------------------------------------- #


def _fetch_metadata_json(repo_id: str) -> Dict[str, Any]:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=repo_id, filename="metadata.json", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _discover_splits(repo_id: str) -> List[str]:
    from datasets import get_dataset_split_names
    try:
        return list(get_dataset_split_names(repo_id, "default"))
    except Exception:
        # Fall back to no-config call (older repos)
        return list(get_dataset_split_names(repo_id))


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


# ---------- Local artifact size probe -------------------------------------- #


def _count_local_rows(src: Path) -> int:
    """Return the row count in <src>/data.json without materializing a copy."""
    data = json.loads((src / "data.json").read_text())
    if not isinstance(data, list):
        raise SystemExit(f"ERROR: {src/'data.json'} is not a list")
    return len(data)


# ---------- Subprocess spawn ----------------------------------------------- #


def _spawn(simple_mmeval: Path, cmd: List[str], gpu: Optional[str], log_path: Path,
           python_bin: Optional[str] = None) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{simple_mmeval}:{env.get('PYTHONPATH', '')}"
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    # When --no-conda is in effect, mmeval/run.py spawns inference via bare
    # `python` (resolved from PATH); make sure that resolves to the same
    # interpreter we used as the orchestrator so torch/transformers come from
    # the right env.
    if python_bin:
        bin_dir = str(Path(python_bin).resolve().parent)
        env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    print(f"[smoke] cwd={simple_mmeval}")
    print(f"[smoke] cmd={' '.join(cmd)}")
    print(f"[smoke] gpu={gpu or '<unset, inheriting env>'} log={log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, cwd=str(simple_mmeval), env=env,
                              stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode


def _build_local_cmd(python_bin: str, src: Path, out_dir: Path, model: str,
                     attn_impl: str, passthrough: Path,
                     rows: int, seed: int, no_conda: bool) -> List[str]:
    cmd = [
        python_bin, "mmeval/run.py",
        "--model_name_or_path", model,
        "--dataset", "local@json",
        "--infile", str(src / "data.json"),
        "--img_dir", str(src / "media"),
        "--out_dir", str(out_dir),
        "--gpu_per_parallel", "1",
        "--parallel_per_task", "1",
        "--attn_implementation", attn_impl,
        "--template", str(passthrough),
        "--sample_num", str(rows),
        "--sample_order", "random",
        "--sample_seed", str(seed),
    ]
    if no_conda:
        cmd.append("--no_conda")
    return cmd


def _build_hf_cmd(python_bin: str, repo: str, subset: str, split: str,
                  out_dir: Path, model: str, attn_impl: str,
                  rows: int, seed: int, no_conda: bool,
                  template_override: Optional[Path] = None) -> List[str]:
    cmd = [
        python_bin, "mmeval/run.py",
        "--model_name_or_path", model,
        "--dataset", f"mmeval_hf@{repo}",
        "--subset", subset,
        "--split", split,
        "--out_dir", str(out_dir),
        "--gpu_per_parallel", "1",
        "--parallel_per_task", "1",
        "--attn_implementation", attn_impl,
        "--sample_num", str(rows),
        "--sample_order", "random",
        "--sample_seed", str(seed),
    ]
    if template_override is not None:
        cmd.extend(["--template", str(template_override)])
    if no_conda:
        cmd.append("--no_conda")
    return cmd


# ---------- Fidelity checks on result.json --------------------------------- #


def _flat_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return "".join(_flat_text(x) for x in v)
    return str(v)


_OPTIONAL_FIELD_TYPES = {
    # Fields that are optional but, when present, must match a strict type.
    "options": dict,
    "choices": list,
    "hint": str,
}
_ANSWER_OK = (str, list)

VIDEO_EXTS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv",
    ".mpeg", ".mpg", ".m4v", ".3gp", ".3g2", ".ts", ".mts", ".vob", ".gif",
}


def _is_video_media(m: Any) -> bool:
    """Check if a media reference is a video (by string extension or "Image Object" marker)."""
    if isinstance(m, str) and m != "Image Object":
        ext = os.path.splitext(m.split("?")[0])[-1].lower()
        return ext in VIDEO_EXTS
    return False


def _verify_result(result_json: Path, expected_rows: Optional[int]) -> List[str]:
    """Return a list of human-readable violations; empty list = all checks passed.

    Smoke validation contract:
      - result.json loads, is a non-empty list, and (when given) row count matches
        the expected sample size.
      - Per row: id is a non-empty scalar; messages[0] has a string ``question``,
        a non-empty rendered ``prompt`` whose ``<image>/<video>`` placeholder
        count equals ``len(media)``; optional ``options``/``choices``/``hint``,
        when present, are the right types; ``answer`` (when present) is a string
        or list.
      - For video rows: check that video media references have recognized extensions
        and are non-empty path strings.
      - messages[1] is the assistant message with a non-empty ``response``
        (str / list[str] / list[list[str]] all flatten).
    """
    violations: List[str] = []
    if not result_json.exists():
        return [f"result.json missing at {result_json}"]
    try:
        rows = json.loads(result_json.read_text())
    except json.JSONDecodeError as e:
        return [f"result.json invalid JSON: {e}"]

    if not isinstance(rows, list):
        return [f"result.json root is not a list (got {type(rows).__name__})"]
    if not rows:
        return ["result.json is an empty list (no rows survived inference)"]

    if expected_rows is not None and len(rows) != expected_rows:
        violations.append(f"expected {expected_rows} rows, got {len(rows)}")

    seen_ids: Dict[str, int] = {}
    video_stats = {"total_video_refs": 0, "video_rows": 0}
    for r in rows:
        rid = r.get("id")
        if rid is None or (isinstance(rid, str) and not rid.strip()):
            violations.append(f"row id missing/empty (eval-id={r.get('eval-id')})")
            rid = f"<no-id eval-id={r.get('eval-id')}>"
        else:
            key = str(rid)
            seen_ids[key] = seen_ids.get(key, 0) + 1

        messages = r.get("messages") or []
        if not messages:
            violations.append(f"id={rid}: messages[] empty")
            continue
        first = messages[0]
        if not isinstance(first, dict):
            violations.append(f"id={rid}: messages[0] is not a dict")
            continue
        question = first.get("question")
        if question is not None and not isinstance(question, str):
            violations.append(
                f"id={rid}: messages[0].question must be a string when present "
                f"(got {type(question).__name__})"
            )
        prompt = first.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            violations.append(f"id={rid}: messages[0].prompt empty or non-string")
            continue
        media_list = r.get("media") or []
        media_count = len(media_list)
        placeholder_count = len(re.findall(r"<(?:image|video)>", prompt))
        if placeholder_count != media_count:
            violations.append(
                f"id={rid}: <image|video> placeholders={placeholder_count} but "
                f"sample-level media count={media_count}"
            )

        # Video-specific checks on media references
        row_has_video = False
        for m in media_list:
            if _is_video_media(m):
                row_has_video = True
                video_stats["total_video_refs"] += 1
                if isinstance(m, str) and not m.strip():
                    violations.append(f"id={rid}: video media path is empty string")
        if row_has_video:
            video_stats["video_rows"] += 1
            video_phs = len(re.findall(r"<video>", prompt))
            video_media = sum(1 for m in media_list if _is_video_media(m))
            if video_phs != video_media:
                violations.append(
                    f"id={rid}: <video> placeholders={video_phs} but "
                    f"video media count={video_media}"
                )

        for key, expected_t in _OPTIONAL_FIELD_TYPES.items():
            if key in first and first[key] is not None and not isinstance(first[key], expected_t):
                violations.append(
                    f"id={rid}: messages[0].{key} must be {expected_t.__name__} "
                    f"when present (got {type(first[key]).__name__})"
                )
        if "answer" in first and first["answer"] is not None \
                and not isinstance(first["answer"], _ANSWER_OK):
            violations.append(
                f"id={rid}: messages[0].answer must be str or list when present "
                f"(got {type(first['answer']).__name__})"
            )

        # Response non-empty. res_handler.py writes the model output to
        # messages[1].response (the assistant message). Tolerate the older
        # top-level r["response"] shape as a fallback.
        asst = messages[1] if len(messages) >= 2 else {}
        resp = asst.get("response") if isinstance(asst, dict) else None
        if resp is None:
            resp = r.get("response")
        if not _flat_text(resp).strip():
            violations.append(f"id={rid}: response empty after flattening")
        if len(messages) >= 2 and isinstance(messages[1], dict) \
                and messages[1].get("role") != "assistant":
            violations.append(f"id={rid}: messages[1].role != 'assistant'")

    for dup_id, count in seen_ids.items():
        if count > 1:
            violations.append(f"id={dup_id!r} appears {count} times in result.json")

    if video_stats["video_rows"] > 0:
        print(f"[smoke] video stats: {video_stats['video_rows']} rows with video, "
              f"{video_stats['total_video_refs']} total video references")

    return violations


# ---------- Top-level mode runners ----------------------------------------- #


def _smoke_local(args, project_root: Path, model: str, attn_impl: str,
                 gpu: Optional[str], python_bin: str) -> int:
    src = Path(args.local).resolve()
    if not (src / "data.json").is_file():
        print(f"ERROR: {src/'data.json'} not found", file=sys.stderr)
        return 2
    if not (src / "media").is_dir():
        print(f"ERROR: {src/'media'} not found", file=sys.stderr)
        return 2

    requested = args.rows_per_split if args.rows_per_split is not None else DEFAULT_ROWS_PER_SPLIT
    total = _count_local_rows(src)
    rows = max(1, min(requested, total))
    seed = args.seed
    slug = _slug(src.name)
    smoke_root = project_root / ".tmp" / "smoke_tests" / slug / "local"
    out_dir = smoke_root / "out"
    work = smoke_root / "artifact"
    smoke_root.mkdir(parents=True, exist_ok=True)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    work.mkdir(parents=True, exist_ok=True)

    # Passthrough template forces LocalJSONDataset to re-emit the rendered
    # prompt the converter froze into messages[0].prompt — otherwise the
    # default template strips <image> placeholders and any post-prompt suffix.
    passthrough = work / "passthrough.j2"
    passthrough.write_text("{{ prompt }}")

    cmd = _build_local_cmd(python_bin, src, out_dir, model, attn_impl, passthrough,
                           rows, seed, args.no_conda)
    log = smoke_root / "run.log"
    rc = _spawn(Path(args.simple_mmeval).resolve(), cmd, gpu, log,
                python_bin if args.no_conda else None)
    summary_path = smoke_root / "smoke_summary.json"
    (smoke_root / "smoke_metadata.json").write_text(json.dumps(
        {"artifact": str(src), "rows_per_split": rows, "total_rows": total,
         "model": model, "seed": seed, "requested": requested},
        indent=2,
    ))
    if rc != 0:
        summary_path.write_text(json.dumps(
            {"rc": rc, "out_dir": str(out_dir), "violations": ["mmeval/run.py crashed"]},
            indent=2,
        ))
        print(f"[smoke][local] FAIL: mmeval/run.py exited {rc}; see {log}", file=sys.stderr)
        print(f"[smoke][local] artifacts preserved under {smoke_root}", file=sys.stderr)
        return rc

    violations = _verify_result(out_dir / "result.json", rows)
    summary_path.write_text(json.dumps(
        {"rc": rc, "out_dir": str(out_dir), "rows": rows,
         "violations": violations}, indent=2,
    ))
    if violations:
        print(f"[smoke][local] FAIL: {len(violations)} fidelity violations:", file=sys.stderr)
        for v in violations[:10]:
            print(f"  - {v}", file=sys.stderr)
        if len(violations) > 10:
            print(f"  ... and {len(violations)-10} more", file=sys.stderr)
        print(f"[smoke][local] artifacts preserved under {smoke_root}", file=sys.stderr)
        return 5
    print(f"[smoke][local] OK — {rows}/{total} rows passed all checks; "
          f"artifacts under {smoke_root}")
    return 0


def _smoke_hf(args, project_root: Path, model: str, attn_impl: str,
              gpu: Optional[str], python_bin: str) -> int:
    repo = args.hf
    only_subset = args.subset

    print(f"[smoke][hf] fetching metadata.json for {repo}")
    metadata = _fetch_metadata_json(repo)
    subsets_meta = metadata.get("subsets") or {}
    if not subsets_meta:
        print(f"ERROR: {repo}: metadata.json has no `subsets` block", file=sys.stderr)
        return 2
    if only_subset is not None:
        if only_subset not in subsets_meta:
            print(f"ERROR: subset {only_subset!r} not in metadata.json subsets "
                  f"({list(subsets_meta)})", file=sys.stderr)
            return 2
        subset_names = [only_subset]
    else:
        subset_names = list(subsets_meta)

    print(f"[smoke][hf] discovering splits for {repo}")
    splits = _discover_splits(repo)
    if not splits:
        print(f"ERROR: {repo}: no splits discovered", file=sys.stderr)
        return 2
    print(f"[smoke][hf] subsets={subset_names} splits={splits}")

    # Build (subset, split) pairs that match the metadata mapping.
    # Each subset's source.url declares which split names it owns;
    # iterating the cross product runs mismatched pairs through mmeval/run.py
    # that have no matching subset/template and fail spuriously.
    def _splits_for_subset(name: str) -> List[str]:
        blk = subsets_meta.get(name, {})
        url = ((blk.get("mapping_from_source") or {}).get("source") or {}).get("url") or {}
        if isinstance(url, dict):
            owned = [s for s in url if s in splits]
            if owned:
                return owned
        # Fallback: if exactly one subset, it owns every split.
        if len(subset_names) == 1:
            return splits
        return []

    pairs: List[Tuple[str, str]] = []
    covered: set[str] = set()
    for subset in subset_names:
        for split in _splits_for_subset(subset):
            pairs.append((subset, split))
            covered.add(split)
    # Surface unmatched splits (likely a missing source.url key).
    for sp in splits:
        if sp not in covered:
            print(f"[smoke][hf] WARN: split {sp!r} has no matching subset in "
                  f"metadata.json source.url; skipping", file=sys.stderr)

    requested = args.rows_per_split if args.rows_per_split is not None else DEFAULT_ROWS_PER_SPLIT
    seed = args.seed
    slug = _slug(repo)
    smoke_root = project_root / ".tmp" / "smoke_tests" / slug
    smoke_root.mkdir(parents=True, exist_ok=True)
    (smoke_root / "smoke_metadata.json").write_text(json.dumps(
        {"repo": repo, "subsets": subset_names, "splits": splits, "pairs": pairs,
         "model": model, "rows_per_split": requested, "seed": seed},
        indent=2,
    ))

    smm = Path(args.simple_mmeval).resolve()
    if not (smm / "mmeval" / "run.py").is_file():
        print(f"ERROR: {smm/'mmeval/run.py'} not found", file=sys.stderr)
        return 2

    summary: List[Dict[str, Any]] = []
    overall_rc = 0
    for subset, split in pairs:
            out_dir = smoke_root / subset / split
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True)
            log = out_dir / "run.log"
            # Optional per-(subset) template override — picks
            # `<override_dir>/<subset>/template.j2` if present, else
            # `<override_dir>/template.j2`. Falls back to the dataset's own
            # metadata.json template when nothing matches.
            template_override: Optional[Path] = None
            if getattr(args, "template_override_dir", None):
                base = Path(args.template_override_dir)
                cand = [base / subset / "template.j2", base / "template.j2"]
                for c in cand:
                    if c.is_file():
                        template_override = c
                        break
            cmd = _build_hf_cmd(python_bin, repo, subset, split, out_dir,
                                model, attn_impl, requested, seed, args.no_conda,
                                template_override=template_override)
            print(f"\n[smoke][hf] === {subset}/{split} ===")
            rc = _spawn(smm, cmd, gpu, log, python_bin if args.no_conda else None)
            entry: Dict[str, Any] = {"subset": subset, "split": split,
                                     "out_dir": str(out_dir), "rc": rc,
                                     "violations": []}
            if rc != 0:
                print(f"[smoke][hf] FAIL ({subset}/{split}): "
                      f"mmeval/run.py exited {rc}; see {log}", file=sys.stderr)
                overall_rc = rc
                summary.append(entry)
                continue
            # mmeval caps --sample_num at split size, so result row count is
            # min(requested, split_size). We learn the actual count from result.json
            # and require every row to pass — partial splits don't get a free pass.
            result_path = out_dir / "result.json"
            actual_rows = (
                len(json.loads(result_path.read_text())) if result_path.exists() else 0
            )
            violations = _verify_result(result_path, expected_rows=actual_rows)
            entry["violations"] = violations
            entry["rows"] = actual_rows
            entry["rows_requested"] = requested
            if violations:
                print(f"[smoke][hf] FAIL ({subset}/{split}): "
                      f"{len(violations)} fidelity violations", file=sys.stderr)
                for v in violations[:5]:
                    print(f"  - {v}", file=sys.stderr)
                if len(violations) > 5:
                    print(f"  ... and {len(violations)-5} more", file=sys.stderr)
                overall_rc = overall_rc or 5
            else:
                print(f"[smoke][hf] OK ({subset}/{split}): {actual_rows} rows passed")
            summary.append(entry)

    (smoke_root / "smoke_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[smoke][hf] summary written to {smoke_root/'smoke_summary.json'}")
    print(f"[smoke][hf] all results persisted under {smoke_root} (cleanup.py skips this dir)")
    return overall_rc


# ---------- main ----------------------------------------------------------- #


def _project_root_of(simple_mmeval: str) -> Path:
    """Return the simple-mmeval checkout root (used to anchor .tmp/smoke_tests/<slug>/)."""
    return Path(simple_mmeval).resolve()


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--simple-mmeval", required=True,
                   help="Path to a simple-mmeval checkout (used as cwd for mmeval/run.py "
                        "and as the root for .tmp/smoke_tests/<dataset>/ result storage)")
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--local",
                        help="Pre-push: path to local artifact dir (data.json + media/)")
    target.add_argument("--hf",
                        help="Post-push: HF repo id (default: iterate every subset "
                             "in metadata.json; use --subset to pick one)")
    p.add_argument("--subset", default=None,
                   help="Restrict --hf to a single subset in metadata.json "
                        "(default: iterate every subset)")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Model name (default {DEFAULT_MODEL}; must be in mmeval/registry.py)")
    p.add_argument("--rows-per-split", type=int, default=None,
                   help=f"Random samples per (subset, split) — default "
                        f"{DEFAULT_ROWS_PER_SPLIT}; capped at split size so smaller "
                        "splits run end-to-end. Every sampled row must pass the "
                        "fidelity checks for the smoke to be considered green.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"Random sampling seed (default {DEFAULT_SEED}; passed to "
                        "mmeval/run.py --sample_seed)")
    p.add_argument("--gpu", default=None,
                   help="CUDA_VISIBLE_DEVICES override (e.g. '0'). Default: auto-pick the "
                        "least-utilized GPU with >= --min-free-mb of free VRAM via nvidia-smi. "
                        "Pass '-1' to skip pinning entirely.")
    p.add_argument("--min-free-mb", type=int, default=DEFAULT_MIN_FREE_MB,
                   help=f"Free-VRAM threshold for auto GPU pick (default {DEFAULT_MIN_FREE_MB})")
    p.add_argument("--python", default=os.environ.get("MMEVAL_SMOKE_PYTHON", "python3"),
                   help="Python interpreter for mmeval/run.py (needs torch + datasets). "
                        "Default reads MMEVAL_SMOKE_PYTHON env var, else falls back to "
                        "`python3` on PATH. Per-model inference still dispatches to "
                        "ENV_DIR/<series>/ per mmeval/registry.py — this is only the "
                        "orchestrator interpreter.")
    p.add_argument("--attn-implementation", default="sdpa",
                   help="Attention backend (default sdpa; avoids the flash-attn interactive "
                        "prompt that hangs with no stdin attached)")
    p.add_argument("--no-conda", action="store_true", default=True,
                   help="Pass --no_conda True to mmeval/run.py so inference runs in the "
                        "orchestrator's python env instead of dispatching to the per-model "
                        "conda env named in registry.py. Default ON — registry env names "
                        "(e.g. 'qwen3_vl') often don't match what's installed locally; turn "
                        "off with --no-no-conda if you have the canonical env layout.")
    p.add_argument("--no-no-conda", dest="no_conda", action="store_false",
                   help="Disable the default --no-conda behavior.")
    p.add_argument("--template-override-dir", default=None,
                   help="Optional dir holding per-subset Jinja templates "
                        "(<dir>/<subset>/template.j2 or <dir>/template.j2). "
                        "When present, the matching file is passed to "
                        "mmeval/run.py as --template, taking priority over the "
                        "dataset's metadata.json prompt_template. Use this when "
                        "validating a corrected prompt before pushing it to the "
                        "Hub.")
    args = p.parse_args()

    # GPU resolution
    if args.gpu == "-1":
        gpu: Optional[str] = None
        print("[smoke] GPU pinning disabled (--gpu -1)")
    elif args.gpu:
        gpu = args.gpu
        print(f"[smoke] using requested GPU {gpu}")
    else:
        gpu = pick_gpu(args.min_free_mb)
        if gpu is None:
            print(f"[smoke] WARN: no GPU has >= {args.min_free_mb} MiB free; "
                  "child will inherit CUDA_VISIBLE_DEVICES from env")

    project_root = _project_root_of(args.simple_mmeval)

    if args.local:
        return _smoke_local(args, project_root, args.model, args.attn_implementation,
                            gpu, args.python)
    return _smoke_hf(args, project_root, args.model, args.attn_implementation,
                     gpu, args.python)


if __name__ == "__main__":
    sys.exit(main())
