"""Batch inference: run many (model x dataset) combinations in one command.

This is a thin scheduler on top of ``mmeval/run.py``. For each selected model it
estimates peak VRAM, decides how many GPUs a shard needs and how many shards can
run in parallel, then invokes ``mmeval/run.py`` for every dataset. Model metadata
(used for VRAM sizing and filtering) lives in ``mmeval/model_metadata.json``; the
model -> series/env resolution stays in ``mmeval/registry.py`` and is handled by
``run.py`` itself.

Examples:
    # Two models on one HF benchmark, auto GPU allocation
    python mmeval/batch_infer.py \\
        --models Qwen2.5-VL-7B-Instruct InternVL3-8B \\
        --datasets mmeval_hf@mm-eval/MMBench-en-V11:test \\
        --out-dir work_dirs/batch

    # A whole series over several datasets, forwarding gen kwargs to run.py
    python mmeval/batch_infer.py \\
        --model-series qwenvl2d5 \\
        --datasets evalkit@LLaVABench local@/data/q.json:/data/imgs \\
        --out-dir work_dirs/batch --max_new_tokens 256
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from mmeval.utils.gpu import (
    estimate_max_tokens,
    estimate_vram_gb,
    enabled_gpu_indices,
    get_gpu_indices,
    gpu_allocation,
    min_gpu_vram,
    parse_gpu_memory,
)

BASE_DIR = Path(__file__).resolve().parent.parent
METADATA_JSON = Path(__file__).with_name("model_metadata.json")
RUN_SCRIPT = Path(__file__).with_name("run.py")

# run.py flags that batch_infer sets per task; users must not pass them through.
MANAGED_FLAGS = {
    "model_name_or_path", "dataset", "split", "infile", "img_dir",
    "out_dir", "gpu_per_parallel", "parallel_per_task", "rank",
}


# --------------------------------------------------------------------------- #
# Model metadata
# --------------------------------------------------------------------------- #
def load_models() -> list[dict]:
    """Load the metadata JSON as a list of model dicts (model name folded in)."""
    with METADATA_JSON.open(encoding="utf-8") as f:
        data = json.load(f)
    return [{"model_name": name, **fields} for name, fields in data.items()]


def model_vram_gb(model: dict) -> float:
    """Estimated peak VRAM (GiB) for one shard; API models need none."""
    if model.get("api_model"):
        return 0.0
    return estimate_vram_gb(model, estimate_max_tokens(model))


def filter_models(models: list[dict], args) -> list[dict]:
    """Apply --model-type / --quantization / --api / --modalities filters."""
    result = []
    for m in models:
        if args.model_type == "non-pretrain":
            if m["model_type"] == "pretrain":
                continue
        elif args.model_type != "all" and m["model_type"] != args.model_type:
            continue
        if args.quantization != "all" and m["quantization"] != (args.quantization == "true"):
            continue
        if args.api != "all" and m["api_model"] != (args.api == "true"):
            continue
        if args.modalities and not set(m["modalities"]).intersection(args.modalities):
            continue
        result.append(m)
    return result


def sort_models(models: list[dict], key: str | None) -> list[dict]:
    """Order models for scheduling. None keeps registry order."""
    if key in ("newest", "oldest"):
        return sorted(models, key=lambda m: m.get("created_at") or "0", reverse=(key == "newest"))
    if key in ("largest", "smallest"):
        def params(m):
            return m.get("parameter_count") or 0
        return sorted(models, key=params, reverse=(key == "largest"))
    return models


def select_models(args, all_models: list[dict]) -> list[dict]:
    """Resolve --models / --model-series into a concrete model list, then filter."""
    by_name = {m["model_name"]: m for m in all_models}

    if args.models:
        if [s.lower() for s in args.models] == ["all"]:
            selected = list(all_models)
        else:
            unknown = [n for n in args.models if n not in by_name]
            if unknown:
                sys.exit(f"Unknown model(s): {', '.join(unknown)} "
                         f"(use --list-models to inspect the registry).")
            selected = [by_name[n] for n in args.models]
    else:
        if [s.lower() for s in args.model_series] == ["all"]:
            selected = list(all_models)
        else:
            wanted = set(args.model_series)
            unknown = sorted(wanted.difference(m["series"] for m in all_models))
            if unknown:
                sys.exit(f"Unknown model series: {', '.join(unknown)}")
            selected = [m for m in all_models if m["series"] in wanted]

    filtered = sort_models(filter_models(selected, args), args.sort_by)
    if not filtered and selected:
        print(f"Note: all {len(selected)} selected model(s) were removed by filters "
              f"(--model-type={args.model_type} --quantization={args.quantization} --api={args.api}"
              + (f" --modalities={' '.join(args.modalities)}" if args.modalities else "") + ").")
    return filtered


# --------------------------------------------------------------------------- #
# Dataset specs
# --------------------------------------------------------------------------- #
def parse_dataset_spec(spec: str) -> tuple[list[str], str]:
    """Translate a dataset spec into (run.py args, short label).

    Supported forms:
        mmeval_hf@org/name:split   HuggingFace mm-eval dataset (split required)
        evalkit@Name               VLMEvalKit dataset by name
        local@/path/file.json[:/img_dir]   local JSON, optional image dir
        tsv@name-or-path-or-url    raw --dataset value (.tsv name, path, or URL)
    """
    spec = spec.strip()
    dtype, sep, rest = spec.partition("@")
    if not sep or not rest:
        sys.exit(f"Invalid dataset spec '{spec}'. Expected '<type>@<value>', "
                 f"e.g. mmeval_hf@mm-eval/MMBench-en-V11:test")

    if dtype == "mmeval_hf":
        if ":" not in rest:
            sys.exit(f"mmeval_hf spec needs a ':<split>' suffix, got '{spec}' "
                     f"(e.g. mmeval_hf@mm-eval/MMBench-en-V11:test).")
        path, split = rest.rsplit(":", 1)
        return ["--dataset", f"mmeval_hf@{path}", "--split", split], f"{path.split('/')[-1]}_{split}"

    if dtype == "evalkit":
        return ["--dataset", f"evalkit@{rest}"], rest

    if dtype == "local":
        if ":" in rest:
            infile, img_dir = rest.rsplit(":", 1)
            return ["--dataset", "local@json", "--infile", infile, "--img_dir", img_dir], Path(infile).stem
        return ["--dataset", "local@json", "--infile", rest], Path(rest).stem

    if dtype == "tsv":
        return ["--dataset", rest], Path(rest).stem

    sys.exit(f"Unknown dataset type '{dtype}' in '{spec}'. "
             f"Use one of: mmeval_hf, evalkit, local, tsv.")


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def run_task(model: dict, ds_args: list[str], label: str, out_base: Path,
             enabled_gpus: list[str], gpu_vram_gb: float, passthrough: list[str],
             api_parallel: int) -> dict:
    """Run one (model, dataset) task via mmeval/run.py."""
    name = model["model_name"]
    is_api = bool(model.get("api_model"))

    if is_api:
        vram, alloc = 0.0, {"can_run": True, "gpu_per_parallel": 0,
                            "parallel_per_task": max(1, api_parallel)}
    else:
        vram = model_vram_gb(model)
        alloc = gpu_allocation(len(enabled_gpus), vram, gpu_vram_gb)

    base = {"model": name, "series": model["series"], "dataset": label,
            "estimated_vram_gb": round(vram, 1)}
    if not alloc["can_run"]:
        print(f"  skipped: {alloc['reason']}")
        return {**base, "status": "skipped", "reason": alloc["reason"], "duration": 0.0}

    out_dir = out_base / label / name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(RUN_SCRIPT),
        "--model_name_or_path", model.get("hf_path") or name,
        *ds_args,
        "--out_dir", str(out_dir),
        "--gpu_per_parallel", str(alloc["gpu_per_parallel"]),
        "--parallel_per_task", str(alloc["parallel_per_task"]),
        *passthrough,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(BASE_DIR), env.get("PYTHONPATH", "")]))
    # Constrain run.py to exactly the GPUs we accounted for (honors --gpu-memory).
    if not is_api:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(enabled_gpus)

    start = time.time()
    status = "error"
    try:
        status = "success" if subprocess.run(cmd, cwd=str(BASE_DIR), env=env).returncode == 0 else "failed"
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  error: {exc}")
    duration = time.time() - start
    print(f"  {status} ({duration:.0f}s)")
    return {**base, "status": status,
            "gpu_per_parallel": alloc["gpu_per_parallel"],
            "parallel_per_task": alloc["parallel_per_task"],
            "duration": round(duration, 1)}


def list_models(models: list[dict]) -> None:
    """Print matching models grouped by series."""
    by_series: dict[str, list[dict]] = {}
    for m in models:
        by_series.setdefault(m["series"], []).append(m)
    print(f"\nMatching models: {len(models)}\n")
    for series in sorted(by_series):
        print(f"{series}:")
        for m in by_series[series]:
            params = m.get("parameter_count") or 0
            p_str = f"{params / 1e9:.1f}B" if params > 0 else ""
            quant = "Q" if m.get("quantization") else " "
            api = "API" if m.get("api_model") else "   "
            print(f"  {m['model_name']:<48} {model_vram_gb(m):>4.0f}GB {m.get('model_type', '?'):<10} "
                  f"{quant} {api} {p_str}")
    print()


def print_summary(results: list[dict]) -> None:
    """Print a one-line tally plus details for non-successful tasks."""
    counts = Counter(r["status"] for r in results)
    total_time = sum(r.get("duration", 0) for r in results)
    print(f"\n{'=' * 60}")
    print(f"Batch inference summary: {len(results)} task(s)  "
          f"✅ {counts['success']}  ❌ {counts['failed']}  "
          f"⏭️ {counts['skipped']}  💥 {counts['error']}  "
          f"({total_time:.0f}s / {total_time / 60:.1f}m)")
    for r in results:
        if r["status"] in ("failed", "error"):
            print(f"  ❌ {r['model']} ({r['dataset']}): {r['status']}")
        elif r["status"] == "skipped":
            print(f"  ⏭️ {r['model']} ({r['dataset']}): {r.get('reason', '')}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="batch_infer",
        description="Run inference over many (model x dataset) combinations, "
                    "scheduling GPUs automatically from estimated VRAM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Disable prefix matching so run.py flags (e.g. --dataset, a prefix of
        # --datasets) fall through to passthrough instead of being swallowed.
        allow_abbrev=False,
    )
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--models", nargs="+", metavar="NAME",
                     help="Model name(s) from the registry, or 'all'. "
                          "Mutually exclusive with --model-series.")
    sel.add_argument("--model-series", nargs="+", metavar="SERIES",
                     help="Model series name(s), or 'all'. "
                          "Mutually exclusive with --models.")
    p.add_argument("--datasets", nargs="+", metavar="SPEC",
                   help="Dataset spec(s): mmeval_hf@org/name:split | evalkit@Name | "
                        "local@/path.json[:/img_dir] | tsv@name-or-path-or-url.")
    p.add_argument("--out-dir", "--out_dir", dest="out_dir", metavar="DIR",
                   help="Base output directory; results go to <out-dir>/<dataset>/<model>/.")
    p.add_argument("--model-type", default="non-pretrain",
                   choices=["pretrain", "instruct", "reasoning", "non-pretrain", "all"],
                   help="Filter models by type (default: non-pretrain).")
    p.add_argument("--quantization", default="false", choices=["true", "false", "all"],
                   help="Filter by quantization (default: false).")
    p.add_argument("--api", default="false", choices=["true", "false", "all"],
                   help="Select API vs local models (default: false = local only).")
    p.add_argument("--modalities", nargs="+", metavar="MODALITY",
                   help="Keep models supporting at least one of these modalities.")
    p.add_argument("--sort-by", choices=["newest", "oldest", "largest", "smallest"],
                   help="Order models for scheduling (default: registry order).")
    p.add_argument("--gpu-memory", nargs="+", type=float, metavar="GIB",
                   help="Per-GPU VRAM in GiB, in visible-GPU order; 0 disables a GPU. "
                        "Default: auto-detected. Example: --gpu-memory 80 80 0 40")
    p.add_argument("--api-parallel", type=int, default=1, metavar="N",
                   help="Concurrent shards for API models (no GPU needed; default: 1).")
    p.add_argument("--list-models", action="store_true",
                   help="Print the resolved model list and exit.")
    p.add_argument("--save-log", action="store_true",
                   help="Write a JSON run log (args + per-task results) into --out-dir.")
    return p


def main() -> int:
    args, passthrough = build_parser().parse_known_args()

    bad = [tok for tok in passthrough if tok.startswith("--")
           and tok.lstrip("-").split("=", 1)[0].replace("-", "_") in MANAGED_FLAGS]
    if bad:
        sys.exit(f"These flags are managed by batch_infer and cannot be passed through: "
                 f"{', '.join(sorted(set(bad)))}.")

    all_models = load_models()
    if not (args.models or args.model_series):
        if args.list_models:
            args.models = ["all"]
        else:
            sys.exit("Select models with --models or --model-series "
                     "(or use --list-models to inspect the registry).")

    models = select_models(args, all_models)
    if not models:
        sys.exit("No models match the given selection/filters.")

    if args.list_models:
        list_models(models)
        return 0

    if not args.datasets:
        sys.exit("--datasets is required (e.g. --datasets mmeval_hf@mm-eval/MMBench-en-V11:test).")
    if not args.out_dir:
        sys.exit("--out-dir is required.")

    datasets = [parse_dataset_spec(s) for s in args.datasets]
    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    visible = get_gpu_indices()
    gpu_memory = parse_gpu_memory(args.gpu_memory, len(visible))
    enabled = enabled_gpu_indices(visible, gpu_memory)
    vram_gb = min_gpu_vram(gpu_memory)

    tasks = [(m, ds_args, label) for ds_args, label in datasets for m in models]
    print(f"Batch inference: {len(tasks)} task(s) — {len(models)} model(s) x {len(datasets)} dataset(s)")
    if gpu_memory:
        mem = ", ".join(f"GPU{i}:{v:.0f}GiB" for i, v in enumerate(gpu_memory))
        print(f"GPUs: {len(enabled)}/{len(visible)} enabled (min VRAM {vram_gb:.0f}GiB) [{mem}]")
    else:
        print("GPUs: none visible (API models only)")
    print(f"Output: {out_base}\n")

    results = []
    for i, (model, ds_args, label) in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {model['model_name']} ({label})")
        results.append(run_task(model, ds_args, label, out_base, enabled,
                                vram_gb, passthrough, args.api_parallel))
        print()

    print_summary(results)

    if args.save_log:
        log_path = out_base / f"batch_log_{datetime.now():%Y%m%d_%H%M%S}.json"
        with log_path.open("w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "passthrough": passthrough, "results": results}, f, indent=2)
        print(f"\nLog saved to {log_path}")

    failed = sum(r["status"] in ("failed", "error") for r in results)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
