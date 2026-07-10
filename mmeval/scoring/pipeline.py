import glob
import json
import os
import platform
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from mmeval.scoring.graders import run_grader
from mmeval.scoring.match import LLM_MATCHER_NAMES, MATCHER_REGISTRY
from mmeval.scoring.match.llm import _resolve_setting
from mmeval.scoring.protocol import ResolvedProtocol, extract_dataset_meta, resolve_protocol
from mmeval.scoring.report import build_summary
from mmeval.scoring.extraction import (
    extract_option_texts,
    extract_options,
    get_field,
    get_question,
    infer_question_type,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None


def discover_result_files(out_dir: str, pattern: str = "**/result.json") -> List[str]:
    if not out_dir:
        return []
    search_pattern = os.path.join(out_dir, pattern)
    return sorted(x for x in glob.glob(search_pattern, recursive=True) if os.path.isfile(x))


def build_matchers(args, matching_order: str):
    matcher_names = [x.strip().lower() for x in matching_order.split(",") if x.strip()]
    matchers = []
    for name in matcher_names:
        if name not in MATCHER_REGISTRY:
            raise ValueError(f"Unknown matcher `{name}` in --matching_order")
        matcher_cls = MATCHER_REGISTRY[name]
        if name in LLM_MATCHER_NAMES:
            matchers.append(matcher_cls(args))
        else:
            matchers.append(matcher_cls())
    return matchers


def _resume_fingerprint(args, proto: ResolvedProtocol) -> Dict[str, Any]:
    """The RESOLVED config subset that determines a sample's verdict (post
    CLI/dataset_meta/default precedence). Cached rows produced under a different
    fingerprint must not be reused — resuming a `template`-only run into a
    `template,llm-match` rerun would silently keep the old verdicts."""
    fingerprint = {
        "matching_order": proto.matching_order,
        "grader": proto.grader,
        "score_gt_field": args.score_gt_field,
        "score_pred_field": args.score_pred_field,
        "score_question_type": args.score_question_type,
        "score_numeric_rel_tol": proto.numeric_rel_tol,
        "score_numeric_abs_tol": proto.numeric_abs_tol,
        "score_string_match": proto.string_match,
        "score_anls_threshold": proto.anls_threshold,
    }
    # With an LLM matcher in the chain, the judge's identity determines verdicts
    # too — a gpt-4o-mini cache must not resume into a gpt-5 run. Rule-only chains
    # deliberately exclude these so a judge-config edit doesn't invalidate them.
    if any(name.strip().lower().startswith("llm") for name in proto.matching_order.split(",") if name.strip()):
        fingerprint["judge_provider"] = args.judge_provider
        fingerprint["judge_model"] = args.judge_model
        fingerprint["judge_temperature"] = args.judge_temperature
        # Both change verdicts: include_reason alters the judge prompt, and
        # max_tokens changes truncation/parse-failure behavior. Fingerprint the
        # RESOLVED max_tokens (CLI > env > default), like the matcher uses.
        fingerprint["judge_include_reason"] = bool(args.judge_include_reason)
        fingerprint["judge_max_tokens"] = _resolve_setting(
            getattr(args, "judge_max_tokens", None), "JUDGE_MAX_TOKENS", 2048)
    return fingerprint


def _provenance() -> Dict[str, Any]:
    """Environment identity embedded in score.json so any run can be reproduced.
    Deterministic on a given checkout (no timestamps — reruns stay byte-identical)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        ).stdout.strip() or None
    except Exception:
        commit = None
    try:
        from importlib.metadata import version
        openai_version = version("openai")
    except Exception:
        openai_version = None
    return {
        "git_commit": commit,
        "python": platform.python_version(),
        "openai": openai_version,
    }



def _load_cached_results(out_path: str, resume_enabled: bool, fingerprint: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if not resume_enabled:
        return {}

    cache_by_id: Dict[str, Dict[str, Any]] = {}
    tmp_path = f"{out_path}.tmp"
    # Union of both caches: the final score.json first, then the tmp overlays
    # it (fresher rows win). Both must pass the fingerprint check, so every
    # retained row was produced under the current verdict-determining config.
    for path in (out_path, tmp_path):
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                continue  # a bare-list tmp carries no config fingerprint; don't trust it
            config = payload.get("config") or {}
            if any(config.get(k) != v for k, v in fingerprint.items()):
                print(f"[score] ignoring resume cache {path}: scoring config changed", flush=True)
                continue
            rows = payload.get("samples")
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if "eval-id" not in row:
                    continue
                # Rows that failed because the judge API failed are not verdicts;
                # re-score them instead of freezing a transient outage into 0s.
                if str(row.get("reason", "")).startswith("llm_error"):
                    continue
                cache_by_id[str(row["eval-id"])] = row
        except Exception:
            continue
    return cache_by_id


def _flush_resume_tmp(tmp_path: str, samples: List[Dict[str, Any]], fingerprint: Dict[str, Any]) -> None:
    # Atomic write: a crash mid-flush must never leave a truncated tmp that a
    # resume would then have to discard (losing all progress since the last flush).
    part_path = f"{tmp_path}.part"
    with open(part_path, "w") as f:
        json.dump({"config": fingerprint, "samples": samples}, f, indent=2)
    os.replace(part_path, tmp_path)


def _score_single_sample(sample: Dict[str, Any], args, matchers, proto: ResolvedProtocol) -> Dict[str, Any]:
    gt = get_field(sample, args.score_gt_field, None)
    if (gt is None or (isinstance(gt, str) and not gt.strip())) and args.score_gt_field == "answer":
        # judge-graded datasets (ex-judge_scored) carry the reference in the
        # reserved `reference_response` field instead of `answer` (their
        # metadata mapping documents it) — generic fallback, never per-dataset.
        gt = get_field(sample, "reference_response", None)
    pred = get_field(sample, args.score_pred_field, None)
    question = get_question(sample)
    question_type = infer_question_type(sample, args.score_question_type, gt=gt)
    options = extract_options(sample) if question_type == "mcq" else []
    option_texts = extract_option_texts(sample, options)

    eval_id = sample.get("eval-id", sample.get("id"))
    output = {
        "eval-id": eval_id,
        "question_type": question_type,
        "gt": gt,
        "pred": pred,
        "matcher_used": None,
        "matched": None,
        "is_correct": 0,
        "status": "ok",
    }

    # Blank gt (answer-withheld test splits) must be invalid, not "confidently 0".
    if gt is None or pred is None or (isinstance(gt, str) and not gt.strip()):
        output["status"] = "invalid"
        output["reason"] = "missing_gt_or_pred"
        return output

    if proto.grader:
        # Dedicated fractional grader (vqa_accuracy / anls): per-sample `score`
        # in 0..1; is_correct = full credit only. summary.mean_score is the
        # official headline aggregate for these protocols.
        score, error = run_grader(proto.grader, pred, gt,
                                  {"anls_threshold": proto.anls_threshold})
        if score is None:
            output["status"] = "invalid"
            output["reason"] = error
            return output
        output["score"] = score
        output["is_correct"] = int(score >= 1.0)
        output["matcher_used"] = proto.grader
        return output

    context = {
        "question_type": question_type,
        "question": question,
        "pred": pred,
        "gt": gt,
        "options": options,
        "option_texts": option_texts,
        "numeric_rel_tol": proto.numeric_rel_tol,
        "numeric_abs_tol": proto.numeric_abs_tol,
        "string_match": proto.string_match,
        "anls_threshold": proto.anls_threshold,
    }

    trace = []
    for matcher in matchers:
        result = matcher.match(sample, context)
        if args.score_debug:
            trace.append(
                {
                    "matcher": matcher.name,
                    "is_match": result.is_match,
                    "matched": result.matched,
                    "reason": result.reason,
                    "meta": result.meta,
                }
            )
        if result.meta and "judge_prob" in result.meta:
            # Observability only (P of the judge's verdict token); never a verdict input.
            output["judge_prob"] = result.meta["judge_prob"]
        if result.is_match:
            output["matcher_used"] = matcher.name
            output["matched"] = result.matched
            output["is_correct"] = 1
            if result.reason is not None and args.judge_include_reason:
                output["reason"] = result.reason
            break
        if matcher.name.startswith("llm") and result.reason is not None:
            # LLM matcher reasons are either error markers ("llm_error: ...") or —
            # only when --judge_include_reason — the judge's explanation; both must
            # survive into the output (errors especially: see summary llm_errors).
            output["reason"] = result.reason

    if args.score_debug:
        output["trace"] = trace
    return output


def _score_sample_at(idx: int, sample: Dict[str, Any], args, matchers, proto: ResolvedProtocol):
    """Worker wrapper: score one sample, keep its position, default a missing eval-id."""
    scored = _score_single_sample(sample, args, matchers, proto)
    if scored["eval-id"] is None:
        scored["eval-id"] = sample.get("eval-id", sample.get("id", idx))
    return idx, scored


def _reject_old_layout(result_file: str, samples: List[Dict[str, Any]], gt_field: str) -> None:
    """The scorer reads grading fields ONLY from the sample TOP level. A
    message-embedded result keeps `answer` inside messages[0] with no
    top-level copy — fail with an actionable message instead of scoring
    everything as missing-gt."""
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        messages = sample.get("messages") or []
        msg = messages[0] if messages and isinstance(messages[0], dict) else {}
        if gt_field in msg and gt_field not in sample:
            raise ValueError(
                f"{result_file} uses the old message-embedded result.json layout: `{gt_field}` sits "
                f"inside messages[0] instead of at the sample top level. "
                f"Migrate it first:\n"
                f"    python3 scripts/migrate_result_layout.py {result_file}"
            )


def score_result_file(result_file: str, args) -> Dict[str, Any]:
    with open(result_file, "r") as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        raise ValueError(f"{result_file} must be a JSON list.")
    _reject_old_layout(result_file, samples, args.score_gt_field)

    # Metadata-driven protocol resolution (SCORE_TAXONOMY_HANDOFF.md): per knob,
    # explicit CLI flag > dataset_meta from result.json > built-in default.
    # Conflicting dataset_meta across samples or unsupported score_types raise here.
    dataset_meta = extract_dataset_meta(samples, result_file)
    proto = resolve_protocol(args, dataset_meta, result_file)

    out_path = os.path.join(os.path.dirname(result_file), args.score_output_name)
    tmp_path = f"{out_path}.tmp"
    fingerprint = _resume_fingerprint(args, proto)
    cached_by_id = _load_cached_results(out_path=out_path, resume_enabled=args.score_resume, fingerprint=fingerprint)
    matchers = build_matchers(args, proto.matching_order) if not proto.grader else []

    scored_samples: List[Dict[str, Any]] = [None] * len(samples)
    resumed_count = 0
    pbar = None
    if args.score_progress_bar and tqdm is not None:
        pbar = tqdm(
            total=len(samples),
            desc=f"Scoring {os.path.basename(os.path.dirname(result_file))}",
            leave=False,
        )

    # Samples whose normalized eval-id is not unique (true duplicates, or an int/str
    # collision like 1 vs "1") cannot be safely resumed from cache: one cached row
    # would stand in for several distinct samples. Always re-score those.
    id_counts = Counter(
        str(s.get("eval-id", s.get("id", i))) if isinstance(s, dict) else str(i)
        for i, s in enumerate(samples)
    )
    dup_ids = {k for k, v in id_counts.items() if v > 1}
    if dup_ids and cached_by_id:
        print(f"[score] {len(dup_ids)} non-unique eval-id(s) in {result_file}; those samples will be re-scored, not resumed", flush=True)

    uncached_samples = []
    for idx, sample in enumerate(samples):
        eval_id = str(sample.get("eval-id", sample.get("id", idx)))
        cached = cached_by_id.get(eval_id) if eval_id not in dup_ids else None
        if cached is not None:
            scored_samples[idx] = cached
            resumed_count += 1
            if pbar is not None:
                pbar.update(1)
            continue
        uncached_samples.append((idx, sample))

    # Threads only help when samples block on API calls (llm matchers); for
    # pure-rule chains and local graders the GIL makes them a net slowdown
    # (measured ~3x on 10k samples). Default to serial there; an explicit
    # --parallel_per_task always wins.
    uses_api = any(n.strip().startswith("llm") for n in proto.matching_order.split(",") if n.strip())
    if "parallel_per_task" in getattr(args, "_explicit_flags", frozenset()) or uses_api:
        sample_workers = max(1, int(getattr(args, "parallel_per_task", 1)))
    else:
        sample_workers = 1
    processed_new = 0

    if sample_workers <= 1 or len(uncached_samples) <= 1:
        for idx, sample in uncached_samples:
            i, scored = _score_sample_at(idx, sample, args, matchers, proto)
            scored_samples[i] = scored
            processed_new += 1
            if pbar is not None:
                pbar.update(1)
            if args.score_resume and processed_new % max(1, int(args.score_save_freq)) == 0:
                _flush_resume_tmp(tmp_path, [x for x in scored_samples if x is not None], fingerprint)
    else:
        with ThreadPoolExecutor(max_workers=sample_workers) as executor:
            futures = [
                executor.submit(_score_sample_at, idx, sample, args, matchers, proto)
                for idx, sample in uncached_samples
            ]
            for future in as_completed(futures):
                i, scored = future.result()
                scored_samples[i] = scored
                processed_new += 1
                if pbar is not None:
                    pbar.update(1)
                if args.score_resume and processed_new % max(1, int(args.score_save_freq)) == 0:
                    _flush_resume_tmp(tmp_path, [x for x in scored_samples if x is not None], fingerprint)

    if pbar is not None:
        pbar.close()

    if any(x is None for x in scored_samples):
        raise RuntimeError(f"Incomplete scoring results found for {result_file}")

    summary = build_summary(scored_samples)

    payload = {
        "summary": summary,
        "config": {
            # RESOLVED per-knob values (CLI > dataset_meta > default); which
            # source decided each knob is recorded in knob_sources. The resume
            # fingerprint is embedded verbatim: _load_cached_results validates
            # cached rows against this block, so it must stay a superset of
            # every fingerprint key.
            **fingerprint,
            "score_type": proto.score_type,
            "task_type": proto.task_type,
            "official_protocol": proto.official_protocol,
            "knob_sources": proto.knob_sources,
            "judge_provider": args.judge_provider,
            "judge_model": args.judge_model,
            "judge_temperature": args.judge_temperature,
            "judge_include_reason": args.judge_include_reason,
            "score_progress_bar": args.score_progress_bar,
            "score_resume": args.score_resume,
            "score_save_freq": args.score_save_freq,
            "parallel_per_task": sample_workers,
            "resumed_count": resumed_count,
            "provenance": _provenance(),
            # List of notes: the scorer-wide caveat, rubric degradations, the
            # none-stamp, and score_protocol.note verbatim (score_note).
            "protocol_notes": proto.protocol_notes,
        },
        "samples": scored_samples,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    # A finished score.json supersedes any resume tmp regardless of how this
    # run was configured — a stale tmp left behind would overlay OLDER rows
    # onto a fresher final in a later resume's cache union.
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    return {
        "result_file": result_file,
        "score_file": out_path,
        "summary": summary,
        "resumed_count": resumed_count,
    }

