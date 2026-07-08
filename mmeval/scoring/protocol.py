"""score_type -> execution resolution (the metadata-driven scoring contract).

One tested place for the SCORE_TAXONOMY_HANDOFF.md mapping table: every sample
may carry a flat top-level `dataset_meta` block (injected at inference time by
the mmeval_hf loader, persisted in result.json); the scorer resolves each knob
with the precedence  explicit CLI flag > dataset_meta > built-in default,
records which source decided every knob, and stamps official-protocol status.
"""
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# The scorer-wide caveat recorded into every score.json (first protocol note).
GLOBAL_PROTOCOL_NOTE = (
    "Per-sample scoring; official aggregations noted per dataset in these notes "
    "are not reproduced. LLM failures are marked wrong and counted in "
    "summary.llm_errors (never randomly guessed, unlike VLMEvalKit)."
)

NONE_PROTOCOL_NOTE = (
    "score_type=none: no official eval protocol exists for this dataset; "
    "numbers are mm-eval-convention, not official benchmark results."
)

# Direct chain mappings (150/168 published subsets).
SCORE_TYPE_CHAINS = {
    "rule": "exact,template",
    "llm_extract": "template,llm-match",
    "rule_llm_judge": "exact,template,llm-judge",
    "llm_judge": "llm-judge",
    "none": "exact,template",
}

# Dedicated fractional graders implemented in mmeval/scoring/graders.py.
GRADER_SCORE_TYPES = {"vqa_accuracy", "anls"}

# Protocols the generic scorer cannot faithfully execute — refuse explicitly,
# never silently mis-grade (SCORE_TAXONOMY_HANDOFF.md mapping table).
UNSUPPORTED_SCORE_TYPES = {
    "caption_metrics": (
        "unsupported: requires corpus-level caption metrics "
        "(CIDEr/BLEU/ROUGE/BERTScore); use the benchmark's own evaluator"
    ),
    "execution": "unsupported: requires official test-case execution (pass@1)",
    "per_task_metric": (
        "unsupported: requires the benchmark's own evaluator "
        "(per-row metric dispatch)"
    ),
}

_KNOWN_SCORE_TYPES = set(SCORE_TYPE_CHAINS) | GRADER_SCORE_TYPES | set(UNSUPPORTED_SCORE_TYPES)

_STRING_MATCH_VALUES = ("exact", "contains", "anls")


@dataclass
class ResolvedProtocol:
    """Effective per-file scoring configuration after precedence resolution."""
    matching_order: str
    grader: Optional[str]                    # vqa_accuracy | anls | None
    numeric_rel_tol: float
    numeric_abs_tol: float
    string_match: str                        # exact | contains | anls
    anls_threshold: float
    score_type: Optional[str]                # from dataset_meta, if any
    task_type: Optional[str]
    official_protocol: Optional[bool]        # None = unknown (no dataset_meta)
    protocol_notes: List[str] = field(default_factory=list)
    knob_sources: Dict[str, str] = field(default_factory=dict)  # knob -> cli|dataset_meta|default


def extract_dataset_meta(samples: List[Dict[str, Any]], result_file: str) -> Optional[Dict[str, Any]]:
    """The single dataset_meta block of a result file, or None. Conflicting
    blocks across samples mean a hand-merged file — refuse, never pick one."""
    distinct: Dict[str, Dict[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        meta = sample.get("dataset_meta")
        if isinstance(meta, dict) and meta:
            distinct[json.dumps(meta, sort_keys=True)] = meta
    if len(distinct) > 1:
        keys = "\n  ".join(sorted(distinct))
        raise ValueError(
            f"{result_file} carries {len(distinct)} conflicting dataset_meta blocks "
            f"(hand-merged file?). Score each source file separately, or strip the "
            f"blocks and pass explicit CLI flags.\n  {keys}"
        )
    return next(iter(distinct.values()), None)


def _explicit(args) -> frozenset:
    return getattr(args, "_explicit_flags", frozenset())


def resolve_protocol(args, dataset_meta: Optional[Dict[str, Any]], result_file: str) -> ResolvedProtocol:
    """Apply the per-knob precedence CLI > dataset_meta > default and the
    score_type mapping table. Raises ValueError for unsupported protocols
    (unless the user explicitly forced a chain via --matching_order)."""
    explicit = _explicit(args)
    meta = dataset_meta or {}
    score_type = meta.get("score_type")
    params = meta.get("score_params") or {}
    if score_type is not None and score_type not in _KNOWN_SCORE_TYPES:
        raise ValueError(
            f"{result_file}: unknown score_type {score_type!r} in dataset_meta; "
            f"known: {sorted(_KNOWN_SCORE_TYPES)}"
        )

    notes: List[str] = [GLOBAL_PROTOCOL_NOTE]
    sources: Dict[str, str] = {}

    # --- matching_order / grader -------------------------------------------
    grader = None
    if "matching_order" in explicit:
        matching_order = args.matching_order
        sources["matching_order"] = "cli"
        if score_type in GRADER_SCORE_TYPES or score_type in UNSUPPORTED_SCORE_TYPES:
            notes.append(
                f"explicit --matching_order overrides score_type={score_type}: the "
                f"official protocol is NOT being executed; numbers are not comparable "
                f"to official results."
            )
    elif score_type in SCORE_TYPE_CHAINS:
        matching_order = SCORE_TYPE_CHAINS[score_type]
        sources["matching_order"] = "dataset_meta"
    elif score_type in GRADER_SCORE_TYPES:
        matching_order = ""
        grader = score_type
        sources["matching_order"] = "dataset_meta"
    elif score_type in UNSUPPORTED_SCORE_TYPES:
        raise ValueError(
            f"{result_file}: score_type={score_type} is {UNSUPPORTED_SCORE_TYPES[score_type]}. "
            f"Pass an explicit --matching_order to force approximate chain scoring "
            f"(numbers will NOT be official)."
        )
    else:  # no dataset_meta / no score_type
        matching_order = args.matching_order
        sources["matching_order"] = "default"

    # llm_extract compare=keyword_set_f1 (Mementos) is its own protocol.
    if score_type == "llm_extract" and params.get("compare") == "keyword_set_f1" \
            and "matching_order" not in explicit:
        raise ValueError(
            f"{result_file}: score_type=llm_extract with compare=keyword_set_f1 is "
            f"unsupported: requires keyword extraction + set-F1 grading (Mementos); "
            f"use the benchmark's own evaluator."
        )

    # --- numeric / string knobs (CLI > score_params > default) --------------
    def _knob(cli_name: str, param_name: str, default):
        if cli_name in explicit:
            sources[cli_name] = "cli"
            return getattr(args, cli_name)
        if param_name in params:
            sources[cli_name] = "dataset_meta"
            return params[param_name]
        sources[cli_name] = "default"
        return default

    numeric_rel_tol = float(_knob("score_numeric_rel_tol", "numeric_rel_tol", args.score_numeric_rel_tol))
    numeric_abs_tol = float(_knob("score_numeric_abs_tol", "numeric_abs_tol", args.score_numeric_abs_tol))
    string_match = str(_knob("score_string_match", "string_match", args.score_string_match))
    if string_match not in _STRING_MATCH_VALUES:
        raise ValueError(f"{result_file}: invalid string_match {string_match!r}; allowed: {_STRING_MATCH_VALUES}")
    # anls grader takes `threshold`; the rule-chain composite takes `anls_threshold`.
    param_thresh = params.get("threshold", params.get("anls_threshold"))
    if "score_anls_threshold" in explicit:
        anls_threshold = float(args.score_anls_threshold)
        sources["score_anls_threshold"] = "cli"
    elif param_thresh is not None:
        anls_threshold = float(param_thresh)
        sources["score_anls_threshold"] = "dataset_meta"
    else:
        anls_threshold = float(args.score_anls_threshold)
        sources["score_anls_threshold"] = "default"

    # --- judge rubric degradation note --------------------------------------
    rubric = params.get("rubric")
    if rubric and rubric != "binary":
        notes.append(
            f"official rubric `{rubric}`"
            + (f" scale={params['scale']}" if params.get("scale") else "")
            + " degraded to a binary llm-judge verdict; official numbers aggregate "
              "the fractional/scaled judge scores and are not directly comparable."
        )

    # --- official-protocol stamp + score_note -------------------------------
    if score_type is None:
        official = None
    elif score_type == "none":
        official = False
        notes.append(NONE_PROTOCOL_NOTE)
    else:
        official = True
    score_note = (meta.get("score_note") or "").strip()
    if score_note:
        notes.append(score_note)  # verbatim, per the handoff contract

    return ResolvedProtocol(
        matching_order=matching_order,
        grader=grader,
        numeric_rel_tol=numeric_rel_tol,
        numeric_abs_tol=numeric_abs_tol,
        string_match=string_match,
        anls_threshold=anls_threshold,
        score_type=score_type,
        task_type=meta.get("task_type"),
        official_protocol=official,
        protocol_notes=notes,
        knob_sources=sources,
    )
