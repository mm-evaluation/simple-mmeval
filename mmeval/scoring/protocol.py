"""score_pipeline -> execution resolution (the metadata-driven scoring contract).

Every sample may carry a flat top-level `dataset_meta` block (injected at
inference time by the mmeval_hf loader, persisted in result.json). Its
`score_pipeline` declares the dataset's official scoring protocol as an
ORDERED LIST OF ATOMIC STAGE NAMES (see mmeval/scoring/stages/):

    "score_pipeline": ["exact-match", "rule-match", "llm-judge"]

Two non-list forms are part of the contract:
- []                                  — the dataset explicitly declares that no
      official protocol exists; the scorer runs the default pipeline and stamps
      the output non-official.
- {"unsupported": "<protocol>",       — the official protocol cannot be executed
   "reason": "<why / official tool>"}   by this scorer; scoring is REFUSED with
      an actionable error (an explicit --score_pipeline forces approximate scoring,
      stamped non-official).

The scorer resolves each knob with the precedence
    explicit CLI flag > dataset_meta > built-in default,
records which source decided every knob (knob_sources), and stamps
official-protocol status.
"""
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mmeval.scoring.stages import STAGE_REGISTRY, GRADER_STAGE_NAMES

# The scorer-wide caveat recorded into every score.json (first protocol note).
GLOBAL_PROTOCOL_NOTE = (
    "Per-sample scoring; official aggregations noted per dataset in these notes "
    "are not reproduced. LLM failures are marked wrong and counted in "
    "summary.llm_errors (never randomly guessed, unlike VLMEvalKit)."
)

NO_PROTOCOL_NOTE = (
    "score_pipeline=[]: the dataset declares that no official eval protocol "
    "exists; numbers are mm-eval-convention, not official benchmark results."
)

_STRING_MATCH_VALUES = ("exact", "contains", "anls")


def parse_pipeline(spec, where: str) -> List[str]:
    """Validate a pipeline spec (comma-separated string from the CLI, or a JSON
    array from dataset metadata) into an ordered list of atomic stage names."""
    if isinstance(spec, str):
        stages = [x.strip() for x in spec.split(",") if x.strip()]
    elif isinstance(spec, (list, tuple)):
        stages = [str(x).strip() for x in spec if str(x).strip()]
    else:
        raise ValueError(f"{where}: score pipeline must be a list of stage names, got {type(spec).__name__}")
    if not stages:
        raise ValueError(f"{where}: empty pipeline; known stages: {sorted(STAGE_REGISTRY)}")
    unknown = [s for s in stages if s not in STAGE_REGISTRY]
    if unknown:
        raise ValueError(f"{where}: unknown stage(s) {unknown}; known stages: {sorted(STAGE_REGISTRY)}")
    if len(set(stages)) != len(stages):
        raise ValueError(f"{where}: duplicate stages in pipeline {stages}")
    for s in stages[:-1]:
        if s in GRADER_STAGE_NAMES:
            raise ValueError(
                f"{where}: `{s}` is a grader stage — it always decides every "
                f"sample, so stages after it can never run; a grader must be "
                f"the pipeline's last stage (and at most one per pipeline)"
            )
    return stages


@dataclass
class ResolvedProtocol:
    """Effective per-file scoring configuration after precedence resolution."""
    pipeline: List[str]
    numeric_rel_tol: float
    numeric_abs_tol: float
    string_match: str                        # exact | contains | anls
    anls_threshold: float
    task_type: Optional[str]
    official_protocol: Optional[bool]        # None = unknown (no declaration)
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
    """Apply the per-knob precedence CLI > dataset_meta > default. Raises
    ValueError for unsupported protocols (unless the user explicitly forced a
    pipeline via --score_pipeline) and for an unsupported `score_type` key."""
    explicit = _explicit(args)
    meta = dataset_meta or {}
    if "score_type" in meta:
        raise ValueError(
            f"{result_file}: dataset_meta contains an unsupported `score_type` "
            f"key. Declare the protocol with the `score_pipeline` schema in the "
            f"dataset's metadata.json (docs/en/SCORING.md, 'Dataset metadata "
            f"contract'), then re-run inference to regenerate this result.json."
        )
    declared = meta.get("score_pipeline")
    params = meta.get("score_params") or {}

    notes: List[str] = [GLOBAL_PROTOCOL_NOTE]
    sources: Dict[str, str] = {}

    # --- pipeline ------------------------------------------------------------
    declared_stages: Optional[List[str]] = None   # a declared, runnable protocol
    unsupported: Optional[Dict[str, Any]] = None
    if isinstance(declared, dict):
        if "unsupported" not in declared:
            raise ValueError(
                f"{result_file}: malformed score_pipeline object {declared!r}; "
                f'expected {{"unsupported": "<protocol>", "reason": "..."}} or a stage list'
            )
        unsupported = declared
    elif isinstance(declared, (list, tuple)):
        declared_stages = parse_pipeline(declared, f"{result_file} dataset_meta.score_pipeline") if declared else []
    elif declared is not None:
        raise ValueError(
            f"{result_file}: score_pipeline must be a stage list or an "
            f'{{"unsupported": ...}} object, got {type(declared).__name__}'
        )

    if "score_pipeline" in explicit:
        pipeline = parse_pipeline(args.score_pipeline, "--score_pipeline")
        sources["score_pipeline"] = "cli"
        if unsupported is not None:
            notes.append(
                f"explicit --score_pipeline overrides the dataset's unsupported official "
                f"protocol ({unsupported.get('unsupported')}): numbers are approximate, "
                f"NOT comparable to official results."
            )
        elif declared_stages and pipeline != declared_stages:
            notes.append(
                f"explicit --score_pipeline overrides the dataset's declared pipeline "
                f"{declared_stages}: the official protocol is NOT being executed; "
                f"numbers are not comparable to official results."
            )
    elif unsupported is not None:
        name = unsupported.get("unsupported", "unknown")
        reason = unsupported.get("reason", "no reason recorded")
        raise ValueError(
            f"{result_file}: the dataset's official protocol ({name}) is unsupported: "
            f"{reason}. Pass an explicit --score_pipeline to force approximate scoring "
            f"(numbers will NOT be official)."
        )
    elif declared_stages:
        pipeline = declared_stages
        sources["score_pipeline"] = "dataset_meta"
    else:  # no declaration ([] or absent): built-in default pipeline
        pipeline = parse_pipeline(args.score_pipeline, "--score_pipeline default")
        sources["score_pipeline"] = "default"

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
    # anls params spell it `threshold`; the rule-stage comparator knob is
    # `anls_threshold` — both feed the same resolved value.
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
            + " degraded to a binary llm-judge; official numbers aggregate "
              "the fractional/scaled judge scores and are not directly comparable."
        )

    # --- official-protocol stamp + score_note -------------------------------
    if declared is None:
        official = None            # nothing declared at all
    elif declared_stages == []:
        official = False           # explicit "no official protocol"
        notes.append(NO_PROTOCOL_NOTE)
    elif sources["score_pipeline"] == "cli" and (unsupported is not None or pipeline != declared_stages):
        official = False           # declared protocol overridden from the CLI
    else:
        official = True
    score_note = (meta.get("score_note") or "").strip()
    if score_note:
        notes.append(score_note)  # verbatim, per the metadata contract

    return ResolvedProtocol(
        pipeline=pipeline,
        numeric_rel_tol=numeric_rel_tol,
        numeric_abs_tol=numeric_abs_tol,
        string_match=string_match,
        anls_threshold=anls_threshold,
        task_type=meta.get("task_type"),
        official_protocol=official,
        protocol_notes=notes,
        knob_sources=sources,
    )
