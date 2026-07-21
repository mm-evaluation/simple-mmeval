"""The atomic scoring stage: the single primitive behind every pipeline.

A scoring pipeline is an ordered list of stages. For each sample, stages run
in order; each returns a StageResult with one of three outcomes:

- PASS    — the stage cannot decide this sample; the next stage runs. If every
            stage passes, the sample scores 0.0 with decided_by null
            (undecided, counted wrong). A pass may carry a reason — notably
            "llm_error: ..." markers, which survive into the output row and
            summary.llm_errors unless a later stage decides the sample.
- SCORE   — the stage decided the sample and the pipeline short-circuits.
            `score` is in [0, 1]: rule stages emit 1.0 on a match and never
            decide a miss (they pass instead); llm-judge emits 1.0/0.0;
            metric stages emit fractional values. is_correct = score >= 1.0.
- INVALID — the sample cannot be graded under this stage's protocol (e.g.
            vqa-accuracy without the multi-annotator reference list). Terminal:
            the sample is recorded as status=invalid with the stage's reason.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

PASS = "pass"
SCORE = "score"
INVALID = "invalid"


@dataclass
class StageResult:
    outcome: str
    score: float = 0.0
    matched: Optional[str] = None
    reason: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def passed(cls, reason: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> "StageResult":
        return cls(outcome=PASS, reason=reason, meta=meta or {})

    @classmethod
    def scored(cls, score: float, matched: Optional[str] = None,
               reason: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> "StageResult":
        return cls(outcome=SCORE, score=float(score), matched=matched, reason=reason, meta=meta or {})

    @classmethod
    def not_gradable(cls, reason: str) -> "StageResult":
        return cls(outcome=INVALID, reason=reason)


class BaseStage:
    name = "base"
    # Declared by each stage class; the registry derives LLM_STAGE_NAMES /
    # GRADER_STAGE_NAMES from these, so no behavior is keyed off name sets.
    uses_llm = False    # calls an LLM API: judge identity enters the fingerprint
    # grader: the stage always decides (fractional score 0..1) or invalidates,
    # never abstains -> only valid as the last pipeline stage, at most one.
    # DELIBERATE COUPLING: `grader` bundles "fractional output" with "always
    # decides / last-only". If a future stage ever needs one without the
    # other, split this flag — known, accepted trade-off, not an oversight.
    grader = False

    def __init__(self, args=None):
        # Uniform construction: every stage accepts the parsed args; stages
        # without external clients ignore them.
        self.args = args

    def run(self, sample: Dict[str, Any], context: Dict[str, Any]) -> StageResult:
        raise NotImplementedError
