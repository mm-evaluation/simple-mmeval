from mmeval.scoring.stages.base import BaseStage, StageResult, INVALID, PASS, SCORE
from mmeval.scoring.stages.exact_match import ExactMatchStage
from mmeval.scoring.stages.rule_match import RuleMatchStage
from mmeval.scoring.stages.llm import LLMMatchStage, LLMJudgeStage
from mmeval.scoring.stages.metrics import AnlsStage, VqaAccuracyStage

# Every atomic scoring stage, by pipeline name (mainstream eval vocabulary:
# exact match, answer extraction, LLM-as-judge, VQA accuracy, ANLS).
STAGE_REGISTRY = {
    "exact-match": ExactMatchStage,
    "rule-match": RuleMatchStage,
    # LLM extracts the chosen option, then rule-based set compare (VLMEvalKit).
    "llm-match": LLMMatchStage,
    # LLM judges whether the response is correct (our approach).
    "llm-judge": LLMJudgeStage,
    "vqa-accuracy": VqaAccuracyStage,
    "anls": AnlsStage,
}

# Derived from the stage classes' own declarations — the class attribute is
# the single source of truth, so a new stage cannot silently miss these sets.
# uses_llm: the stage calls an LLM API; its presence pulls the judge identity
# into the resume fingerprint. grader: the stage always decides (or
# invalidates) and never abstains, so it is only valid as the last stage.
LLM_STAGE_NAMES = frozenset(n for n, c in STAGE_REGISTRY.items() if c.uses_llm)
GRADER_STAGE_NAMES = frozenset(n for n, c in STAGE_REGISTRY.items() if c.grader)

__all__ = [
    "BaseStage",
    "StageResult",
    "PASS",
    "SCORE",
    "INVALID",
    "ExactMatchStage",
    "RuleMatchStage",
    "LLMMatchStage",
    "LLMJudgeStage",
    "VqaAccuracyStage",
    "AnlsStage",
    "STAGE_REGISTRY",
    "LLM_STAGE_NAMES",
    "GRADER_STAGE_NAMES",
]
