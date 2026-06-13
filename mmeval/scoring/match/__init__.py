from mmeval.scoring.match.base import MatchResult
from mmeval.scoring.match.exact import ExactMatcher
from mmeval.scoring.match.template import TemplateMatcher
from mmeval.scoring.match.llm import LLMJudgeMatcher, LLMMatchMatcher

MATCHER_REGISTRY = {
    "exact": ExactMatcher,
    "template": TemplateMatcher,
    # LLM judges whether the response is correct (our approach).
    "llm-judge": LLMJudgeMatcher,
    # LLM only extracts the chosen option, then rule-based exact compare (VLMEvalKit).
    "llm-match": LLMMatchMatcher,
    # Backwards-compatible alias for the old name.
    "llm": LLMJudgeMatcher,
}

# Names whose matcher needs the parsed `args` (API client/config) at construction.
LLM_MATCHER_NAMES = {"llm", "llm-judge", "llm-match"}

__all__ = [
    "MatchResult",
    "ExactMatcher",
    "TemplateMatcher",
    "LLMJudgeMatcher",
    "LLMMatchMatcher",
    "MATCHER_REGISTRY",
    "LLM_MATCHER_NAMES",
]
