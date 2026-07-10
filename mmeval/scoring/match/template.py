import re

from mmeval.scoring.match.base import BaseMatcher, MatchResult
from mmeval.scoring.match.exact import gt_references, numbers_match, open_text_match
from mmeval.scoring.extraction import (
    extract_letter_set,
    extract_option_robust,
    extract_yes_no,
    normalize_for_exact,
    normalize_open_answer,
    normalize_text,
    parse_gt_letters,
    parse_number,
)

# Answer cues that introduce the model's final value in a chain of thought.
_NUMERIC_CUE = re.compile(r"(?:final answer|the answer is|answer is|answer\s*[:\-]|=)")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_OPEN_ANSWER_RES = (
    re.compile(r"(?:final answer|answer is|the answer is)\s*[:\-]?\s*(.+)$", re.IGNORECASE),
    re.compile(r"^\s*(.+?)\s*$", re.IGNORECASE),
)


class TemplateMatcher(BaseMatcher):
    """Robust rule matcher: per-type extraction tuned for chain-of-thought
    output. Still fully deterministic — no API calls."""

    name = "template"

    def match(self, sample, context):
        if context["gt"] is None or context["pred"] is None:
            return MatchResult(is_match=False)

        handler = {
            "mcq": self._match_mcq,
            "yes_no": self._match_yes_no,
            "numeric": self._match_numeric,
        }.get(context["question_type"], self._match_open)
        return handler(context)

    def _match_mcq(self, context):
        # Letter-SET grading (single-select = |set|=1; LogicVista multi-letter
        # answers compare sorted sets, vlmeval/dataset/utils/logicvista.py:50-68).
        pred_raw = context["pred"]
        gt_raw = context["gt"]
        candidates = context["options"]
        option_texts = context.get("option_texts", {})

        # Compact multi-letter gts ("AC") are accepted at match time (the row
        # is already typed mcq) — but only with REAL parsed options: under the
        # A-F fallback, word-like gts would parse as letter sets ("BED" -> B,D,E).
        gt_letters = parse_gt_letters(gt_raw)
        if gt_letters is None and context.get("options_parsed"):
            gt_letters = extract_letter_set(gt_raw, candidates)
        if gt_letters is None:
            g = extract_option_robust(gt_raw, candidates, option_texts) or normalize_for_exact(gt_raw).upper()
            gt_letters = (g,) if g else None

        # Clean letters-and-separators segments first ("A, C", "<answer>BD</answer>",
        # "the answers are A and C"); then the robust single-letter extractor
        # (VLMEvalKit can_infer + our patches: <answer> tags, cues, CJK, \boxed,
        # option-text restatement).
        pred_letters = extract_letter_set(pred_raw, candidates)
        if pred_letters is None:
            option = extract_option_robust(pred_raw, candidates, option_texts)
            pred_letters = (option,) if option else None

        if gt_letters and pred_letters and pred_letters == gt_letters:
            return MatchResult(is_match=True, matched=", ".join(gt_letters))
        return MatchResult(is_match=False)

    def _match_yes_no(self, context):
        # VLMEvalKit YOrN_Extraction parity (vlmeval/dataset/utils/yorn.py:254-261):
        # word-level yes XOR no; a response containing both defers to the next
        # matcher (upstream marks it 'Unknown').
        gt_norm = normalize_open_answer(context["gt"])
        pred_yn = extract_yes_no(context["pred"])
        if pred_yn is not None and pred_yn == gt_norm:
            return MatchResult(is_match=True, matched=pred_yn)
        return MatchResult(is_match=False)

    def _match_numeric(self, context):
        gt_nums = [n for n in (parse_number(r) for r in gt_references(context["gt"])) if n is not None]
        if not gt_nums:
            return MatchResult(is_match=False)
        pred_norm = normalize_open_answer(context["pred"])
        # Prefer the segment after the LAST answer cue (a CoT usually walks
        # through intermediate numbers first); fall back to the whole text.
        cues = list(_NUMERIC_CUE.finditer(pred_norm))
        segment = pred_norm[cues[-1].end():] if cues else pred_norm
        nums = _NUMBER_RE.findall(segment) or _NUMBER_RE.findall(pred_norm)
        if nums and any(numbers_match(float(nums[0]), g, context) for g in gt_nums):
            return MatchResult(is_match=True, matched=str(context["gt"]))
        return MatchResult(is_match=False)

    def _match_open(self, context):
        pred_norm_text = normalize_text(context["pred"])

        # generic "final answer is ..." extraction, then whole-string restatement;
        # each candidate compared per the protocol's string_match (exact/contains/anls).
        for pattern in _OPEN_ANSWER_RES:
            m = pattern.search(pred_norm_text)
            if not m:
                continue
            if open_text_match(m.group(1), m.group(1), context["gt"], context):
                return MatchResult(is_match=True, matched=str(context["gt"]))
        return MatchResult(is_match=False)
