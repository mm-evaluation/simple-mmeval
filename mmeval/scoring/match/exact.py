from mmeval.scoring.match.base import BaseMatcher, MatchResult
from mmeval.scoring.extraction import (
    extract_letter_set,
    extract_option_strict,
    normalize_for_exact,
    normalize_open_answer,
    parse_gt_letters,
    parse_number,
)


def extract_letters_strict(pred, candidates):
    """Strict-tier letter-set extraction: a clean letters-and-separators answer
    ("B", "A, C", "BD"), else a single unambiguous strict-form letter."""
    letters = extract_letter_set(pred, candidates)
    if letters:
        return letters
    single = extract_option_strict(pred, candidates)
    return (single,) if single else None


def numbers_match(pred: float, gt: float, context) -> bool:
    """Exact float equality by default; the protocol's tolerances widen it:
    relative (`numeric_rel_tol` — ChartQA 0.05) and/or absolute
    (`numeric_abs_tol` — DynaMath 0.001, OlympiadBench-style epsilon). Either
    satisfied tolerance matches. Relative tolerance does NOT apply when the
    ground truth is 0 — the official relaxed-accuracy protocol falls back to
    exact matching there (lmms-eval chartqa/utils.py relaxed_correctness:
    `target_float` truthiness sends target==0 to the exact-match branch)."""
    if pred == gt:
        return True
    abs_tol = float(context.get("numeric_abs_tol") or 0.0)
    if abs_tol > 0.0 and abs(pred - gt) <= abs_tol:
        return True
    rel_tol = float(context.get("numeric_rel_tol") or 0.0)
    if rel_tol > 0.0 and gt != 0.0:
        return abs(pred - gt) / abs(gt) <= rel_tol
    return False


def gt_references(gt) -> list:
    """A gt may be a single value or a reference LIST (OCRBench answer lists,
    VQA-style multi-annotator answers)."""
    if isinstance(gt, (list, tuple)):
        return [x for x in gt if x is not None]
    return [gt]


def open_text_match(pred_text: str, pred_raw, gt_raw, context) -> bool:
    """The rule-chain text comparator, per the protocol's `string_match`:
    exact (default) — normalized equality against any reference;
    contains — OCRBench protocol: a normalized reference appears as a substring
    of the normalized prediction;
    anls — threshold ANLS against the references (ChartQAPro composite)."""
    mode = context.get("string_match") or "exact"
    refs = gt_references(gt_raw)
    if mode == "anls":
        from mmeval.scoring.graders import anls
        score, error = anls(pred_raw, [str(r) for r in refs],
                            threshold=float(context.get("anls_threshold") or 0.5))
        return error is None and score > 0.0
    pred_norm = normalize_open_answer(pred_text)
    if not pred_norm:
        return False
    for ref in refs:
        ref_norm = normalize_open_answer(ref)
        if not ref_norm:
            continue
        if mode == "contains":
            if ref_norm in pred_norm:
                return True
        elif pred_norm == ref_norm:
            return True
    return False


class ExactMatcher(BaseMatcher):
    """Strict first pass: near-zero false positives, zero cost. Anything it
    cannot decide unambiguously is left for the next matcher in the chain."""

    name = "exact"

    def match(self, sample, context):
        question_type = context["question_type"]
        pred_raw = context["pred"]
        gt_raw = context["gt"]

        if gt_raw is None or pred_raw is None:
            return MatchResult(is_match=False)

        if question_type == "mcq":
            # Letter-SET grading: single-select is the |set|=1 special case,
            # multi-letter gts like LogicVista's "A, C" the general one.
            candidates = context["options"]
            # Compact multi-letter gts ("AC") are accepted HERE — the row is
            # already typed mcq — while parse_gt_letters stays conservative for
            # type inference. Only with REAL parsed options, though: under the
            # A-F fallback (options-in-image datasets) word-like gts would
            # parse as letter sets ("BED" -> B,D,E).
            gt_letters = parse_gt_letters(gt_raw)
            if gt_letters is None and context.get("options_parsed"):
                gt_letters = extract_letter_set(gt_raw, candidates)
            if gt_letters is None:
                g = extract_option_strict(gt_raw, candidates) or normalize_for_exact(gt_raw).upper()
                gt_letters = (g,) if g else None
            pred_letters = extract_letters_strict(pred_raw, candidates)
            if gt_letters and pred_letters and pred_letters == gt_letters:
                return MatchResult(is_match=True, matched=", ".join(gt_letters))
            return MatchResult(is_match=False)

        if question_type == "yes_no":
            pred_norm = normalize_open_answer(pred_raw)
            if pred_norm in {"yes", "no"} and pred_norm == normalize_open_answer(gt_raw):
                return MatchResult(is_match=True, matched=pred_norm)
            return MatchResult(is_match=False)

        if question_type == "numeric":
            # Strict tier: the whole prediction is a number.
            gt_num = parse_number(gt_raw)
            pred_num = parse_number(pred_raw)
            if gt_num is not None and pred_num is not None and numbers_match(pred_num, gt_num, context):
                return MatchResult(is_match=True, matched=str(gt_raw))
            return MatchResult(is_match=False)

        if open_text_match(str(pred_raw if not isinstance(pred_raw, list) else (pred_raw[0] if pred_raw else "")),
                           pred_raw, gt_raw, context):
            return MatchResult(is_match=True, matched=str(gt_raw))
        return MatchResult(is_match=False)
