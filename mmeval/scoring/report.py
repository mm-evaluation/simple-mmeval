from collections import defaultdict
from typing import Any, Dict, List


def build_summary(sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(sample_results)
    correct = sum(1 for x in sample_results if x.get("is_correct") == 1)
    # accuracy divides by ALL samples (invalid rows count as wrong);
    # accuracy_valid divides by the gradable samples only — the honest number
    # on splits with withheld answers or malformed rows.
    accuracy = (correct / total) if total else 0.0

    by_stage = defaultdict(lambda: {"total": 0, "correct": 0})
    by_question_type = defaultdict(lambda: {"total": 0, "correct": 0})
    invalid = 0
    llm_errors = 0
    judge_probs = []
    for item in sample_results:
        stage = item.get("decided_by") or "undecided"
        qtype = item.get("question_type") or "unknown"
        by_stage[stage]["total"] += 1
        by_stage[stage]["correct"] += int(item.get("is_correct") == 1)
        by_question_type[qtype]["total"] += 1
        by_question_type[qtype]["correct"] += int(item.get("is_correct") == 1)
        if item.get("status") == "invalid":
            invalid += 1
        # Samples whose judge-stage API call failed (API/parse error) — their 0s are
        # not verdicts; a non-zero count here means the accuracy is a lower bound.
        if str(item.get("reason", "")).startswith("llm_error"):
            llm_errors += 1
        if item.get("judge_prob") is not None:
            judge_probs.append(item["judge_prob"])

    # accuracy = strict full-credit rate (mean is_correct). mean_score averages
    # the fractional per-sample `score` (metric stages emit true fractions;
    # rule/judge stages emit 1.0/0.0) — the OFFICIAL headline for fractional
    # protocols. Equal to accuracy whenever no sample carries a fractional score.
    mean_score = (
        sum((x["score"] if x.get("score") is not None else float(x.get("is_correct") == 1))
            for x in sample_results) / total
        if total else 0.0
    )
    valid = total - invalid
    summary = {
        "total": total,
        "valid": valid,
        "correct": correct,
        "accuracy": accuracy,
        "accuracy_valid": (correct / valid) if valid else 0.0,
        "mean_score": mean_score,
        "invalid": invalid,
        "llm_errors": llm_errors,
        "by_stage": dict(by_stage),
        "by_question_type": dict(by_question_type),
    }
    # Mean P(verdict) over judge-graded samples that returned logprobs —
    # observability for how confident the judge was, never a verdict input.
    if judge_probs:
        summary["judge_prob_mean"] = sum(judge_probs) / len(judge_probs)
    return summary
