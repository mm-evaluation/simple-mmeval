from collections import defaultdict
from typing import Any, Dict, List


def build_summary(sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(sample_results)
    correct = sum(1 for x in sample_results if x.get("is_correct") == 1)
    accuracy = (correct / total) if total else 0.0

    by_matcher = defaultdict(lambda: {"total": 0, "correct": 0})
    by_question_type = defaultdict(lambda: {"total": 0, "correct": 0})
    invalid = 0
    llm_errors = 0
    judge_probs = []
    for item in sample_results:
        matcher = item.get("matcher_used") or "unmatched"
        qtype = item.get("question_type") or "unknown"
        by_matcher[matcher]["total"] += 1
        by_matcher[matcher]["correct"] += int(item.get("is_correct") == 1)
        by_question_type[qtype]["total"] += 1
        by_question_type[qtype]["correct"] += int(item.get("is_correct") == 1)
        if item.get("status") == "invalid":
            invalid += 1
        # Samples whose LLM matcher call failed (API/parse error) — their 0s are
        # not verdicts; a non-zero count here means the accuracy is a lower bound.
        if str(item.get("reason", "")).startswith("llm_error"):
            llm_errors += 1
        if item.get("judge_prob") is not None:
            judge_probs.append(item["judge_prob"])

    # accuracy = strict full-credit rate (mean is_correct). mean_score averages
    # the fractional per-sample `score` where present (vqa_accuracy/anls graders;
    # falls back to is_correct per sample) — the OFFICIAL headline for fractional
    # protocols. Equal to accuracy whenever no sample carries a fractional score.
    mean_score = (
        sum((x["score"] if x.get("score") is not None else float(x.get("is_correct") == 1))
            for x in sample_results) / total
        if total else 0.0
    )
    summary = {
        "total": total,
        "valid": total - invalid,
        "correct": correct,
        "accuracy": accuracy,
        "mean_score": mean_score,
        "invalid": invalid,
        "llm_errors": llm_errors,
        "by_matcher": dict(by_matcher),
        "by_question_type": dict(by_question_type),
    }
    # Mean P(verdict) over judge-graded samples that returned logprobs —
    # observability for how confident the judge was, never a verdict input.
    if judge_probs:
        summary["judge_prob_mean"] = sum(judge_probs) / len(judge_probs)
    return summary
