"""Fractional grader stages. Both always DECIDE (score 0..1) or mark the
sample invalid — they never pass — so they are only valid as a pipeline's
last stage. is_correct = 1 only at full credit; the official headline
aggregate for fractional protocols is summary.mean_score.

Protocol sources (implemented 1:1):
- vqa-accuracy: lmms-eval lmms_eval/tasks/vqav2/utils.py:15-42
  (vqav2_process_results) — whitespace-normalize, EvalAIAnswerProcessor
  punctuation+digit/article normalization, then the official leave-one-out
  consensus accuracy: for each reference answer, acc = min(1, #matching OTHER
  references / 3); sample score = mean over references.
- anls: lmms-eval lmms_eval/api/metrics.py:293-321 (anls) — per
  reference, normalized Levenshtein distance over lowercased
  whitespace-collapsed strings; score = 1 - min distance, zeroed below the
  threshold (default 0.5).
"""
from typing import Any, List, Optional, Tuple

from mmeval.scoring.stages.base import BaseStage, StageResult
from mmeval.scoring.vqa_eval_metric import EvalAIAnswerProcessor

_VQA_PROCESSOR = EvalAIAnswerProcessor()


def _reference_list(gt: Any) -> Optional[List[str]]:
    if isinstance(gt, (list, tuple)) and gt:
        return [str(x) for x in gt]
    if isinstance(gt, str) and gt.strip():
        return [gt]
    return None


def _vqa_normalize(text: str) -> str:
    # lmms-eval vqav2_process_results: \n/\t -> space, strip, then the official
    # EvalAI punctuation and digit/article passes.
    text = text.replace("\n", " ").replace("\t", " ").strip()
    text = _VQA_PROCESSOR.process_punctuation(text)
    text = _VQA_PROCESSOR.process_digit_article(text)
    return text


def vqa_consensus(pred: Any, gt: Any) -> Tuple[Optional[float], Optional[str]]:
    """Official VQA consensus accuracy. Returns (score, error): score is None
    on error. Requires the multi-annotator reference list (VQAv2/OK-VQA/
    TextVQA/VizWiz ship 10 answers) — the leave-one-out formula is undefined
    for a single reference."""
    refs = _reference_list(gt)
    if refs is None or len(refs) < 2:
        return None, "vqa-accuracy requires the multi-annotator answer list (>=2 references) as gt"
    pred_text = pred[0] if isinstance(pred, list) and pred else pred
    if not isinstance(pred_text, str):
        return None, "vqa-accuracy: prediction is not text"
    res = _vqa_normalize(pred_text)
    norm_refs = [_vqa_normalize(r) for r in refs]
    accs = []
    for i in range(len(norm_refs)):
        others = norm_refs[:i] + norm_refs[i + 1:]
        matching = sum(1 for o in others if o == res)
        accs.append(min(1.0, matching / 3.0))
    return sum(accs) / len(accs), None


def _levenshtein(s1: str, s2: str) -> int:
    # lmms-eval api/metrics.py levenshtein_distance, verbatim algorithm.
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def anls(pred: Any, gt: Any, threshold: float = 0.5) -> Tuple[Optional[float], Optional[str]]:
    """ANLS (lmms-eval api/metrics.py:293-321): 1 - min normalized Levenshtein
    over the references, zeroed below `threshold`. Accepts a single reference
    string or a reference list."""
    refs = _reference_list(gt)
    if refs is None:
        return None, "anls: empty ground truth"
    pred_text = pred[0] if isinstance(pred, list) and pred else pred
    if not isinstance(pred_text, str):
        return None, "anls: prediction is not text"
    det = " ".join(pred_text.strip().lower().split())
    values = []
    for answer in refs:
        gt_answer = " ".join(answer.strip().lower().split())
        dist = _levenshtein(gt_answer, det)
        length = max(len(answer.upper()), len(pred_text.upper()))
        values.append(0.0 if length == 0 else float(dist) / float(length))
    score = 1 - min(values)
    if score < threshold:
        score = 0.0
    return score, None


class VqaAccuracyStage(BaseStage):
    """Official VQA consensus accuracy (fractional grader)."""

    name = "vqa-accuracy"
    grader = True

    def run(self, sample, context):
        score, error = vqa_consensus(context["pred"], context["gt"])
        if score is None:
            return StageResult.not_gradable(error)
        return StageResult.scored(score)


class AnlsStage(BaseStage):
    """ANLS similarity against the references (fractional grader)."""

    name = "anls"
    grader = True

    def run(self, sample, context):
        score, error = anls(context["pred"], context["gt"],
                            threshold=float(context.get("anls_threshold") or 0.5))
        if score is None:
            return StageResult.not_gradable(error)
        return StageResult.scored(score)
