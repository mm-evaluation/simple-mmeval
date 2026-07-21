# Validation: reproduced official results

Has this framework been validated against official evaluation results? Yes.
Each check below runs a real model end to end — the framework's own inference
and scoring — using the dataset's **official scoring pipeline unchanged**, and
compares the result against an official reference: a model card, a benchmark's
technical report, the [OpenVLM leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard),
or the upstream [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
evaluator run on the same machine.

## How to read the table

- **`eval_score`** is this framework's result; **`reference_score`** is the
  official number; **`delta` = `eval_score` − `reference_score`** (percentage
  points).
- Scores are each benchmark's official headline metric, as a percentage:
  accuracy for the rule-based pipelines (ChartQA, OCRBench) and mean
  ANLS / VQA-accuracy for the fractional metrics (DocVQA, InfographicVQA,
  TextVQA).
- **`score_pipeline`** is the ordered list of scoring stages that ran (see
  [SCORING.md](SCORING.md)); it is the dataset's declared official protocol.
- **`custom_template`** = `yes` where the model was prompted with the official
  benchmark's own prompt template rather than the dataset's shipped default
  (see notes below).

**Agreement.** All ten checks agree with their official reference to within
**±1.0 percentage point** on the inference side, and the scoring side is
**exact**: on identical model predictions the framework's per-sample verdicts
match the official comparators with **zero mismatches across 33,300 samples**
(ANLS and VQA-accuracy agree to 1e-9). The small delta that remains is
inference-side variance — prompt construction, image handling, generation
settings — not scoring.

## Reproductions

| model | model_revision | dataset | split | score_pipeline | custom_template | samples | eval_score | reference_score | delta | reference_source | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2-VL-2B-Instruct | 895c3a49bc3f | ChartQA | test | exact-match, rule-match (numeric_rel_tol 0.05) | no | 2500 | 74.24 | 73.50 | +0.74 | Qwen2-VL-2B-Instruct model card (ChartQA_test) |  |
| Qwen2-VL-2B-Instruct | 895c3a49bc3f | OCRBench | test | exact-match, rule-match (string_match contains) | no | 1000 | 79.10 | 79.70 | -0.60 | OpenVLM leaderboard (797/1000); model card 79.4 |  |
| Qwen2-VL-2B-Instruct | 895c3a49bc3f | DocVQA | validation | anls (threshold 0.5) | no | 5349 | 88.48 | 89.24 | -0.76 | Upstream VLMEvalKit run, same machine |  |
| Qwen2-VL-2B-Instruct | 895c3a49bc3f | InfographicVQA | validation | anls (threshold 0.5) | yes | 2801 | 64.31 | 64.20 | +0.11 | Upstream VLMEvalKit run, same machine | matched official inference config |
| Qwen2-VL-2B-Instruct | 895c3a49bc3f | TextVQA | validation | vqa-accuracy | no | 5000 | 79.16 | 79.70 | -0.54 | Qwen2-VL-2B-Instruct model card (TextVQA_val) |  |
| Qwen2.5-VL-3B-Instruct | 66285546d2b8 | ChartQA | test | exact-match, rule-match (numeric_rel_tol 0.05) | no | 2500 | 83.64 | 84.00 | -0.36 | Qwen2.5-VL Technical Report Table 5 (arXiv:2502.13923) |  |
| Qwen2.5-VL-3B-Instruct | 66285546d2b8 | OCRBench | test | exact-match, rule-match (string_match contains) | no | 1000 | 82.70 | 82.80 | -0.10 | OpenVLM leaderboard (828/1000); same-machine upstream 82.6 | matched official inference config |
| Qwen2.5-VL-3B-Instruct | 66285546d2b8 | DocVQA | validation | anls (threshold 0.5) | no | 5349 | 92.46 | 93.01 | -0.55 | Upstream VLMEvalKit run, same machine |  |
| Qwen2.5-VL-3B-Instruct | 66285546d2b8 | InfographicVQA | validation | anls (threshold 0.5) | no | 2801 | 75.15 | 76.03 | -0.88 | Upstream VLMEvalKit run, same machine | matched official inference config |
| Qwen2.5-VL-3B-Instruct | 66285546d2b8 | TextVQA | validation | vqa-accuracy | no | 5000 | 79.82 | 79.30 | +0.52 | Qwen2.5-VL Technical Report Table 5 + model card |  |

**Matched official inference config.** Three checks needed the official
inference configuration — image pixel window, prompt construction, and
generation length — reproduced before the model's outputs agreed with the
reference to within tolerance. Where the official prompt template itself was
reproduced, `custom_template` is `yes` (here, 2B InfographicVQA). This is
inference-side alignment only; no scoring behaviour changes.
