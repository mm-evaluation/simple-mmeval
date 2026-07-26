# Open-weight judge

Two scoring stages use an LLM to grade a sample: `llm-judge` (the LLM judges
correctness directly) and `llm-match` (the LLM extracts the chosen option for an
MCQ, then rules compare it). By default these call an OpenAI-compatible API. The
**local** provider runs an open-weight model in-process instead, so you can grade
datasets whose official protocol uses an LLM stage without any external service
or API key.

## Enabling it

```bash
PYTHONPATH=. python mmeval/score.py --score_out_dir work_dirs/my_run \
  --score_pipeline llm-judge \
  --judge_provider local --judge_model Qwen/Qwen2.5-7B-Instruct
```

The model is loaded with `transformers` (`device_map="auto"`); the first call
loads the weights and the rest of the run reuses them.

## Configuration

| Flag | Purpose | Default |
|---|---|---|
| `--judge_provider` | `local`, `openai`, or `azure_openai` | `openai` |
| `--judge_model` | for `local`, a Hugging Face causal-LM id | — |
| `--judge_temperature` | generation temperature | `0.0` |
| `--judge_max_tokens` | max completion tokens | env `JUDGE_MAX_TOKENS`, else 2048 |
| `--judge_max_retry` | attempts per call | env `JUDGE_MAX_RETRY`, else 3 |
| `--judge_concurrency` | max concurrent calls | env `JUDGE_MAX_CONCURRENCY`, else 4 |
| `--judge_include_reason` | keep the judge's reason in the output | off |

These flags affect only the two LLM stages; a rule-only pipeline (for example
`exact-match,rule-match`) ignores them.

## Supported models

`--judge_model` accepts any open-weight text instruction model (a Hugging Face
causal-LM id) — for example `Qwen/Qwen2.5-7B-Instruct`,
`Qwen/Qwen2.5-72B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`, or
`microsoft/Phi-3.5-mini-instruct`. Which one to use is your choice.

## Measured against a GPT judge

Reliability as a substitute for a GPT judge was measured directly. For each
baseline below, an open judge re-graded the same model responses that the
benchmark's official GPT grader had already scored, and the two sets of verdicts
were compared sample by sample. Baselines are separated by what the recorded GPT
label actually is: a correctness verdict, which tests `llm-judge`, or an answer
extraction, which tests `llm-match`.

Per-sample agreement is the primary figure. A dataset score can match by
coincidence when errors in opposite directions cancel out, whereas agreement
counts every disagreement.

| Baseline (GPT reference) | Metric | Qwen2.5-7B | Qwen2.5-72B | Mistral-7B | Phi-3.5-mini |
|---|---|---|---|---|---|
| **MathVerse** — GPT correctness verdict (`llm-judge`, n=500, GPT score 44.2) | Per-sample agreement | 93.6% | 96.2% | 90.2% | 90.4% |
| | Δ vs GPT score | +0.8 | +2.2 | +9.8 | +8.4 |
| **MM-Vet** — GPT graded verdict (`llm-judge`, n=191) | Per-sample agreement † | 88.2% | 86.1% | 85.4% | 81.8% |
| | Δ vs GPT score | — † | — † | — † | — † |
| **MathVista** — GPT answer extraction (`llm-match`, n=300, GPT score 45.3) | Per-sample agreement | 97.3% | 97.0% | 98.7% | 98.7% |
| | Δ vs GPT score | +1.3 | +0.3 | 0.0 | 0.0 |

† MM-Vet's official metric is partial credit (0.0–1.0) while a judge verdict is
binary, so the dataset-level scores are not directly comparable. Agreement is
measured on the 144 rows where GPT gave an unambiguous correct/incorrect verdict.

Two limits are worth reading off the table. Agreement depends on the model
chosen, so these are not one result but a range across the models tested. And all
three baselines are math and reasoning benchmarks, so agreement in other domains
is not established.

## Behaviour that affects your runs

- **Deterministic.** Judging is greedy (temperature 0), so re-scoring the same
  run with the same judge produces byte-identical output.
- **Resume cache.** On resume a scored row is reused only when the judge
  fingerprint matches: provider, model, temperature, max tokens, the reason flag,
  and — for `local` — the model's resolved snapshot revision. Change any of these
  and the affected rows are re-judged.
- **Failures are never guessed.** If a judge call fails or its reply cannot be
  parsed, that row is not turned into a verdict. It is recorded with an
  `llm_error` reason, counted in `summary.llm_errors`, and re-judged on resume.
