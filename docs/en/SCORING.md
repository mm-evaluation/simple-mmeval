# Scoring

`mmeval/score.py` grades the `result.json` files produced by inference into
`score.json` files:

```bash
PYTHONPATH=. python3 mmeval/score.py --score_out_dir work_dirs/my_run
```

For how well the framework reproduces official published results, see
[VALIDATION.md](VALIDATION.md).

## The pipeline model

All scoring is expressed as a **pipeline: an ordered list of atomic stages**.
There are no composite scorer types — a protocol like "rules first, LLM judge
for the rest" is written as the list `exact-match,rule-match,llm-judge`.
The naming scheme: `X-match` = candidate extraction via X (exact form /
rule heuristics / an LLM) followed by rule comparison; `llm-judge` = the LLM
decides correctness directly; `vqa-accuracy` / `anls` = named official
metrics (grader stages).

| Stage | Mechanism | What it does |
|---|---|---|
| `exact-match` | rules, free | Strict-form comparison: bare option letter / clean letter set, pure yes/no, whole-string number or text. Near-zero false positives; passes anything ambiguous. |
| `rule-match` | rules, free | Robust answer extraction from verbose / chain-of-thought output (option-letter inference, answer cues, `\boxed{}`, option-text restatement), then comparison. |
| `llm-match` | LLM | The LLM extracts the chosen option letter(s); rule-based set compare (VLMEvalKit protocol). MCQ only; passes non-MCQ samples. |
| `llm-judge` | LLM | The LLM judges correctness directly from question + response + ground truth. Both verdicts decide the sample (yes → 1.0, no → 0.0). |
| `vqa-accuracy` | grader, fractional | Official VQA consensus accuracy (leave-one-out `min(1, matches/3)` over the multi-annotator answer list). |
| `anls` | grader, fractional | ANLS: `1 − min` normalized Levenshtein distance over the references, zeroed below the threshold. |

The two LLM stages run their judge through a configurable provider
(`--judge_provider`): `local` (an in-process open-weight model loaded with the
framework's own conventions — no external service), `openai`, or
`azure_openai`. The stage logic, prompt, retry, and `llm_error` accounting are
identical across providers.

### Execution semantics

Stages run in order; each returns one of three outcomes:

- **pass** — the stage cannot decide the sample; the next stage runs. If every
  stage passes, the sample scores `0.0` with `decided_by: null` (undecided,
  counted wrong). Rule stages never decide a miss — they pass instead, so a
  later stage can still grade the sample.
- **score** — the stage decided; the pipeline short-circuits. Each sample row
  records `score` (0..1), `is_correct` (`1` only at full credit) and
  `decided_by` (the stage name). Fractional protocols read
  `summary.mean_score` as the headline; `summary.accuracy` is the strict
  full-credit rate.
- **invalid** — the sample cannot be graded under the stage's protocol (e.g.
  `vqa-accuracy` without the multi-annotator reference list); recorded as
  `status: invalid` with the stage's reason.

The grader stages (`vqa-accuracy`, `anls`) always decide, so they are only
valid as the **last** stage, at most one per pipeline (validated with a clear
error). Compositions like
`exact-match,anls` (exact tier first, fractional similarity for the rest) are
legal pipelines.

LLM/API failures never become verdicts: the failing stage passes with an
`llm_error: ...` reason that survives into the sample row and
`summary.llm_errors`, and such rows are re-scored on resume — never guessed.

### CLI

```bash
--score_pipeline exact-match,rule-match     # default
--score_pipeline rule-match,llm-match       # MCQ letter-extraction protocol
--score_pipeline llm-judge                  # pure LLM judging
--score_pipeline vqa-accuracy               # official VQA accuracy
```

Stage parameters are pipeline-level flags, consumed by the stages that use
them (per-knob precedence: explicit CLI flag > dataset metadata > default;
`score.json config.knob_sources` records who decided each knob):

| Flag | Consumed by |
|---|---|
| `--score_numeric_rel_tol`, `--score_numeric_abs_tol` | `exact-match`, `rule-match` (numeric answers) |
| `--score_string_match` (`exact`/`contains`/`anls`) | `exact-match`, `rule-match` (open answers) |
| `--score_anls_threshold` | `anls` and the `anls` string-match mode of the rule-based stages |
| `--judge_provider/model/temperature/max_tokens/...` | the LLM stages (`llm-match`, `llm-judge`) |

## Dataset metadata contract

mm-eval datasets declare their official protocol per subset in the repo-root
`metadata.json`. The scorer reads it from the `dataset_meta` block the loader
injects into every sample (so it travels inside `result.json`):

```jsonc
"subsets": {
  "en": {
    "task_type": "multiple_choice_qa",
    "score_pipeline": ["rule-match", "llm-match"],   // the official protocol
    "score_params": {"numeric_rel_tol": 0.05},            // optional stage params
    "score_protocol": {"note": "..."}                     // verbatim protocol note
  }
}
```

`score_pipeline` takes one of three forms:

1. **A stage list** — the official protocol; running it stamps
   `config.official_protocol: true`. An explicit `--score_pipeline` that differs
   overrides it, stamps the output `official_protocol: false`, and appends a
   protocol note saying so.
2. **`[]`** — the dataset explicitly declares that **no official protocol
   exists**; the scorer runs the default pipeline and stamps
   `official_protocol: false` with a note.
3. **`{"unsupported": "<protocol>", "reason": "..."}`** — the official
   protocol cannot be executed by this scorer (e.g. corpus-level caption
   metrics, code execution). Scoring is **refused** with the reason; an
   explicit `--score_pipeline` forces approximate scoring, stamped non-official.

Datasets with no `score_pipeline` declaration score under the default
pipeline with `official_protocol: null` (unknown).

### Published metadata

Every dataset on the mm-eval org declares `score_pipeline` natively
(org-wide migration completed 2026-07-19; the loader's temporary
`score_type`→`score_pipeline` translation boundary is deleted). The retired
composite `score_type` vocabulary is rejected everywhere with an actionable
error: the loader refuses a metadata.json that still carries it (re-push the
dataset), and the scorer refuses a `result.json` whose `dataset_meta`
contains it (produced by an older loader — re-run inference or upgrade the
file).

The audited per-subset pipeline for all published datasets is listed in
[SCORING_COVERAGE.md](SCORING_COVERAGE.md).

## Resume

Scoring resumes from `score.json` / `score.json.tmp` caches keyed by a config
fingerprint that captures everything verdict-determining: the resolved
pipeline, stage params, and — only when an LLM stage is present — the
judge identity (provider/model/temperature/include_reason/resolved
max_tokens). Any change invalidates the cache; rule-only pipelines survive
judge-config edits. Rows that failed with `llm_error` are always re-scored.
Reruns of an unchanged config are byte-identical.
