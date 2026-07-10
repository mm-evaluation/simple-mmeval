# Jinja templates — canonical set + deterministic selection

The HF layout stores the same template string in `metadata.json → subsets[<subset>].prompt_template`. At eval time Simple-MMEval re-renders the template against the **current message dict** (one user message per row); the dict context is exactly the fields below.

This file defines **one canonical template per task type**. When the table at the bottom forces a choice between two templates, the selection rule is deterministic — based on observable fields, not judgment. The flow is:

1. **Is there an official model-input prompt** for this dataset (paper appendix / official eval-code's prompt-construction function, e.g. `construct_prompt` / `build_prompt` / `doc_to_text`, or dataset card)? If yes, **copy it byte-for-byte** into `prompt_template`. Do not use any canonical fallback template below. Record `prompt_template_source.origin = "official"` and cite the source path + line numbers in `prompt_template_source.reference`.
2. **Does the source dataset ship a full per-row prompt column** (e.g. MathVista `query`, MathVerse `query_wo`, VisOnlyQA `prompt_no_reasoning`, CharXiv per-category instructions)? If yes, use that source-provided prompt instead of falling through to a canonical fallback template:
   - **2.1** If the source provides both the full prompt and its component fields (`question`, `options`, `hint`, etc.), reverse-engineer a Jinja template that renders byte-identical to the source prompt on sampled rows.
   - **2.2** If the source provides only the full prompt string, map it to a pass-through field and use the one-key template `{{ prompt }}` (or the mapped field name).
   Record `prompt_template_source.origin = "source_column"` and the source column name in `prompt_template_source.reference`.
3. **Otherwise** (no authoritative prompt and no source-provided full prompt column): pick exactly one canonical fallback template from the selection table.

Priority 1 and 2.2 differ by where the prompt is defined. Priority 1 uses a prompt specification outside the data (paper, official eval code, or dataset card) and transcribes it into a real Jinja template. Priority 2.2 forwards an already-materialized prompt column from the data with `{{ prompt }}`. When both exist, priority 1 wins; the official published specification is more authoritative than a possibly third-party rendered column.

The fallback set deliberately tracks VLMEvalKit canonical wording (`open-compass/VLMEvalKit`) — its `ImageMCQDataset.build_prompt` / `ImageVQADataset.build_prompt` / `ImageBaseDataset.build_prompt` defaults are the de-facto ecosystem baseline.

## Visible fields on the message dict

- `role` — always `"user"` for converter output.
- `question` — string (`""` if absent in the source).
- `answer` — string, or list when `--answer-list` is set (`""` / `[]` if absent).
- `hint` — string (`""` if absent).
- `options` — dict (`{}` if absent). Source list options like `["cat", "dog"]` are normalized to `{"A": "cat", "B": "dog"}`, so MCQ templates can iterate `options.items()` regardless of source shape.
- `choices` — list (`[]` if absent).
- `n_images` — integer (set only when the dataset has variable image counts per row, e.g., OlympiadBench / Mantis / ENEM).
- Any extra `--map foo=bar` pair lands as `foo` (and only when the source value is non-`None`).

Renderer globals: `zip`, `enumerate`, `len`, `range`, `list`, `dict`, `str`, `int`, `float`, `bool`, `sum`, `max`, `min`.

## Canonical templates

### T1 — Short-answer VQA, single image

```jinja
<image>{{ question }}
Answer the question using a single word or phrase.
```

Default for free-form VQA (single image) when the question carries no answer-format cue and the dataset is not judge-scored.

### T2 — Bare question (one media file)

```jinja
<image>{{ question }}
```

Use **instead of T1** when any of the deterministic triggers in the selection table applies (the question already carries the instruction, or the benchmark is on the judge-scored allow-list, or the question already contains `<image>` placeholders). Replace `<image>` with `<video>` for video sources.

### T3 — MCQ with `options` dict, single image

```jinja
<image>{% if hint %}Hint: {{ hint }}
{% endif %}Question: {{ question }}
Options:
{% for k, v in options.items() %}{{ k }}. {{ v }}
{% endfor %}Please select the correct answer from the options above. 
```

This is VLMEvalKit `vlmeval/dataset/image_mcq.py:ImageMCQDataset.build_prompt`. Preserve the trailing space + newline after `above.` byte-for-byte. Option format is `A. opt` (period + space) — not `(A) opt`, not `A: opt`.

### T4 — MCQ with `options` dict, multi-image (images at head)

```jinja
{% for _ in range(n_images) %}<image>{% endfor %}{{ question }}
Options:
{% for k, v in options.items() %}{{ k }}. {{ v }}
{% endfor %}Please select the correct answer from the options above. 
```

Same trailer as T3. Requires `n_images` on the message dict and equal to `len(media)` per row.

### T5 — Captioning, single image

```jinja
<image>Provide a one-sentence caption for the provided image.
```

The `question` field is ignored at render time — captioning prompts are synthetic.

### T6 — Video MCQ

```jinja
<video>{{ question }}
Answer with the option's letter from the given choices directly.
```

For video benchmarks without an official model-input prompt; aligns with Video-MME's `post_prompt`.

### T7 — Text-only short-answer

```jinja
{{ question }}
Answer the question using a single word or phrase.
```

Use for text-only subsets / configs (no media). When the text question already carries an instruction, fall back to bare `{{ question }}` (the text-only analogue of T2).

### T8 — Video open-ended VQA

```jinja
<video>{{ question }}
Answer the question using a single word or phrase.
```

For video benchmarks with free-form answers (ActivityNet-QA, MSVD-QA, MSRVTT-QA). Same structure as T1 but with `<video>` placeholder.

### T9 — Bare video question

```jinja
<video>{{ question }}
```

Video analogue of T2. Use when the question already carries an answer-format instruction, or the benchmark is judge-scored. Replaces T2's `<image>` with `<video>`.

### T10 — Video MCQ with `options` dict

```jinja
<video>{% if hint %}Hint: {{ hint }}
{% endif %}Question: {{ question }}
Options:
{% for k, v in options.items() %}{{ k }}. {{ v }}
{% endfor %}Please select the correct answer from the options above. 
```

Video analogue of T3 (VLMEvalKit-style MCQ). Use when the video benchmark provides structured options but no official prompt. Preserve the trailing space + newline after `above.`.

## Selection table — exactly one template per dataset/subset

Evaluate the rows top-to-bottom. The first row whose **condition** is true selects the template. The condition column is checked against observable per-row fields and a short, finite allow-list — no judgment calls.

| # | Condition (top-down, first match wins) | Template |
|---|---|---|
| 1 | An official model-input prompt is documented in the dataset's paper / official eval-code prompt-construction function / dataset card | **Copy official byte-for-byte** (not a canonical fallback template) |
| 2 | The source dataset ships a full per-row prompt column | **Reverse-engineer Jinja byte-for-byte** (2.1) or use a **one-key pass-through template** such as `{{ prompt }}` (2.2) |
| 3 | Task is captioning (`task_type == "captioning"`) | **T5** |
| 4 | Any media item is a video AND `options` is non-empty AND the question already carries the answer-format instruction or benchmark is judge-scored | **T9** (bare video question) |
| 5 | Any media item is a video AND `options` is non-empty AND dataset provides structured options dict | **T10** (video MCQ with options dict) |
| 6 | Any media item is a video AND `options` is non-empty (simple case) | **T6** (video MCQ) |
| 7 | Any media item is a video AND `options` is empty AND question carries instruction or benchmark is judge-scored | **T9** (bare video question) |
| 8 | Any media item is a video AND `options` is empty (open-ended video QA) | **T8** (video open-ended VQA) |
| 9 | Media count is zero (text-only dataset/subset, `--modalities text`) | **T7** |
| 10 | The dataset is on the judge-scored allow-list **OR** the `question` field already carries the answer-format instruction (see the precise triggers below) | **T2** |
| 11 | `options` is a non-empty dict AND per-row `n_images > 1` | **T4** |
| 12 | `options` is a non-empty dict AND per-row media count is 1 | **T3** |
| 13 | Otherwise (single-image free-form, no inline instruction) | **T1** |

### Precise triggers for row 10 (T2)

T2 is selected when **either** of these is true:

**(a) judge-scored allow-list.** The dataset is one of:
- `MM-Vet`
- `LLaVA-Bench-in-the-Wild`
- `MMHal-Bench`
- `WildVision-Bench`
- `VibeEval`

These benchmarks score with a GPT-4 / GPT-4o judge that expects free-form output; appending a short-answer cue truncates responses and lowers scores. New judge-scored benchmarks added to this skill must be added to this list explicitly.

**(b) instruction already inside `question`.** A representative sample of rows from the source `question` column contains at least one of:
- `Please answer directly` / `Please answer the question with`
- `Answer the question with` / `Answer the question using`
- `Answer Yes or No` / `Answer with yes or no`
- `True or False?`
- `Please select the correct` / `Select the best answer`
- `Answer with the option` / `Respond with only the letter`
- The literal answer space inlined (e.g., `(A) ... (B) ...` or `A. ... B. ...` already in the question)
- A `<image>` / `<video>` placeholder already in the question text (any case — bare question keeps placeholder/media count consistent)

If any one of these strings is present in a row, T2 applies to that subset. Confirm by running the smell test in the bottom of this file (no string appearing twice in the rendered prompt).

## Per-row preamble switching (multi-config datasets)

For benchmarks whose official prompt varies by per-row attributes (subject, language, problem type — e.g., OlympiadBench's 18 splits over `OE`/`TP × maths/physics × en/zh`), the discriminating fields **must live on the per-row message dict** (typical names: `subject`, `language`, `is_theorem_proving`, `answer_type`). Map them via `--map`. Then write the template as a Jinja switch on those fields:

```jinja
{% for _ in range(n_images) %}<image>{% endfor %}{% if language == 'English' %}<official English preamble>{% else %}<official Chinese preamble>{% endif %}
{{ question }}
```

Jinja templates render against the message dict only — they cannot see the split name. If the source does not carry the discriminating field, re-process the source to add it before uploading.

## Non-English prompts

When the **official source publishes the prompt in a non-English language** (e.g., MMBench-CN, CCBench, ENEM Portuguese), copy that wording verbatim — translation or paraphrase changes reported scores. The skeleton stays the same as T3/T4 (Question:/Options:/A. opt/<trailing instruction>); only the instruction literals change.

The canonical fallback templates above are English-only. There is no language-aware fallback set — when no official source exists for a non-English benchmark, write the template with the same shape as T3/T4 and instruction literals translated by the dataset's authors. Record that translation source in `prompt_template_source.reference`; do not invent translations.

## Smell tests (run before push)

```python
import re
from jinja2 import Environment
env = Environment()
env.globals.update({"zip": zip, "enumerate": enumerate, "len": len, "range": range,
                    "list": list, "dict": dict, "str": str, "int": int, "float": float,
                    "bool": bool, "sum": sum, "max": max, "min": min})
rendered = env.from_string(TEMPLATE).render(question="What is this?",
                                            options={"A": "car", "B": "boat"}, hint="")

# 1. Placeholder count equals row-level media count
assert len(re.findall(r"<(image|video)>", rendered)) == len(media_list)

# 2. No duplicated trailer / Options block
for needle in ("Options:", "Please select", "Answer with the option",
               "Answer the question using"):
    assert rendered.count(needle) <= 1, f"duplicate {needle!r} in rendered prompt"
```

The `validate.py --audit` and `smoke_run.py` scripts re-run the placeholder-count check at converter-output time and at evaluation time respectively. The duplicate-trailer check is the most-common audit failure mode this skill has surfaced — keep it in the loop.
