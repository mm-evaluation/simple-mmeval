# `metadata.json` authoring guide (mm-eval / Simple-MMEval HF datasets)

This file lives at the **repository root** next to the `data/` tree (or alongside `hf_dataset/` before push). Simple-MMEval's HF loader downloads `metadata.json` via `hf_hub_download` and reads the Jinja template from `subsets[<subset>].prompt_template`. This root-level manifest is the current format; it replaces the deprecated sibling HF `metadata` DatasetDict config that older repos shipped.

The converter (`scripts/convert.py`) can **emit** `metadata.json` automatically from `--map` / inferred stats, or **consume** a draft via `--metadata-json` (to derive `--map` + default template, then refresh media counts).

## Top-level schema

| Field | Required | Description |
| --- | --- | --- |
| `name` | yes | Human-readable dataset name (matches paper title or official benchmark name, not the HF repo slug). |
| `release_date` | yes | `YYYY-MM-DD` string. |
| `subsets` | yes | Map of subset name → per-subset metadata. Most repos use a single key `main`. |

## Per-subset object (`subsets[name]`)

| Field | Required | Description |
| --- | --- | --- |
| `language` | yes | List of BCP-47 language tags, e.g. `["en"]`, `["en", "zh"]`. |
| `modalities` | yes | See **Modalities** below. |
| `task_type` | yes | See **Task types** below. |
| `prompt_template` | yes | Jinja string. Must render from the **mmeval message dict** keys present in `messages[0]` (`question`, `options`, `hint`, `answer`, plus any pass-through fields). |
| `prompt_template_source` | yes | Structured provenance for `prompt_template`; see **Prompt template source** below. |
| `mapping_from_source` | yes | Describes how **source rows** map into message fields + HF `media` column. |

## Prompt template source

Every subset must explain where its `prompt_template` came from:

```json
"prompt_template_source": {
  "origin": "official",
  "reference": "https://github.com/example/repo/path.py#L10-L25",
  "notes": "Copied byte-for-byte from the official eval-code prompt-construction function."
}
```

Use only these `origin` values:

| `origin` | When to use | `reference` convention |
| --- | --- | --- |
| `official` | A paper appendix, official eval-code prompt-construction function (e.g. `construct_prompt` / `build_prompt` / `doc_to_text`), or dataset card publishes the model-input prompt. | URL/path + line range, paper section, or dataset-card URL. |
| `source_column` | The source dataset ships a full per-row prompt column. Prefer reverse-engineering a byte-identical Jinja template when component fields are also available; otherwise use a one-key pass-through template such as `{{ prompt }}`. | Source prompt column name, e.g. `query_wo` or `prompt_no_reasoning`. |
| `fallback` | No official prompt and no source-provided full prompt column exists, so the template follows `references/jinja-templates.md`. | Canonical fallback template id, e.g. `T1`, `T3`, `T8`. When the shipped template is based on a canonical `Tn` but modified (e.g. an added domain instruction), use `<Tn>-adapted` (e.g. `T3-adapted`) and explain the change in `notes`. |

`notes` is optional free text for details such as `FALLBACK class=A`, "2.1 reverse-engineered from source prompt", or "translated instruction literals copied from the authors' dataset card". Do not put prompt provenance in a separate top-level free-text notes object; it belongs in this per-subset object.

## Modalities

Use **only** the following tags. List every tag that applies to at least one row in the subset.

| Tag | When to use |
| --- | --- |
| `single_image_start` | Every image row has exactly one image, placed **before** the question text in the rendered prompt. |
| `single_video_start` | Every video row has exactly one video, placed before the question text. |
| `multi_image_start` | Some rows have two or more images, **all placed before** the question text (e.g. `{% for … %}<image>{% endfor %}{{ question }}`). |
| `multi_image_interleave` | Images are interleaved **within** the question text (e.g. `<image>question part A<image>question part B`). |
| `multi_video_interleave` | Videos are interleaved within the question text. |
| `multi_image_video_interleave` | A row mixes images and videos interleaved with text. |
| `text` | Some rows have **zero** media items (text-only questions). |

**Rules:**
- Include `text` only when at least one row genuinely has no media (e.g. mixed-modality benchmarks like ScienceQA or ENEM). Do **not** add `text` simply because the modality type is language.
- When a dataset has rows with 1 image and rows with 2+ images (both at start), use `multi_image_start` — it subsumes the single-image case when max > 1.
- `single_image_start` and `multi_image_start` are mutually exclusive per subset; use whichever matches the actual data.
- For video datasets, `single_video_start` is the most common tag (one video per question). Use `multi_video_interleave` only when multiple videos appear interleaved with question text in the same row.

## Task types

Use **only** the following values:

| Value | When |
| --- | --- |
| `vqa` | Free-form visual question answering — the model generates a natural-language answer. Covers open-ended, numeric, short-answer, and yes/no formats. |
| `multiple_choice_vqa` | The correct answer is one of a discrete set of labelled options (A / B / C …), whether the options appear in a separate `options` dict or are embedded in the question text. |
| `captioning` | Image / video captioning — no explicit question; the model produces a free-form caption. The prompt template is canonical T5 in `jinja-templates.md` (`<image>Provide a one-sentence caption for the provided image.`) and the reference `answer` is a string or list of reference captions used by BLEU/CIDEr/SPICE scorers. |

When a dataset mixes both formats across rows (e.g. some rows are MCQ, others are free-form), use `vqa` if the majority are free-form, or `multiple_choice_vqa` if the majority are MCQ.

## `video_storage` (video datasets only)

For subsets where any modality tag contains `video`, include a `video_storage` block describing how video files are stored and accessed. This block is optional for image-only datasets.

| Field | Required | Type | Description |
| --- | --- | --- | --- |
| `format` | yes | string | `"files"` (loose video files in a directory) or `"archives"` (zip/tar that must be extracted before loading). |
| `media_root` | yes | string | Directory containing video files, relative to the artifact/repo root. Typically `"media"` (local mode) or `"videos"` (HF mode). |
| `archive_format` | no | string or null | `"zip"`, `"tar"`, `"tar.gz"`, or `null`. Required when `format == "archives"`. |
| `archive_files` | no | list of string | Archive filenames when `format == "archives"`. Empty list otherwise. |
| `notes` | no | string | Free-text notes: download instructions, licensing caveats, frame sampling recommendations, known missing videos. |

Example:
```json
"video_storage": {
  "format": "files",
  "media_root": "media",
  "archive_format": null,
  "archive_files": [],
  "notes": "Videos downloaded from YouTube via yt-dlp; 12 videos unavailable due to takedowns."
}
```

When `format == "archives"`:
```json
"video_storage": {
  "format": "archives",
  "media_root": "videos",
  "archive_format": "zip",
  "archive_files": ["videos_001.zip", "videos_002.zip"],
  "notes": "Extract all archives into videos/ before running evaluation."
}
```

## `mapping_from_source`

### `source` block

| Field | Required | Description |
| --- | --- | --- |
| `format` | yes | One of `huggingface`, `tsv`, `csv`, `json`. |
| `url` | yes | **Dict mapping each uploaded split name → source URL.** Use the most specific URL available (HF dataset page, GitHub data directory, …). When multiple splits came from the same source, repeat the URL for each key. Do **not** add a top-level `repo` or separate `links` field — `url` is the only provenance field. |

### Standard field entries

Each mapping entry is either:

- `{"from": "<source-column>", "optional": true?}` for scalar fields (`id`, `question`, `answer`, `hint`, `options`), or
- `"choices": [{"key":"A","from":"A","optional":?}, …]` for MCQ columns stored as separate source columns (MMBench-style), or
- `"choices": {"from": "<list-column>"}` when a single column already holds list choices, or
- `"media": {"from": "<column>", "type": "list", "min_items": <int>, "max_items": <int>}` — **min/max are recomputed by `convert.py`** from surviving rows.

> **Do not add a `choices` mapping unless the source dataset actually provides a choices field.** Adding it for datasets where options are extracted from question text fabricates a false provenance claim.

### `extra`

Any additional `--map foo=bar` pass-through fields:

```json
"extra": { "category": {"from": "category"} }
```

### `id` vs `source_id` (composite-id convention)

When the source's `id` field is not row-unique (e.g., one chart UUID shared by multiple QA paraphrases), `convert.py` rewrites each duplicate to `{source_id}_q{k}` and adds a `source_id` field on the per-row message dict. The metadata.json reflects this with two mappings:

```json
"id": {
  "from": "id",
  "note": "Composite key {source_id}_q{occurrence_index} produced by walking rows in upstream order and counting per-source-id occurrences (1-indexed)."
},
"extra": {
  "source_id": { "from": "id", "note": "Original upstream id before composite-id formation." }
}
```

This is automatic — the converter emits both mappings when disambiguation fires. `validate.py --audit` recognizes the convention and reports `composite-id convention active: N/M source_id(s) span >1 row` so the operator sees at a glance whether the dataset is 1-row-per-source or multi-row-per-source.

## Source `format` decision table

| Source you read | `format` |
| --- | --- |
| `datasets.load_dataset(…)` | `huggingface` |
| Pre-downloaded `*.json` list of dicts | `json` |
| `*.tsv` + `csv.DictReader(…, delimiter='\t')` | `tsv` |
| `*.csv` | `csv` |

Use `json` whenever you pre-downloaded the source to a local file before running `convert.py`, even if the original source is a HuggingFace dataset.

## Runtime selection (Simple-MMEval)

- **Single subset** in `metadata.json`: `--dataset mmeval_hf@org/name` is enough.
- **Multiple subsets**: keep `--dataset mmeval_hf@org/name` and pass `--subset subset_key` (wired through `DataArguments`).

HF **data split** (`train` / `val` / …) is still `--split` and refers to the `default` config on the Hub, **not** the subset key inside `metadata.json`.

## Agent workflow

1. `python3 scripts/inspect_source.py --hf … --split …` (or `--json …`) to discover column names/types.
2. Draft `metadata.json` focusing on a faithful `prompt_template` + `mapping_from_source` (`source.url` can be filled after conversion).
3. `python3 scripts/convert.py --metadata-json draft.json --mode hf --out …` (optional `--map` overrides). On this path the converter **only refreshes** `media.min_items` / `max_items`, `source.url` (merged for the current split), and `modalities` (unless `--modalities` overrides); **everything else** in the seed is **preserved**.
4. Run `python3 scripts/validate.py --audit <out>/hf_dataset` to catch schema errors (invalid modalities, stale `repo`/`links` fields, placeholder/media mismatches) **before pushing**.
5. `python3 scripts/push_to_hf.py --artifact-dir … --repo-id …`.

## Example — MMBench-style TSV (single image MCQ)

```json
{
  "name": "MMBench-en-V11",
  "release_date": "2026-03-25",
  "subsets": {
    "main": {
      "language": ["en"],
      "modalities": ["single_image_start"],
      "task_type": "multiple_choice_vqa",
      "prompt_template": "<image>{% if hint %}Hint: {{ hint }}\n{% endif %}Question: {{ question }}\nOptions:\n{% for k, v in options.items() %}{{ k }}. {{ v }}\n{% endfor %}Please select the correct answer from the options above. \n",
      "prompt_template_source": {
        "origin": "fallback",
        "reference": "T3",
        "notes": "VLMEvalKit canonical MCQ fallback."
      },
      "mapping_from_source": {
        "source": {
          "format": "tsv",
          "url": { "test": "https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_TEST_EN_V11.tsv" }
        },
        "id": {"from": "index"},
        "question": {"from": "question"},
        "choices": [
          {"key": "A", "from": "A"},
          {"key": "B", "from": "B"},
          {"key": "C", "from": "C", "optional": true},
          {"key": "D", "from": "D", "optional": true}
        ],
        "hint": {"from": "hint", "optional": true},
        "answer": {"from": "answer", "optional": true, "note": "Empty for unlabeled test splits."},
        "media": {"from": "image", "type": "list", "min_items": 1, "max_items": 1}
      }
    }
  }
}
```

## Example — HF single-split free-form VQA (VizWiz val)

```json
{
  "name": "VizWiz-VQA",
  "release_date": "2026-04-30",
  "subsets": {
    "main": {
      "language": ["en"],
      "modalities": ["single_image_start"],
      "task_type": "vqa",
      "prompt_template": "<image>{{ question }}\nAnswer the question using a single word or phrase.",
      "prompt_template_source": {
        "origin": "fallback",
        "reference": "T1",
        "notes": "Canonical single-image short-answer VQA fallback."
      },
      "mapping_from_source": {
        "source": {
          "format": "huggingface",
          "url": { "val": "https://huggingface.co/datasets/lmms-lab/VizWiz-VQA" }
        },
        "id": {"from": "question_id"},
        "question": {"from": "question"},
        "media": {"from": "image", "type": "list", "min_items": 1, "max_items": 1},
        "answer": {"from": "answers", "optional": true}
      }
    }
  }
}
```

## Example — Mixed-modality MCQ (ScienceQA: image + text-only rows)

```json
{
  "name": "ScienceQA",
  "release_date": "2022-09-20",
  "subsets": {
    "main": {
      "language": ["en"],
      "modalities": ["single_image_start", "text"],
      "task_type": "multiple_choice_vqa",
      "prompt_template": "{% if has_image %}<image>{% endif %}{% if hint %}Context: {{ hint }}\n{% endif %}Question: {{ question }}\nOptions:\n{% for k, v in options.items() %}({{ k }}) {{ v }}\n{% endfor %}Answer with the option letter from the given choices.",
      "prompt_template_source": {
        "origin": "fallback",
        "reference": "T3-adapted",
        "notes": "ScienceQA-style context MCQ fallback using source hint/options fields."
      },
      "mapping_from_source": {
        "source": {
          "format": "json",
          "url": {
            "train":      "https://huggingface.co/datasets/derek-thomas/ScienceQA",
            "validation": "https://huggingface.co/datasets/derek-thomas/ScienceQA",
            "test":       "https://huggingface.co/datasets/derek-thomas/ScienceQA"
          }
        },
        "id": {"from": "id"},
        "question": {"from": "question"},
        "hint": {"from": "hint", "optional": true},
        "options": {"from": "choices", "optional": true, "note": "list normalized to {A,B,...} dict"},
        "answer": {"from": "answer", "optional": true},
        "media": {"from": "image", "type": "list", "min_items": 0, "max_items": 1},
        "extra": { "has_image": {"from": "has_image"} }
      }
    }
  }
}
```

## Example — Multi-subset dataset (MathVerse: image + text-only splits)

```json
{
  "name": "MathVerse",
  "release_date": "2024-03-01",
  "subsets": {
    "main": {
      "language": ["en"],
      "modalities": ["single_image_start"],
      "task_type": "vqa",
      "prompt_template": "<image>{{ question }}\n",
      "prompt_template_source": {
        "origin": "source_column",
        "reference": "query_wo",
        "notes": "Source column already carries the benchmark prompt text; template only prepends the image placeholder."
      },
      "mapping_from_source": {
        "source": {
          "format": "huggingface",
          "url": { "testmini": "https://huggingface.co/datasets/AI4Math/MathVerse" }
        },
        "id": {"from": "sample_index"},
        "question": {"from": "query_wo"},
        "media": {"from": "image", "type": "list", "min_items": 1, "max_items": 1},
        "answer": {"from": "answer", "optional": true}
      }
    },
    "text_only": {
      "language": ["en"],
      "modalities": ["text"],
      "task_type": "vqa",
      "prompt_template": "{{ question }}\n",
      "prompt_template_source": {
        "origin": "source_column",
        "reference": "query_wo",
        "notes": "Text-only variant of the source-provided prompt column."
      },
      "mapping_from_source": {
        "source": {
          "format": "huggingface",
          "url": { "testmini_text_only": "https://huggingface.co/datasets/AI4Math/MathVerse" }
        },
        "id": {"from": "sample_index"},
        "question": {"from": "query_wo"},
        "answer": {"from": "answer", "optional": true}
      }
    }
  }
}
```

## Example — Video MCQ (Video-MME style)

```json
{
  "name": "Video-MME",
  "release_date": "2024-06-01",
  "subsets": {
    "main": {
      "language": ["en"],
      "modalities": ["single_video_start"],
      "task_type": "multiple_choice_vqa",
      "prompt_template": "<video>{{ question }}\nAnswer with the option's letter from the given choices directly.",
      "prompt_template_source": {
        "origin": "fallback",
        "reference": "T6",
        "notes": "Canonical video MCQ fallback aligned with Video-MME-style post prompts."
      },
      "video_storage": {
        "format": "files",
        "media_root": "media",
        "notes": "900 videos downloaded from YouTube. 12 unavailable due to takedowns (skipped in conversion)."
      },
      "mapping_from_source": {
        "source": {
          "format": "huggingface",
          "url": { "test": "https://huggingface.co/datasets/lmms-lab/Video-MME" }
        },
        "id": { "from": "question_id" },
        "question": { "from": "question" },
        "options": { "from": "options", "note": "list normalized to {A,B,...} dict" },
        "answer": { "from": "answer", "optional": true },
        "media": { "from": "videoID", "type": "list", "min_items": 1, "max_items": 1 },
        "extra": {
          "video_id": { "from": "videoID" },
          "duration_category": { "from": "duration" }
        }
      }
    }
  }
}
```

## Example — Video open-ended QA (ActivityNet-QA style)

```json
{
  "name": "ActivityNet-QA",
  "release_date": "2019-06-01",
  "subsets": {
    "main": {
      "language": ["en"],
      "modalities": ["single_video_start"],
      "task_type": "vqa",
      "prompt_template": "<video>{{ question }}\nAnswer the question using a single word or phrase.",
      "prompt_template_source": {
        "origin": "fallback",
        "reference": "T8",
        "notes": "Canonical video open-ended VQA fallback."
      },
      "video_storage": {
        "format": "files",
        "media_root": "media",
        "notes": "Videos from ActivityNet v1.3 (YouTube). Some unavailable due to takedowns."
      },
      "mapping_from_source": {
        "source": {
          "format": "huggingface",
          "url": { "test": "https://huggingface.co/datasets/lmms-lab/ActivityNetQA" }
        },
        "id": { "from": "question_id" },
        "question": { "from": "question" },
        "answer": { "from": "answer", "optional": true },
        "media": { "from": "video_name", "type": "list", "min_items": 1, "max_items": 1 },
        "extra": {
          "video_id": { "from": "video_name" },
          "question_type": { "from": "type" }
        }
      }
    }
  }
}
```

## Common pitfalls

- **`prompt_template` must match placeholders** (`<image>` / `<video>`) with the emitted HF `media` sequence (see `mmeval-format.md`). Run `validate.py --audit` to catch mismatches before push.
- **HF video mode stores video paths as strings, not bytes.** The HF `media` column uses `Sequence(Value('string'))` for video datasets (not `Sequence(Image())`). The dataloader resolves these paths to local files at runtime. Use `--mode local` for the primary artifact; use HF video mode only for distribution.
- **`source.url` is not a substitute for HF `--split`**. `url` is documentation; the Arrow split name is whatever you passed to `convert.py --split` (or `train` when omitting `--split` on local sources).
- **Never add a `choices` mapping unless the source truly provides a choices column.** For benchmarks where options are embedded in the question text or extracted by the converter, document them under `options` only.
- **`source.url` keys must match the uploaded split names exactly.** For multi-subset datasets, each subset's `url` should only reference that subset's splits — do not copy keys from a sibling subset.
- **Do not add `repo` or `links` fields** — they are superseded by `source.url`.
- **Include `video_storage` for video subsets.** The block is required when any modality tag contains `video`. Omitting it causes the dataloader to fall back to default path resolution, which may fail for archived or non-standard layouts.
- **Archive extraction is a manual step.** The dataloader does not extract archives automatically. Document the extraction command in `video_storage.notes` and in the dataset card.
