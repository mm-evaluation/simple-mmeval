---
name: dataset-upload
description: Convert any multimodal benchmark (HuggingFace dataset, local JSON, CSV/TSV with images) into Simple-MMEval runnable format and upload it to the HuggingFace Hub. Use this skill whenever the user wants to prepare/normalize/port a vision-language eval dataset for Simple-MMEval, run an existing benchmark through Simple-MMEval, build a `local@json` data file with an `img_dir`, build an `mmeval_hf@`-compatible HF dataset (canonical `media` / `messages` / `id` columns plus a top-level `metadata.json` manifest with the prompt template + source mapping), or push a converted dataset to HuggingFace Hub. Trigger even if the user does not explicitly say "Simple-MMEval" — phrases like "convert this VQA dataset", "make this runnable in mmeval", "wrap this for evaluation", "turn this into an eval dataset", or "push this benchmark to HF for mmeval" all apply.
---

# dataset-upload

Convert a multimodal eval dataset into one of two runnable input formats accepted by [Simple-MMEval](https://github.com/mm-evaluation/simple-mmeval):

1. **`local`** — a single JSON file plus a directory of media files (`--dataset local@json --infile <data.json> --img_dir <media_dir>`).
2. **`hf`** — a HuggingFace `DatasetDict` for the `default` config plus a top-level `metadata.json` manifest at the repo root (`--dataset mmeval_hf@<user>/<repo>`; pass `--subset <name>` when the manifest packs multiple subsets). The legacy `metadata` config (a sibling DatasetDict) is no longer used; the top-level `metadata.json` manifest replaces it.

The converter can emit one mode or **both at once in a single source pass** (`--mode both`), which is the recommended path when the user wants the local artifact for quick smoke runs *and* an HF artifact to push.

## Dependencies

```bash
pip install -r requirements.txt
# or, equivalently:
pip install datasets huggingface_hub jinja2 Pillow requests
```

If `push_to_hf.py` errors with `Mask must be a pyarrow.Array of type boolean`, upgrade `datasets` to the latest stable release first (this comes from Arrow/`embed_storage` edge cases in some versions). If it still reproduces on your shard layout, pin to a version your environment has verified — for example `pip install 'datasets<4.6'` — rather than assuming one pin fits all machines.

## Files in this skill

```
dataset-upload/
├── SKILL.md                   ← you are here
├── references/
│   ├── mmeval-format.md       ← exact target schema (image + video)
│   ├── metadata-json.md       ← how to author metadata.json (incl. video_storage)
│   ├── jinja-templates.md     ← copy-paste templates (T1–T10, incl. video T6/T8/T9/T10)
│   └── video-datasets.md     ← survey of video eval benchmarks + mm-eval video design
└── scripts/
    ├── inspect_source.py            ← peek source schema (HF or local JSON)
    ├── convert.py                   ← generic conversion (one split at a time)
    ├── convert_indexed_multimage.py ← MMMU-style: <image N> refs + per-subject configs
    ├── merge_splits.py              ← merge per-split HF artifacts into one DatasetDict
    ├── validate.py                  ← data-loader round-trip + video file checks (no model)
    ├── smoke_run.py                 ← end-to-end smoke: tiny model + N rows through Simple-MMEval
    ├── push_to_hf.py                ← final HF push, single-`default`-config (legacy single-subset path)
    ├── push_to_hf_multiconfig.py    ← multi-config HF push, one HF config per metadata.json subset (preferred for multi-subset datasets)
    ├── cleanup.py                   ← remove intermediate artifacts and HF cache after push
    └── test_audit_regressions.py    ← internal regression tests for convert.py + validate.py --audit
```

> The inspect script is named `inspect_source.py` rather than `inspect.py` because the latter shadows Python's stdlib `inspect` module when run from the same directory and breaks pandas import.

## Resource Management

Image benchmarks involve large data volumes. Follow these practices to avoid disk exhaustion and memory pressure throughout every conversion.

### Project `.tmp/` layout (where ephemeral work lives)

All conversion-related I/O lives under `<project-root>/.tmp/`. The layout is fixed so disk audits and bulk cleanup are predictable:

```
<project-root>/.tmp/
├── conversions/<dataset>/    ← per-dataset HF cache + temp (auto-created by convert.py --work-dir)
│   ├── hf/                   ← HF_HOME (datasets/, hub/)
│   └── tmp/                  ← TMPDIR / TEMP / TMP
├── smoke_tests/<slug>/       ← Step 5b/6b smoke artifacts (preserved by cleanup.py)
│   ├── smoke_metadata.json
│   ├── smoke_summary.json
│   └── <subset>/<split>/{result.json, run.log}
├── upload_work/              ← active per-session orchestration (driver scripts + manifests)
├── audits/<YYYY-MM-DD>/      ← dated audit reports + per-dataset fix bundles
├── manifests/                ← canonical reference docs (e.g. eval_datasets.csv)
└── archive/                  ← quarantined items kept for human review only
```

Rules every contributor (human or agent) must follow:

- **Never write files at `.tmp/` root.** Use one of the named subdirs above. If none fits, create a new top-level bucket and document it in `.tmp/README.md` — don't drop loose files.
- **`--out` basenames must be dataset-specific.** Generic names like `val`, `test`, `train`, `dev`, `img`, `txt`, or a language code share one work-dir across unrelated runs and accumulate misnamed multi-GB HF caches. Always pass `--out $WORK/<DatasetName>` (or per-split `<DatasetName>_<split>`).
- **`smoke_tests/` is never deleted by automated cleanup.** Smoke results are the validation record; `cleanup.py` refuses to remove the tree unless `--include-smoke-results` is passed.
- **`upload_work/`, `audits/`, `manifests/`, and `archive/` are preserved** by automated cleanup. Move anything you want a human to look at later into `archive/` rather than deleting it.

### Pre-flight: check available disk space and route temp/cache to project `.tmp`

Before starting any conversion, check free space on the partition that will hold the output and the HF download cache:

```bash
df -h /path/to/output   # check the output artifact partition
df -h "$(git rev-parse --show-toplevel)/.tmp"  # (or whatever filesystem the project's .tmp lives on)
```

A benchmark conversion needs roughly **source_size + output_size** of free space simultaneously. For image datasets, output ≈ source (images are re-compressed). As a rule: ensure at least `2 × estimated_dataset_size + 5 GB` of free space before starting.

`convert.py` automatically routes all HF download cache (`HF_HOME`, `HF_DATASETS_CACHE`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`) and system temp dirs (`TMPDIR`/`TEMP`/`TMP`) to a per-dataset subdirectory under `<project-root>/.tmp/conversions/` by default. This keeps all conversion-related I/O off `/tmp` and `~/.cache` *and* under one well-known parent so disk audits and bulk cleanup are simple:

```
<project-root>/.tmp/conversions/<out-basename>/
├── hf/          ← HF_HOME (all dataset parquet + model hub downloads)
│   ├── datasets/
│   └── hub/
└── tmp/         ← TMPDIR / TEMP / TMP
```

The converter prints the work-dir and available disk space at startup:
```
[work-dir] <project-root>/.tmp/conversions/my_dataset  (42.3/500.0 GB free/total)
```

Override the work-dir location or disable it entirely:
```bash
# Use a different base (e.g., a larger disk)
python3 scripts/convert.py ... --work-dir /mnt/large-disk/.tmp/conversions/my_dataset

# Disable (reverts to system defaults for HF cache / temp):
python3 scripts/convert.py ... --work-dir ''
```

> **Always pass `--out` a dataset-specific basename** (e.g., `--out $WORK/MyDataset`) — never a generic split or category name like `val`, `test`, `train`, `dev`, `img`, or a language code. The work-dir is derived from `Path(--out).name`, so a generic basename causes multiple conversions to share one HF cache directory and accumulate tens of gigabytes under a misnamed `.tmp/conversions/val/`-style folder.

### Datasets that require a config name

Repos with named configs (e.g., `maritaca-ai/enem` with configs `2022/2023/2024`, `Hothan/OlympiadBench` with 18 configs, `ibm-granite/ChartNet`) cannot be loaded from the default config alone. Pass the config explicitly: `convert.py --hf <repo> --hf-config <name> --split <split>` (the converter forwards it to `load_dataset(repo, config_name, split=split)`). When `--hf-config` is omitted, `load_dataset(repo, split=split)` uses the repo's default config — fine for single-config repos.

Detect configs upfront:
```python
from datasets import get_dataset_config_names
configs = get_dataset_config_names("repo/name")
# If configs != ['default'], pass --hf-config <name>.
```

Only fall back to the pre-download-to-JSON pattern below when `--hf-config` still cannot load a config directly (e.g. the rows need custom decoding before conversion). It **uses base64 JPEG instead of saving image files to disk** — separate image files accumulate fast and are equivalent in size to the JSON embed:

```python
import base64, io, json
from datasets import load_dataset
from pathlib import Path
from PIL import Image

def pil_to_b64(img: Image.Image, quality: int = 85) -> str:
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")

# Example: one config/split → one JSON
def download_split(repo: str, config: str, split: str, out_path: str) -> None:
    ds = load_dataset(repo, config, split=split)  # non-streaming for full dataset
    rows = []
    for i, row in enumerate(ds):
        img = row.get("image")
        rows.append({
            "id":       str(row.get("id", i)),
            "question": row.get("question", ""),
            "answer":   row.get("answer", ""),
            "image":    pil_to_b64(img) if img is not None else None,
            # add other fields as needed
        })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"  {len(rows)} rows → {out_path}")
```

`convert.py` reads the base64 string via `_value_to_pil` → `base64.b64decode`, so no special handling is needed in the conversion command.

**Mixed-modality rows** (some with images, some without): store `None` for absent images — **not an empty string** `""`. An empty string triggers `encode_failed` and silently drops the row:

```python
"image": pil_to_b64(img) if img is not None else None   # ✓ correct
"image": pil_to_b64(img) if img is not None else ""      # ✗ silently drops text-only rows
```

For mixed-modality datasets, also store a `has_image` integer field and use it in the template:
```jinja
{% if has_image %}<image>{% endif %}{{ question }}...
```

For datasets with variable image counts per row (0–N), store `n_images` and use `range()`:
```jinja
{% for i in range(n_images) %}<image>{% endfor %}{{ question }}...
```

Both `has_image` and `n_images` are stored in the HF message dict as integers and are available when the template is re-rendered at eval time.

### Very large datasets: use the HF rows API

For datasets where downloading the full parquet files is impractical (>10 GB), use the HuggingFace Datasets Server rows API to fetch metadata and image URLs without downloading the parquet files:

```python
import requests, json

TOKEN = "hf_..."
headers = {"Authorization": f"Bearer {TOKEN}"}

# 1. Check total rows
r = requests.get(
    "https://datasets-server.huggingface.co/rows"
    "?dataset=<repo>&config=<cfg>&split=<split>&offset=0&length=1",
    headers=headers, timeout=30,
)
total = r.json().get("num_rows_total")
print(f"Total rows: {total}")

# 2. Fetch in pages of 100, store image URL (convert.py fetches via HTTP)
rows = []
for offset in range(0, min(total, N_ROWS_NEEDED), 100):
    r = requests.get(
        f"https://datasets-server.huggingface.co/rows"
        f"?dataset=<repo>&config=<cfg>&split=<split>&offset={offset}&length=100",
        headers=headers, timeout=60,
    )
    for item in r.json()["rows"]:
        row = item["row"]
        rows.append({
            "id":    row["id"],
            "question": ...,
            "image": row["image"]["src"],  # signed HTTPS URL – convert.py fetches it
        })
with open("out.json", "w") as f:
    json.dump(rows, f, ensure_ascii=False)
```

The signed URLs expire within hours; run the conversion immediately after fetching.

**When to use this instead of streaming:**
- Dataset parquet files are >500 MB each (check: `HfFileSystem().ls("datasets/<repo>/<config>/", detail=True)`)
- Full download would exceed available disk space
- You only need a representative sample (e.g., 2000 rows from a 600k-row synthetic benchmark)

### Image format: default to JPEG

`convert.py` now defaults to `--image-format jpeg` (changed from PNG). JPEG at quality 92 is 5-10× smaller than PNG and is appropriate for all eval benchmarks except OCR benchmarks requiring pixel-perfect text rendering. Use `--image-format png` only when lossless fidelity is critical.

For pre-downloaded JSON preprocessing, always use JPEG at quality 85 (slightly lower than the converter's 92 to keep the JSON compact):
```python
img.save(buf, format="JPEG", quality=85)
```

### Video datasets

Video benchmarks (Video-MME, MVBench, EgoSchema, ActivityNet-QA, NExT-QA, PerceptionTest, LongVideoBench, MLVU, TempCompass, etc.) require special handling because video files are large and cannot be embedded in HF Arrow datasets. See `references/video-datasets.md` for a full survey of common patterns.

**Primary mode: `--mode local` (always works for video)**

```
<artifact>/
├── data.json          # annotation array
├── media/             # video files (same dir as images)
│   ├── vid_001_0.mp4
│   ├── vid_002_0.mp4
│   └── ...
└── metadata.json      # manifest with video_storage block
```

Video files are materialized (copied or downloaded) into `media/` during conversion. The `media` field in `data.json` stores basenames relative to `media/`, exactly like images. The dataloader returns the video file path as a string — it never loads video bytes into memory.

**HF video mode: `--mode hf --hf-video` (for distribution)**

When you need to push a video dataset to HuggingFace, pass `--hf-video` to store video filenames as strings in the Arrow `media` column (`Sequence(Value('string'))` instead of `Sequence(Image())`). Video files must be uploaded to the repo separately (e.g., via `huggingface_hub.upload_folder` or `git lfs push`).

```bash
# Convert with HF video mode
python3 scripts/convert.py \
    --json preprocessed/video_mme.json --media-dir /path/to/downloaded/videos \
    --map id=question_id question=question media=video_path options=options answer=answer \
    --template '<video>{{ question }}
Answer with the option'"'"'s letter from the given choices directly.' \
    --mode hf --hf-video --out $WORK/VideoMME --split test
```

**Video download strategies**

Video benchmarks distribute videos in several ways. Choose the strategy that matches the source:

1. **YouTube-sourced** (Video-MME, ActivityNet-QA, MSRVTT-QA): Use `yt-dlp` to download videos. Some will be unavailable (takedowns); log and skip them.
   ```bash
   yt-dlp -f "best[ext=mp4]" -o "%(id)s.%(ext)s" <youtube_url>
   ```

2. **Bundled on HuggingFace** (Video-MME v2, MVBench, MLVU): Download via `huggingface_hub.snapshot_download` with `allow_patterns=["*.mp4", "*.webm"]` or download archive files and extract.

3. **External cloud storage** (EgoSchema: Kaggle/Wasabi; NExT-QA: Drive): Follow the dataset's official download instructions. Check for link expiration.

4. **Archive-based** (Video-MME v2 zip files, LongVideoBench tar archives): Download archives, then extract into a flat video directory before running the converter.

**Video integrity checks**

Pass `--verify-video` to check each materialized video file:
```bash
python3 scripts/convert.py ... --verify-video
```
This checks that every video file is non-empty and (when PyAV is installed) can be opened with at least one video stream. Without `--verify-video`, only file existence and extension are checked.

Install PyAV for deeper probing:
```bash
pip install av
```

**Video ID mapping**

- Preserve the official video ID as a pass-through field via `--map video_id=<source_field>`.
- When multiple QA pairs share one video (common in video benchmarks: 3 questions per video in Video-MME, 10 per video in ActivityNet-QA), the converter auto-disambiguates to `{video_id}_q{k}` via the composite-id convention. The original video ID is preserved in `source_id` on the per-row message dict.
- Always verify that the video filename in `media` matches the actual file in `media/` — a mismatch means the mapping is broken.

**Missing / unavailable videos**

- The converter skips rows whose video cannot be materialized and counts them in `convert_summary.json` as `encode_failed:<reason>`.
- YouTube takedowns are the most common source of missing videos. Document the count and list of missing video IDs in `metadata.json → video_storage.notes`.
- Never silently drop rows. Always check `convert_summary.json:skipped` after conversion.
- For gated datasets (LongVideoBench, MLVU, EgoSchema), ensure you have accepted the license and are authenticated before downloading.

**Video metadata in `metadata.json`**

Video subsets require a `video_storage` block:
```json
"video_storage": {
  "format": "files",
  "media_root": "media",
  "notes": "900 videos from YouTube; 12 unavailable due to takedowns."
}
```
See `references/metadata-json.md` for the full `video_storage` schema.

**Video templates**

Use template T6 for video MCQ (canonical fallback), T8 for video open-ended VQA, T9 for bare video question (when question carries instruction), or T10 for video MCQ with structured options. See `references/jinja-templates.md` for the full set.

Always check whether the benchmark publishes an official prompt first — copy it byte-for-byte if available.

**Frame sampling**

Frame sampling is the responsibility of the model inference backend, not the converter or dataloader. The dataloader passes the video file path to the model as-is. Document any benchmark-specific frame sampling requirements in `video_storage.notes` (e.g., "16 frames TSN sampling" for MVBench, "max_num_frames=64" for LongVideoBench).

### Process splits sequentially on disk-constrained systems

Never convert N splits in parallel when disk is tight. Each parallel conversion downloads N copies of the HF parquet files to cache and writes N output directories simultaneously. On a constrained partition, process one split at a time:

```bash
# Instead of parallel (&) with wait:
for split in 2022 2023 2024; do
    python3 scripts/convert.py --json preprocessed/enem_${split}.json ...
    # optionally inspect convert_summary.json here before continuing
done
```

After each split is converted and you've verified its `convert_summary.json`, you can free the preprocessed JSON immediately (the Arrow output is the durable artifact):
```bash
rm preprocessed/enem_${split}.json
```

## Workflow

### Step 1 — Inspect the source

```bash
python3 scripts/inspect_source.py --hf <repo_id> --split <split>
python3 scripts/inspect_source.py --json <data.json> --media-dir <dir>
```

Use the printed columns + types + sample row to decide the column mapping with the user.

> **Copy all identifiers verbatim from official sources.** Split names, config names, and column names are case-sensitive — using the wrong case silently produces wrong results or dropped rows. Before writing any `--map` or `metadata.json`: copy the HF repo ID directly from the URL bar (e.g., `Lin-Chen/MMStar`, not `lin-chen/mmstar`); take split and config names exactly as returned by `get_dataset_split_names` / `get_dataset_config_names`; use column names exactly as printed by `inspect_source.py` — never infer them from the dataset description or paper.

### Step 2 — Decide the column mapping and template

Ask the user (or infer) which source fields correspond to:

- `id` → unique id (required). Rows with `id == None` are skipped, not given a row-index fallback.
- `image` (or `images` / `media`) → media — single value or list (required for image/video benchmarks).
- `question` → required.
- `answer` → optional (often empty for test splits).
- `options` / `choices` / `hint` → only for MCQ-style benchmarks.

Any extra `--map foo=bar` is passed through to the message dict as `foo`. Pass-through values that are `None` are dropped (so the Jinja template never sees the literal string `"None"`); only schema fields (`question / answer / hint / options / choices`) keep their empty representations (`""` / `{}` / `[]`).

If `--explode <field>` is given, each row's id is replaced with `--id-template '{id}_q{idx}'.format(id=outer, idx=j)`. Rows whose outer id is `None` skip — the converter does not invent an `outer_{global_idx}` to keep them.

Source `options` given as a list (`["cat", "dog", ...]`) are normalized to `{"A": "cat", "B": "dog", ...}` — a documented label transform, so the dict-options MCQ template in `references/jinja-templates.md` works for both shapes.

Then pick the Jinja template. The decision is deterministic — see [`references/jinja-templates.md`](references/jinja-templates.md) for the full set:

1. **If an official model-input prompt exists** for this benchmark (paper appendix / official eval-code's prompt-construction function, e.g. `construct_prompt` / `build_prompt` / `doc_to_text`, or dataset card), copy it byte-for-byte into `prompt_template`. Set `prompt_template_source.origin = "official"` and cite the source path + line numbers in `prompt_template_source.reference`. This is always the highest-priority choice.
2. **If the source dataset ships a full per-row prompt column** (e.g. MathVista `query`, MathVerse `query_wo`, VisOnlyQA `prompt_no_reasoning`, CharXiv per-category instructions), use that source-provided prompt instead of falling through to a canonical fallback template:
   - **2.1** If the source provides both the full prompt and its component fields (`question`, `options`, `hint`, etc.), reverse-engineer a Jinja template that renders byte-identical to the source prompt on sampled rows.
   - **2.2** If the source provides only the full prompt string, map it to a pass-through field and use the one-key template `{{ prompt }}` (or the mapped field name).
   Set `prompt_template_source.origin = "source_column"` and record the source column name in `prompt_template_source.reference`.
3. **Otherwise**, walk the selection table in `jinja-templates.md` top-to-bottom and pick the first canonical fallback template (T1–T10) whose condition matches. The conditions are observable: `task_type`, media kind/count, presence of `options`, and either (a) membership in the judge-scored allow-list or (b) presence of an answer-format cue inside the `question` field. Set `prompt_template_source.origin = "fallback"` and record the selected template id (for example, `T3`) in `prompt_template_source.reference`.

Priority 1 and 2.2 differ by where the prompt is defined. Priority 1 uses a prompt specification outside the data (paper, official eval code, or dataset card) and transcribes it into a real Jinja template. Priority 2.2 forwards an already-materialized prompt column from the data with `{{ prompt }}`. When both exist, priority 1 wins; the official published specification is more authoritative than a possibly third-party rendered column.

This tier decision **directly determines** `prompt_template_source`: tier 1 → `origin = "official"`, tier 2 → `origin = "source_column"`, tier 3 → `origin = "fallback"`. There is always exactly one answer, so the agent must carry it into Step 3 by passing `--template-source-origin <origin> --template-source-ref <reference>` (and `--template-source-notes` when an adaptation needs explaining) on every conversion, or by setting `prompt_template_source` in the `--metadata-json` seed. Never leave it for a manual follow-up — the audit in Step 5 rejects a missing or `unspecified` value.

The default single-image short-answer fallback (T1):

```jinja
<image>{{ question }}
Answer the question using a single word or phrase.
```

If the source question already carries an answer-format cue (e.g., the question contains `Please answer directly`, `True or False?`, `(A) ... (B) ...`, or a `<image>` placeholder), drop the trailer and use the bare-question variant T2 — `jinja-templates.md` lists the exact triggers.

### Step 3 — Convert

`scripts/convert.py` reads the source, applies the column map and template, and writes the artifact(s).

**Source flags** (pick one):
- `--hf <repo_id> --split <split>` (HF **requires** an explicit split name); add `--hf-config <name>` for multi-config repos (forwarded to `load_dataset(repo, config_name, split=split)`)
- `--json <path> [--media-dir <dir>]` | `--tsv <path>` | `--csv <path>` (local splits optional; omit `--split` for a flat HF `Dataset` on disk / Hub split `train` after push)

**Mapping flags:**
- `--map id=<src> question=<src> image=<src> answer=<src> ...` — space-separated `canonical=source` pairs (omit when using `--metadata-json` unless you want overrides).
- With `--metadata-json`, `--map` is **override-only**: extra pairs like `category=cat` merge onto the seed mapping and **do not** need to repeat `id` / `question` (those come from the template).
- `--metadata-json <path>` — load `mapping_from_source` + default `prompt_template` from a draft `metadata.json` (see `references/metadata-json.md`).
- `--name`, `--subset`, `--language`, `--task-type`, `--modalities`, `--release-date`, `--source-url`, `--source-format`, `--template-source-origin`, `--template-source-ref`, `--template-source-notes` — control emitted `metadata.json`.
- `--template <path-or-string>` — Jinja template (string or file path).
- `--prompt-prefix "<image>"` — auto-prepend the placeholder if the template doesn't include it.
- `--answer-list` — keep `answer` as a list when the source provides a list (default joins with space).
- `--answer-join " "` — join string when reducing list answers to a single string.

**Output flags:**
- `--mode local | hf | both`
- `--out <dir>` — output directory.
- `--workers N` — parallelism for image encoding (default 8; 16 is a good fit on multi-core hosts).
- `--image-format jpeg | png` — JPEG (default, 5-10× smaller) or PNG (lossless). Use PNG only when pixel-perfect fidelity is required.
- `--jpeg-quality N` — JPEG quality 1–100 (default 92).
- `--work-dir <path>` — per-dataset directory for HF caches and temp files (default: `<project-root>/.tmp/conversions/<out-basename>`). Override when `.tmp` is on a full partition; pass `''` to disable.
- `--hf-cache-dir <path>` — redirect only `HF_DATASETS_CACHE` (narrower than `--work-dir`; prefer `--work-dir` for full redirection).
- `--limit N` — cap rows for testing. With `--hf` this also flips `load_dataset(streaming=True)` so you don't download the whole benchmark.
- `--explode <field>` — flatten a list-valued field (e.g. `--explode questions` for nested-MCQ datasets like CaptionQA). Each element becomes its own row, with the inner dict's keys merged onto the outer row, and the id auto-suffixed via `--id-template '{id}_q{idx}'` (default).
- `--id-template '{id}_q{idx}'` — id format string for exploded rows.
- `--save-num-proc N` — `num_proc` forwarded to `DatasetDict.save_to_disk` for `--mode hf` / `both` (default 1).
- `--json-indent N` — pretty-print `data.json` (default: most compact JSON via `separators=(",", ":")`; pass `2` for human-readable).

For datasets with **multiple domain splits** that should land in one HF repo (e.g. CaptionQA with `natural`/`document`/`ecommerce`/`embodiedai`), run `convert.py --mode hf` once per split into separate output dirs, then combine them:

```bash
python3 scripts/merge_splits.py \
    --inputs <out>/natural <out>/document <out>/ecommerce <out>/embodiedai \
    --out <out>/merged
```

The merged directory has one `hf_dataset/` and one merged `metadata.json` ready for `push_to_hf.py`.

**Output layout:**

```
<out>/
├── data.json               # local mode (compact JSON; --json-indent 2 to format)
├── media/                  # local mode (also reused for HF bytes)
├── hf_dataset/             # HF mode — DatasetDict for the `default` config
│                           #   (named split, or flat Dataset when --split is omitted)
├── metadata.json           # always written — manifest, uploaded to repo root by push_to_hf.py
└── convert_summary.json    # rows kept/skipped + skip_reason_counts + first 50 skips
```

`convert_summary.json` records `skip_reason_counts` (`missing_required:id`, `encode_failed`, `unknown_video_ext`) and a sample of the first 50 skipped rows. Inspect it before declaring a conversion done — silently dropping 30% of a benchmark is the most common failure mode.

**Optional `metadata.json` overrides** (defaults are usually fine; see `references/metadata-json.md` for the full schema):

- `--task-type <type>` — default: inferred from the column mapping (`multiple_choice_vqa` when `options`/`choices` is mapped, otherwise `vqa`). Supported: `vqa`, `multiple_choice_vqa`, `captioning`.
- `--modalities <m> [...]` — default: inferred from the observed media counts (`single_image_start`, `multi_image_*`, `text`, etc.). Override only when the inference is wrong.
- `--release-date YYYY-MM-DD` — default: today (UTC).
- `--subset <name>` — default `main`. Use distinct names when packing multiple sub-benchmarks into one repo.
- `--source-url <url>` — default: `https://huggingface.co/datasets/<--hf>` for HF sources; otherwise omitted.
- `--name <name>` — top-level `name`. Default: `--hf` repo or basename of `--out`.
- `--language <lang> [...]` — default `["en"]` (or the seed value when `--metadata-json` is used).
- `--template-source-origin official|source_column|fallback`, `--template-source-ref <reference>`, `--template-source-notes <text>` — fill `subsets[<subset>].prompt_template_source`, which records where the `prompt_template` came from.

The `mapping_from_source` block is auto-built from `--map`: canonical keys (`id`/`question`/`answer`/`image`/`options`/...) land at the top; everything else lands under `extra:`. The `media.min_items`/`max_items` are filled in from the actual per-row counts observed during conversion.

### Step 4 — Validate splits against the source

**Before finalizing any conversion, confirm that the converted dataset's splits exactly match the source dataset's splits** — same names, same row counts. Mismatches mean samples are missing or were double-counted.

For HuggingFace sources:

```python
from datasets import get_dataset_split_names, load_dataset

# Check all configs the source exposes
from datasets import get_dataset_config_names
configs = get_dataset_config_names("<source_repo>")
# For each config, check its splits and row counts
for cfg in configs:
    for split in get_dataset_split_names("<source_repo>", cfg):
        ds = load_dataset("<source_repo>", cfg, split=split)
        print(f"config={cfg!r} split={split!r}: {len(ds)} rows")
```

For GitHub / local sources, count rows in the source JSON manually and compare:

```python
import json
with open("raw/data.json") as f:
    data = json.load(f)
print(f"source rows: {len(data)}")
# Compare against convert_summary.json
with open("<out>/convert_summary.json") as f:
    summary = json.load(f)
print(f"converted rows: {summary['rows']}, skipped: {summary['skipped']}")
```

Then check the converted artifact for each expected split:

```python
from datasets import load_from_disk
dd = load_from_disk("<out>/hf_dataset")
for split_name, ds in dd.items():
    print(f"converted split={split_name!r}: {len(ds)} rows")
```

**Checklist before proceeding:**
- [ ] Every split from the source exists in the converted dataset (no missing splits).
- [ ] Row counts match for every split (skipped rows in `convert_summary.json` are justified — only rows with `id=None`, broken images, or unknown video extensions should be dropped).
- [ ] Split names match the source exactly (e.g., `testmini` not `test`).
- [ ] For HF sources with multiple configs (e.g., `testmini` + `testmini_text_only`), all configs are represented — either as separate HF splits in one artifact or documented as intentionally excluded.
- [ ] Source field names in `metadata.json mapping_from_source` (and in every `--map` invocation) match the actual column names from `inspect_source.py` exactly — same spelling and same case.
- [ ] `metadata.json name` is the official benchmark name as it appears in the paper or dataset card — not the HF repo slug (e.g., `"MMStar"` not `"Lin-Chen/MMStar"`).
- [ ] `metadata.json modalities` uses only the supported taxonomy: `single_image_start`, `single_video_start`, `multi_image_start`, `multi_image_interleave`, `multi_video_interleave`, `multi_image_video_interleave`, `text`. No legacy values (`"image"`, `"multi_image"`, etc.).
- [ ] `metadata.json task_type` is one of: `vqa` (free-form answer), `multiple_choice_vqa` (discrete option selection), `captioning` (free-form caption with no explicit question). No ad-hoc values.
- [ ] `metadata.json source` contains **only** `format` and `url`. No `repo` field, no `links` field. `url` is a `{split_name: URL, …}` dict — one key per uploaded split covered by that subset.
- [ ] `metadata.json` contains **no local machine paths**. The `source.path` key sometimes records a pre-downloaded JSON path during authoring — always remove it before pushing.
- [ ] For multi-subset datasets, each subset's `source.url` keys reference **only that subset's** uploaded split names — not a sibling subset's keys. Copy-paste errors across subsets are easy to miss.
- [ ] `choices` mapping is present only when the source actually provides a choices column. Do not add it for benchmarks where options are embedded in the question text or extracted by the converter.
- [ ] `validate.py --audit <out>/hf_dataset` passes with no issues (covers placeholder/media count, `<image>` inside question text, schema fields, no local paths).
- [ ] **Prompt template matches the authoritative source, source-provided prompt column, or standardized fallback policy.** Render the template against sample rows and compare against the official prompt-construction output byte-for-byte when eval code exists. Fill `prompt_template_source` for every subset: `origin` must be `official`, `source_column`, or `fallback`; `reference` must cite a paper section / repo path + line numbers / HF dataset-card URL, source prompt column name, or canonical template id (for example, `T3`). Never invent an "official" string. If no official or source-provided full prompt exists (e.g., MM-Vet, LLaVA-Bench-in-the-Wild, MMHal-Bench, WildVision-Bench, NoCaps, VizWiz-Captions, VQA-RAD), set `origin = "fallback"` and document the selected fallback template id in `reference`.
- [ ] **No trailer duplication.** If the official upstream renders Options/instructions into a `question` field (MathVerse `query_wo`, MathVista `query`, MEGA-Bench `task_description+example_text+query_text`, CharXiv per-category, MuirBench `<image>`-interleaved options, RealWorldQA baked instruction), the template should NOT add another `Options:` block, another `Answer with …` trailer, or another `<image>` placeholder. Smell test: render against one sample row and grep for `Options:` / `option letter` / `Answer with` — if either string appears twice, you have a duplicate.
- [ ] **No harness-style suffixes on judge-scored benchmarks.** Free-form benchmarks routed to a GPT-4/GPT-4o judge (MM-Vet, LLaVA-Bench-in-the-Wild, MMHal-Bench, WildVision-Bench, VibeEval) must NOT append `Answer the question using a single word or phrase.` / `Answer briefly.` / `Answer the question.` — those are lmms-eval defaults, not benchmark policy, and they directly conflict with judge prompts that score response detail or hallucinations.

If a source has both image and text-only configs for the same benchmark, convert each config separately (with appropriate templates), then merge the splits into one artifact:

```bash
# Convert image config → out_img/ (template with <image>)
python3 scripts/convert.py --hf <repo> --split <img_split> --template '<image>{{ question }}' \
    --template-source-origin fallback --template-source-ref <Tn> …

# Convert text-only config → out_text/ (no image, different subset key)
python3 scripts/convert.py --hf <repo> --split <text_split> --subset text_only \
    --template '{{ question }}' --modalities text \
    --template-source-origin fallback --template-source-ref <Tn> …

# Merge DatasetDicts and combine metadata.json subsets manually
python3 - << 'PYEOF'
import json
from datasets import DatasetDict, load_from_disk
from pathlib import Path

out_img = Path("out_img")
out_text = Path("out_text")
merged = Path("out_merged")
merged.mkdir(parents=True, exist_ok=True)

dd = DatasetDict({
    **load_from_disk(str(out_img / "hf_dataset")),
    **load_from_disk(str(out_text / "hf_dataset")),
})
dd.save_to_disk(str(merged / "hf_dataset"))

with open(out_img / "metadata.json") as f: meta = json.load(f)
with open(out_text / "metadata.json") as f: text_meta = json.load(f)
meta["subsets"].update(text_meta["subsets"])
with open(merged / "metadata.json", "w") as f: json.dump(meta, f, indent=2)
PYEOF
```

### Step 5 — Verify prompts and media

`scripts/validate.py` has two modes:

**Standalone audit (no Simple-MMEval install needed) — run this first:**
```bash
python3 scripts/validate.py --audit <out>/hf_dataset
# or with an explicit metadata.json path:
python3 scripts/validate.py --audit <out>/hf_dataset --metadata <out>/metadata.json
```
Checks: (1) `metadata.json` has no leaked local paths, (2) `source.url` keys match actual split names and no deprecated `repo`/`links` fields are present, (3) modalities/task_type use the supported taxonomy, (4) every row's `<image>`/`<video>` placeholder count equals its media count, (5) `<image>` inside question text when the template already supplies placeholders, (6) every subset's `prompt_template_source.origin` is one of `official`/`source_column`/`fallback` with a non-empty `reference`.

**Full round-trip (requires Simple-MMEval checkout):**
```bash
python3 scripts/validate.py \
    --simple-mmeval <path-to-simple-mmeval> \
    --local <out> \
    --hf <out> \
    -n 3
```
Instantiates the actual `LocalJSONDataset` / `MMEvalHFDataset` data loaders, prints rendered prompts, and confirms images load as PIL with the right dimensions.

Run the standalone audit before the full round-trip; both should pass before declaring a conversion done.

**If the audit fails on `prompt_template_source`** (missing, `unspecified`, or an invalid `origin`): re-derive the value from the Step 2 tier decision and re-run the conversion with the correct `--template-source-origin/-ref/-notes` flags (or fix the `--metadata-json` seed), then re-audit. Do not hand-edit `metadata.json` to silence the check and do not skip the subset. Only if the provenance genuinely cannot be resolved after retrying, surface it in the final report with: the dataset/subset, the current `prompt_template_source` value, the inference path you attempted (which tiers you checked and what you found), and the specific reason it could not be resolved.

### Step 5b — Pre-push smoke (local mode, 50 random rows)

`validate.py` only proves the artifact loads. It does NOT prove the chat
template + processor accept the rendered prompt + media count, that
inference produces output, or that the result file is well-formed. The
classic silent failure mode is: `validate.py` is green but the processor
sees a different `<image>` count than the row's media list at run time.

`scripts/smoke_run.py --local` closes that gap before you push. It
points `mmeval/run.py` at the full local artifact (using mmeval's own
`--sample_num 50 --sample_order random --sample_seed 42` sampler) with
`Qwen3-VL-2B-Instruct` (the unified test model used throughout this
skill), and gates the smoke on **every** sampled row passing the
fidelity checklist below.

First export the path to a python env that has `torch` + `transformers`
(recent enough for the chosen model series) + `datasets` once per shell,
then reuse it everywhere:

```bash
# Pick whichever interpreter has the right deps on this machine.
# On this dev box, `qwen3_vl` is a conda env with transformers >=5.x;
# on yours it might be `~/.venvs/mmeval/bin/python`, a global one, etc.
export MMEVAL_SMOKE_PYTHON="$(conda run -n qwen3_vl which python 2>/dev/null \
    || command -v python3)"
```

`smoke_run.py --python` reads `$MMEVAL_SMOKE_PYTHON` as its default, so once
this is exported you can omit `--python` entirely. The examples below pass
it explicitly so they work without the export step.

```bash
python3 scripts/smoke_run.py \
    --simple-mmeval <path-to-simple-mmeval> \
    --local <out> \
    --python "$MMEVAL_SMOKE_PYTHON"
```

GPU selection is automatic by default — see the GPU notes in Step 6b.
Local-mode smoke runs in a few minutes on a freed H200 once the model
is cached. It is the pre-push gate; do not push until it is green.

**Required flags & defaults (each fixes a real failure mode seen in the field):**

- `--python <bin>` (default reads `$MMEVAL_SMOKE_PYTHON`, else `python3` on
  PATH) — orchestrator interpreter for `mmeval/run.py`, which imports
  `torch` at top-level. The default `python3` on the box often isn't a
  venv/conda env that has torch installed; point this at a conda env
  with `torch` + `datasets`. Per-model inference still dispatches to the
  env in `mmeval/registry.py` — this flag is only the orchestrator
  interpreter.
- `--attn-implementation` (default `sdpa`) — passed through to
  `mmeval/run.py`. The default sidesteps a common silent failure: if the
  per-model conda env has a broken `flash_attn` install (torch ABI
  mismatch), `transformers` falls into an interactive `input()` prompt
  asking whether to fetch a remote flash-attn kernel. With no stdin
  attached the prompt hits EOF, every sample is retried-then-skipped, and
  `result.json` ends up empty. `sdpa` always works.
- `--rows-per-split N` (default `50`, capped at the split size so smaller
  splits run end-to-end) — random samples per (subset, split). mmeval
  does the sampling internally via `--sample_num/--sample_order random`;
  no derived artifact, no row-selection shim. Every sampled row must
  pass the fidelity checks below for the smoke to be considered green.
- `--seed <int>` (default `42`) — forwarded to `--sample_seed` so smoke
  runs are reproducible across machines/sessions.
- `--gpu <id>` — explicit `CUDA_VISIBLE_DEVICES`. Default: auto-pick the
  least-utilized GPU with at least `--min-free-mb` (8 GiB) of free VRAM.
  Pass `--gpu -1` to disable pinning.

**Implementation details (handled automatically):**

1. **Passthrough template (local mode only).** `LocalJSONDataset` only
   uses the saved `messages[0].prompt` when no template override is in
   play; if anything else loads the default template (`{{ question }}`),
   `<image>` placeholders and any post-prompt suffix get stripped and
   the model answers "I'm a text-based AI". `smoke_run.py --local`
   writes a passthrough `{{ prompt }}` template into the work dir and
   passes it via `--template`, which forces the loader to re-emit the
   exact rendered prompt the converter wrote. If you ever invoke
   `mmeval/run.py` against a `local@json` artifact by hand, pass
   `--template <path-to-jinja>` with the same template you handed to
   `convert.py`. HF mode does not need this — `MMEvalHFDataset` reads
   `prompt_template` from `metadata.json` and renders correctly without
   an override.
2. **Response shape.** `mmeval`'s `res_handler.py` writes the model output
   into `result[i].messages[1].response` (the assistant message). It is
   stored as `str`, `list[str]`, or `list[list[str]]` depending on the
   inference backend; `smoke_run.py` flattens before checking emptiness.
   Tolerating a legacy top-level `r["response"]` is kept as a fallback.
3. **Placeholder/media alignment.** `res_handler.py` strips per-message
   `media` before saving, but preserves the sample-level `media` list
   (as `"Image Object"` strings, one per image). The fidelity check
   counts `<image>`/`<video>` placeholders in `messages[0].prompt` and
   compares to `len(media)` — a mismatch here is the most reliable
   signal that the prompt and media list desynced.

**Per-row fidelity checks (must pass for every sampled row in every (subset, split)):**

- `id` is present, non-empty, and unique within `result.json`.
- `messages[0].question` (when present) is a string — mapping-from-source
  drift that overwrites question with a non-string fails here.
- `messages[0].prompt` is a non-empty string (proves the template rendered).
- `<image>`/`<video>` placeholder count in the rendered prompt equals
  the sample-level `media` count.
- When present: `options` is a dict, `choices` is a list, `hint` is a
  string, `answer` is a string or list.
- `messages[1].role == "assistant"` and `response` is non-empty after
  flattening str / list[str] / list[list[str]].
- `result.json` row count equals `min(--rows-per-split, split_size)` —
  a partial result file (silent post-load drop) fails the smoke.

A dataset passes the smoke only when **every** sampled row in **every**
(subset, split) clears every check above; any failure exits non-zero and
leaves the artifacts under `<simple-mmeval>/.tmp/smoke_tests/<dataset-slug>/`
for follow-up.

### Step 6 — Push to HuggingFace (for `hf` mode)

`scripts/push_to_hf.py` is the **final user-facing push script**. It only needs the artifact dir + an HF token + a target repo name:

```bash
python3 scripts/push_to_hf.py \
    --artifact-dir <out> \
    --repo-id <user>/<repo> \
    --token <hf_token> \
    [--private] \
    [--cleanup-artifact]   # delete <out> after a successful push
```

It pushes the `default` config and uploads `metadata.json` to the repo root, then prints the exact `simple-mmeval` invocation for the user to copy.

Add `--cleanup-artifact` on disk-constrained systems to delete the local Arrow artifact immediately after a successful push (the Hub becomes the durable copy).

For datasets whose `metadata.json` packs **multiple subsets** that should land as separate HF configs (one config per subset), use `scripts/push_to_hf_multiconfig.py` instead of `push_to_hf.py`; it pushes one HF config per `metadata.json` subset.

> **Do not push until the split-count checklist in Step 4 passes.** A missing split cannot be patched post-push without a full re-push of the affected splits.

### Step 6b — Post-push framework smoke (HF mode, 50 random rows / (subset, split))

After the push lands, run `scripts/smoke_run.py --hf` to exercise the
**actual** uploaded artifact through `MMEvalHFDataset` — the same
dataloader that real evaluations use. This is the authoritative gate
between "push reported success" and "the dataset is safe to evaluate on":
it catches `metadata.json` upload bugs, prompt-template regressions, and
Arrow round-trip issues that pre-push local smoke cannot see.

```bash
python3 scripts/smoke_run.py \
    --simple-mmeval <path-to-simple-mmeval> \
    --hf <user>/<repo> \
    --python "$MMEVAL_SMOKE_PYTHON"
# add --subset <name> to scope to one subset, e.g. --subset main
```

What it does, in order:

1. Picks the **least-utilized GPU with ≥ `--min-free-mb` (8 GiB)** of
   free VRAM via `nvidia-smi --query-gpu=index,memory.free,utilization.gpu`
   and binds the child via `CUDA_VISIBLE_DEVICES`. Override with
   `--gpu N`; pass `--gpu -1` to disable pinning.
2. Downloads `metadata.json` from the Hub, enumerates every subset, and
   calls `datasets.get_dataset_split_names(repo, "default")` to discover
   every split that will land in front of evaluators.
3. For **every** `(subset, split)`: spawns `mmeval/run.py
   --dataset mmeval_hf@<repo> --subset <subset> --split <split>
   --sample_num 50 --sample_order random --sample_seed 42` (or fewer
   rows when the split has < 50, in which case mmeval caps at the split
   size and every row runs). This uses mmeval's own sampler — no derived
   artifact, no row-selection shim — so what gets exercised is exactly
   what a real evaluation would see.
4. After each run, parses `result.json` and requires that **every** row
   passes the same per-row fidelity checks as Step 5b (id present and
   unique, question/options/choices/hint/answer types, non-empty
   rendered prompt, placeholder count == media count, assistant
   response non-empty), AND that the result row count equals
   `min(--rows-per-split, split_size)`. Anything less fails the smoke.
5. Writes everything to a persistent location that **never gets cleaned
   up by default** (see Step 7):

   ```
   <simple-mmeval>/.tmp/smoke_tests/<repo-slug>/
   ├── smoke_metadata.json   # repo, subsets, splits, model, seed
   ├── smoke_summary.json    # per-(subset,split) rc + rows + violations
   └── <subset>/<split>/
       ├── result.json       # rendered prompts + responses
       └── run.log           # full mmeval/run.py stdout/stderr
   ```

Defaults (`Qwen3-VL-2B-Instruct`, sdpa, seed=42, 50 rows per
(subset, split), auto-GPU) are all chosen so a run is reproducible
across machines and sessions; pin them explicitly only when you have a
specific reason.

**Pass/fail discipline.** A dataset is smoke-test passed only when
**every** `(subset, split)` succeeds and **every** sampled row in
**every** result.json clears every fidelity check. A non-zero exit
means at least one pair either crashed mmeval/run.py (dataloader,
prompt-rendering, media-loading, or runtime error) or produced rows
that violated the structural checks. The `.tmp/smoke_tests/<repo-slug>/`
tree is preserved; inspect `run.log` (raw error) and `result.json`
(the rendered prompt the model actually saw) before declaring the
converted dataset healthy. If a failure points at a converter bug, fix
`convert.py` (or the relevant template in `references/`), re-run the
conversion + push, and re-smoke before signing off.

### Step 7 — Clean up intermediate artifacts

After `push_to_hf.py` exits 0 and you've verified the Hub repo, free disk space with `cleanup.py`. Always confirm the push succeeded first.

```bash
# Quick: delete merged artifact after push (built into push_to_hf.py)
python3 scripts/push_to_hf.py \
    --artifact-dir <merged_dir> --repo-id <user>/<repo> --token "$HF_TOKEN" \
    --cleanup-artifact

# Full: explicit control with --dry-run preview
python3 scripts/cleanup.py \
    --artifact-splits <split1_dir> <split2_dir> ... \
    --merged-artifact <merged_dir> \
    --preprocessed <preprocess.json> \
    --work-dir .tmp/conversions/<dataset>  \  # removes entire per-dataset work dir (HF cache + temp)
    --dry-run   # preview first, then re-run without --dry-run
```

| What | Safe to delete when |
|------|---------------------|
| Per-split `out/<split>/` dirs | After `merge_splits.py` is verified |
| Merged `out/merged/` dir | After `push_to_hf.py` exits 0 |
| Preprocessed `*.json` files | After `convert.py` writes `hf_dataset/` |
| `.tmp/conversions/<dataset>/` work dir | After push succeeds (contains HF cache + temp files) |
| `media/` dir (local mode) | After push succeeds |
| `<simple-mmeval>/.tmp/smoke_tests/<dataset-slug>/` | **Never by default** — Step 5b (local) and Step 6b (HF) smoke artifacts are preserved for human inspection. `cleanup.py` refuses to remove the `smoke_tests/` tree (and skips any legacy `smoke_*` subdir under a work-dir) unless `--include-smoke-results` is passed. |

**Never delete** the merged artifact before a successful push.

**Repo layout after push:**

```
<user>/<repo>/
├── data/                   # default config parquet shards
├── metadata.json           # manifest (top-level)
├── README.md               # auto-generated (default config only)
└── .gitattributes
```

See `references/metadata-json.md` for the full manifest schema (top-level `name`/`release_date`/`subsets[name]` with `language`/`modalities`/`task_type`/`prompt_template`/`prompt_template_source`/`mapping_from_source`).

## Worked example — CaptionQA (nested-MCQ, 4 domain splits)

```bash
SKILL=<skill-path>
ROOT=<workdir>/captionqa

# 1. Convert each split.
# Run in parallel (&) only if disk has enough free space for N simultaneous
# HF cache downloads; on constrained partitions process sequentially (remove &).
for split in natural document ecommerce embodiedai; do
  python3 -u $SKILL/scripts/convert.py \
    --hf Borise/CaptionQA --split $split \
    --map id=id image=images question=question options=choices answer=answer category=category \
    --explode questions \
    --template '<image>
You are given an image and a question about the image. Answer with a SINGLE LETTER (A, B, C, ...), no explanation.

Question:
{{ question }}

Options:
{% for k, v in options.items() %}{{ k }}. {{ v }}{% if not loop.last %}
{% endif %}{% endfor %}

Answer:' \
    --template-source-origin official \
    --template-source-ref 'https://github.com/bronyayang/CaptionQA/blob/main/qa.py (build_caption_qa_prompt)' \
    --template-source-notes 'visual analogue of build_caption_qa_prompt; Caption text replaced by <image>, "given an image" instead of "given a caption"' \
    --mode hf --out $ROOT/$split --workers 16 &
done; wait

# 2. Merge into a single multi-split artifact
python3 $SKILL/scripts/merge_splits.py \
    --inputs $ROOT/natural $ROOT/document $ROOT/ecommerce $ROOT/embodiedai \
    --out $ROOT/merged

# 3. Push (add --cleanup-artifact to delete the merged dir after push)
python3 $SKILL/scripts/push_to_hf.py \
    --artifact-dir $ROOT/merged --repo-id <user>/CaptionQA --token "$HF_TOKEN" \
    --cleanup-artifact

# 4. Clean up per-split dirs and HF cache
python3 $SKILL/scripts/cleanup.py \
    --artifact-splits $ROOT/natural $ROOT/document $ROOT/ecommerce $ROOT/embodiedai \
    --hf-cache Borise/CaptionQA
```

The image-mode prompt is the direct visual analogue of the official `qa.py:build_caption_qa_prompt` from [bronyayang/CaptionQA](https://github.com/bronyayang/CaptionQA): `Caption:\n{caption}` is replaced by `<image>` and the system instruction adapted to "given an image" instead of "given a caption". Choices are *not* shuffled and the "Cannot answer from the caption" option is *not* added — those are caption-mode artifacts.

## Worked example — VizWiz-VQA val

Single command produces both artifacts in ~60 seconds for 4319 rows:

```bash
python3 scripts/convert.py \
    --hf lmms-lab/VizWiz-VQA --split val \
    --map id=question_id question=question image=image answer=answers category=category \
    --template '<image>{{ question }}
Answer the question using a single word or phrase.' \
    --answer-list \
    --template-source-origin fallback \
    --template-source-ref T1 \
    --mode both \
    --out <workdir>/vizwiz_val \
    --workers 16
```

Then verify both, then smoke-run:

```bash
python3 scripts/validate.py \
    --simple-mmeval <simple-mmeval-checkout> \
    --local <workdir>/vizwiz_val \
    --hf <workdir>/vizwiz_val \
    -n 3

python3 scripts/smoke_run.py \
    --simple-mmeval <simple-mmeval-checkout> \
    --local <workdir>/vizwiz_val \
    --python "$MMEVAL_SMOKE_PYTHON"
```

Run via Simple-MMEval (local artifact, no HF push needed):

```bash
cd <simple-mmeval-checkout>
export PYTHONPATH=./:$PYTHONPATH
python3 mmeval/run.py \
    --model_name_or_path Qwen3-VL-2B-Instruct \
    --dataset local@json \
    --infile <workdir>/vizwiz_val/data.json \
    --img_dir <workdir>/vizwiz_val/media \
    --out_dir work_dirs/vizwiz_val \
    --gpu_per_parallel 1 --parallel_per_task 1
```

Push HF artifact to the user's Hub:

```bash
python3 scripts/push_to_hf.py \
    --artifact-dir <workdir>/vizwiz_val \
    --repo-id <user>/VizWiz-VQA \
    --token "$HF_TOKEN" \
    --cleanup-artifact   # optional: free local disk after push

# If not using --cleanup-artifact, clean up manually afterward:
python3 scripts/cleanup.py \
    --merged-artifact <workdir>/vizwiz_val \
    --hf-cache lmms-lab/VizWiz-VQA
```

## Worked example — MMMU / MMMU-Pro (indexed multi-image MCQ + multi-config)

MMMU-style benchmarks are the **most error-prone shape this skill supports**.
They have three quirks that the generic `convert.py` doesn't handle directly:
indexed `<image N>` references, image references inside option strings, and
multiple subject configs that should land in one repo. Use a custom
preprocessor and feed the result into the HF artifact directly.

The full converter for MMMU and MMMU-Pro lives in this repo at
`scripts/convert_indexed_multimage.py` (template-driven, no baked prompt).
The key per-row logic is:

```python
IMG_REF_RE = re.compile(r"<image (\d+)>")

def process_row(row):
    options = ast.literal_eval(row["options"]) if row["options"] else []
    question = row["question"] or ""

    # 1) Collect refs from question AND every option string, in textual order.
    refs = [int(m) for m in IMG_REF_RE.findall(question)]
    for o in options:
        refs.extend(int(m) for m in IMG_REF_RE.findall(o))

    # 2) Build media list: one entry per reference (the same image may repeat).
    images = [row.get(f"image_{i}") for i in range(1, 8)]
    media = [images[r - 1] for r in refs]   # 1-indexed in the source

    # 3) Normalize <image N> → <image> in question + options BEFORE storing.
    msg_question = IMG_REF_RE.sub("<image>", question)
    msg_options = {chr(ord("A") + i): IMG_REF_RE.sub("<image>", v)
                   for i, v in enumerate(options)}

    # 4) Sanity: total <image> tokens the template will emit must equal media count.
    assert msg_question.count("<image>") + sum(v.count("<image>") for v in msg_options.values()) == len(media)
```

The Jinja template for MMMU mirrors the official paper / eval-code prompt
[byte-for-byte](https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/utils/data_utils.py):

```jinja
{{ question }}{% if options %}

{% for k, v in options.items() %}({{ k }}) {{ v }}
{% endfor %}
Answer with the option's letter from the given choices directly.{% else %}

Answer the question using a single word or phrase.{% endif %}
```

Verify it matches the upstream prompt-construction output verbatim:

```python
got = env.from_string(TEMPLATE).render(question=q, options=opts)
exp = upstream_prompt(row)
assert got == exp   # byte-for-byte
```

For MMMU-Pro the prompt format is *different* (`A. opt` not `(A) opt`,
single newlines, "option letter" without apostrophe-s) — see
[`mmmu-pro/infer/infer_gpt.py`](https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu-pro/infer/infer_gpt.py).
Always grep the upstream eval code; do not assume two related benchmarks
share a prompt.

## Worked example — Video-MME (video MCQ, YouTube-sourced)

Video-MME is a representative video MCQ benchmark with 900 YouTube videos and 2,700 QA pairs (3 per video). This example shows the full workflow for a YouTube-sourced video dataset.

**Step 1: Download videos.** Videos must be obtained from YouTube using the `videoID` field. Some will be unavailable.

```bash
# Pre-download: fetch annotation, download videos, record mapping
python3 - << 'PYEOF'
import json, subprocess, os
from datasets import load_dataset
from pathlib import Path

ds = load_dataset("lmms-lab/Video-MME", "videomme", split="test")
video_dir = Path("$WORK/video_mme_videos")
video_dir.mkdir(parents=True, exist_ok=True)

# Deduplicate by videoID (900 unique videos, 2700 QA pairs)
unique_videos = {row["videoID"]: row["url"] for row in ds}
missing = []
for vid_id, url in unique_videos.items():
    out_path = video_dir / f"{vid_id}.mp4"
    if out_path.exists():
        continue
    try:
        subprocess.run(["yt-dlp", "-f", "best[ext=mp4]", "-o", str(out_path), url],
                       check=True, timeout=300)
    except Exception as e:
        missing.append({"videoID": vid_id, "error": str(e)})
        print(f"SKIP {vid_id}: {e}")

# Build preprocessed JSON with video paths
rows = []
for row in ds:
    vid_path = video_dir / f"{row['videoID']}.mp4"
    if not vid_path.exists():
        continue
    rows.append({
        "id": row["question_id"],
        "question": row["question"],
        "options": row["options"],  # list → normalized to dict by converter
        "answer": row["answer"],
        "video_path": str(vid_path),
        "video_id": row["videoID"],
        "duration_category": row["duration"],
        "task_type_source": row["task_type"],
    })
with open("$WORK/video_mme_preprocessed.json", "w") as f:
    json.dump(rows, f, ensure_ascii=False)
print(f"Wrote {len(rows)} rows ({len(missing)} videos missing)")
PYEOF
```

**Step 2: Convert.**

```bash
SKILL=<skill-path>
# video_path values are absolute, so --media-dir is the filesystem root.
# --mode both also writes hf_dataset/ so Step 3's --audit has something to check;
# --hf-video stores video paths as strings (video files are not redistributable).
python3 -u $SKILL/scripts/convert.py \
    --json $WORK/video_mme_preprocessed.json \
    --media-dir / \
    --map id=id question=question media=video_path options=options answer=answer \
         video_id=video_id duration_category=duration_category \
    --template '<video>{{ question }}
Answer with the option'"'"'s letter from the given choices directly.' \
    --task-type multiple_choice_vqa --modalities single_video_start \
    --name "Video-MME" --release-date 2024-06-01 \
    --template-source-origin official \
    --template-source-ref 'Video-MME official post_prompt (MME-Benchmarks/Video-MME eval code)' \
    --mode both --hf-video --out $WORK/VideoMME --split test \
    --verify-video --workers 1
```

**Step 3: Validate.**

```bash
python3 $SKILL/scripts/validate.py --audit $WORK/VideoMME/hf_dataset \
    --metadata $WORK/VideoMME/metadata.json
```

**Step 4: Smoke test.**

```bash
python3 $SKILL/scripts/smoke_run.py \
    --simple-mmeval <simple-mmeval-checkout> \
    --local $WORK/VideoMME \
    --python "$MMEVAL_SMOKE_PYTHON"
```

**Step 5: Add `video_storage` to metadata.json** (manual step — the converter does not auto-populate this block yet):

```python
import json
with open("$WORK/VideoMME/metadata.json") as f:
    meta = json.load(f)
meta["subsets"]["main"]["video_storage"] = {
    "format": "files",
    "media_root": "media",
    "notes": "900 videos from YouTube via yt-dlp. N videos unavailable due to takedowns."
}
with open("$WORK/VideoMME/metadata.json", "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
```

## Worked example — MVBench (video MCQ, bundled on HF)

MVBench is a multi-task video benchmark with 20 subsets (200 videos each) where most videos are bundled directly in the HF dataset as WebM files.

```bash
SKILL=<skill-path>
# MVBench has 20 configs; process each as a separate split
for config in action_antonym action_count action_localization action_prediction \
    action_sequence character_order counterfactual_inference egocentric_navigation \
    episodic_reasoning fine_grained_action fine_grained_pose moving_attribute \
    moving_count moving_direction object_existence object_interaction \
    object_shuffle scene_transition state_change unexpected_action; do

    # Pre-download: HF dataset has video filenames as strings;
    # actual video files must be downloaded from the video/ subfolder in the HF repo
    python3 - << PYEOF
import json
from datasets import load_dataset
from huggingface_hub import hf_hub_download

ds = load_dataset("OpenGVLab/MVBench", "$config", split="train")
rows = []
for i, row in enumerate(ds):
    video_file = row["video"]  # e.g., "action_antonym/xxx.webm"
    # Download the video file from HF
    local_path = hf_hub_download("OpenGVLab/MVBench", f"video/{video_file}",
                                  repo_type="dataset")
    rows.append({
        "id": f"${config}_{i:04d}",
        "question": row["question"],
        "candidates": row["candidates"],  # list of 3 options
        "answer": row["answer"],
        "video_path": local_path,
    })
with open("\$WORK/mvbench_${config}.json", "w") as f:
    json.dump(rows, f, ensure_ascii=False)
print(f"${config}: {len(rows)} rows")
PYEOF

    python3 -u $SKILL/scripts/convert.py \
        --json $WORK/mvbench_${config}.json \
        --map id=id question=question media=video_path options=candidates answer=answer \
        --template '<video>{{ question }}
Answer with the option'"'"'s letter from the given choices directly.' \
        --mode local --out $WORK/MVBench_${config} --split $config \
        --verify-video --workers 1
done
```

## Common pitfalls (learned the hard way)

- **Mirror the official upstream's HF config/split layout byte-for-byte — do not invent or collapse axes.** The first step before pushing ANY multi-subset dataset is to call `get_dataset_config_names(official_repo)` + `get_dataset_split_names(official_repo, cfg)` for every config the official upstream exposes, and reproduce that exact `{config: [splits…]}` shape. Two failure modes are equally bad:
  - **Over-flattening** (wrong): the official has multiple configs (`lmms-lab/MMBench` → `cc`/`cn`/`en`, `BLINK-Benchmark/BLINK` → 14 task configs) but the mm-eval push concatenates them into one `default` config with `<subset>_<split>` flat splits. Fix by re-pushing as multi-config via `scripts/push_to_hf_multiconfig.py`.
  - **Over-restructuring** (also wrong): the official has a single `default` config where a category column distinguishes logical subsets (e.g. `lmms-lab/CMMMU` → `default/[dev, val, test]` with a `category` field; `princeton-nlp/CharXiv` → `default/[validation, test]` with `category`; `OpenGVLab/CRPE` → 2 eval files; `lmms-lab/ICON-QA` → `default/[val, test]`), but the mm-eval push splits these into N configs. Fix by reverting to `default` and keeping the category as a per-row column. `mm-eval/CMMMU`, `mm-eval/CharXiv`, `mm-eval/CRPE`, `mm-eval/IconQA`, `mm-eval/HallusionBench` were all corrected to default-single-config in the 2026-05-29 audit for this reason.
  The mm-eval logical "subsets" in `metadata.json` are a separate concept used for prompt-template selection at eval time, and they do NOT have to map to HF configs 1-to-1. When the official upstream encodes the subset as a row column, leave the data as a single `default` config and use `metadata.json subsets[].mapping_from_source.extra.category` to surface that column for downstream filtering.
  Decision table:
  | Official HF layout | mm-eval HF layout | metadata.json subsets |
  |---|---|---|
  | Multiple HF configs (cc/cn/en × dev/test) | mirror byte-for-byte (cc/cn/en × dev/test) | one per official config; source.url per-config |
  | Single `default` config with row-level `category` field | `default` + same splits as official; row-level category preserved | one per category; each `source.url` references the same `default` config |
  | Single `default` config with a single homogeneous split | `default` config, single split | single subset (typically `main`) |
  Either `scripts/push_to_hf_multiconfig.py` (multi-config) or `scripts/push_to_hf.py` (single-default) is appropriate depending on which shape matches the official upstream.
- **HF repo name must match the official dataset name exactly — no appended suffixes.** Use the benchmark's official name (e.g., `MMStar`, `ScienceQA`, `VQA-RAD`) as the HF repo name, including exact capitalization, spelling, and punctuation. Never append `-mmeval` or any other suffix; the `mm-eval/` org prefix already namespaces the dataset unambiguously.
- **Don't trust intermediate "convenience" reformats — re-source from the official upstream.** Third-party consolidations of a benchmark sometimes pre-render a `prompt` field whose token count silently disagrees with the unique-image count, or strip metadata (subject, difficulty, explanation) you'd want later. When the upstream is available, source from it.
- **Case-sensitive identifiers — silent wrong results or dropped rows.** HF split names, config names, and dataset column names are all case-sensitive: `val` ≠ `Val`, `question_id` ≠ `Question_ID`. Always take split and config names from `get_dataset_split_names` / `get_dataset_config_names`, and column names from `inspect_source.py` output — never from descriptions, papers, or memory.
- **Missing splits — the silent but critical failure.** HF datasets often expose multiple splits (`test`, `testmini`) or multiple configs (`testmini`, `testmini_text_only`). Converting only one and pushing is wrong even when row counts look right. Always enumerate all configs and splits with `get_dataset_config_names` + `get_dataset_split_names` before starting, and verify against the converted artifact at Step 4.
- **Config name ≠ split name.** For HF datasets that require a config (`load_dataset(repo, config_name, split=...)`), pass `convert.py --hf <repo> --hf-config <name> --split <split>`. Only pre-download to local JSON (then `--json`) when a config still cannot be loaded directly. See the "Datasets that require a config name" section above.
- **Multi-config benchmarks → one split per config.** When a source has multiple HF configs that should evaluate as a single benchmark (MMMU-Pro: `standard (4 options)` + `standard (10 options)` + `vision`; MMMU: 30 subjects), run the converter per config into separate output dirs and `merge_splits.py` them. Pick split names without spaces or parens (`standard_4_options`, not `standard (4 options)`) — HF Hub config/split names disallow special chars.
- **Source the prompt from the official eval code, then the paper, in that order.** When they disagree the eval-code version wins (it produced the reported numbers). Cite path + line numbers in `prompt_template_source.reference` and set `prompt_template_source.origin = "official"`. Show the user a rendered prompt for one short-answer and one MC row, name the source, and wait for sign-off before the full conversion — re-pushing a multi-GB HF artifact is more expensive than a 5-line preview.
- **Assert template == upstream byte-for-byte.** When the source ships an eval script, render your template with one sample row and compare it to the upstream prompt-construction output. Tiny discrepancies (extra blank line, `(A)` vs `A.`, `option's letter` vs `option letter`, trailing space + newline vs newline only) silently change scores.
- **Prefer a transparent Jinja template over a pre-rendered prompt when both are available.** Per the format spec, when the metadata template is non-empty Simple-MMEval *always* re-renders it (the per-row `prompt` becomes a fallback only when rendering throws). If the source has both a full prompt column and component fields, reverse-engineer a byte-identical template (priority 2.1). Use a one-key `{{ prompt }}` pass-through (priority 2.2) only when the full prompt cannot be cleanly decomposed.
- **Multi-image rows.** Use `--map images=<list-field>` and ensure the template emits the right number of `<image>` placeholders. For indexed `<image N>` patterns (MMMU-style, note the space: `<image 1>`) see the worked example above — that shape needs custom preprocessing that replaces each `<image N>` ref with `<image>` and builds a multi-item media list.
- **`<imageN>` tokens (no space) are text labels, not media placeholders.** Some datasets (e.g., MathVision) use `<image1>`, `<image2>`, … (no space before the digit) as textual labels for subfigures *within* a single composite image — each row has exactly **one** media file. These are distinct from MMMU's `<image N>` (space before digit) which reference separate images. Preserve `<imageN>` verbatim in the question text; the canonical T2 template (`<image>{{ question }}`) supplies the single real placeholder. Never replace `<image1>` with `<image>` in the question field. The `validate.py --audit` check catches `<image>` embedded in question text as a structural error.
- **Text-only configs in image benchmarks.** Benchmarks like MathVerse have both an image config and a `text_only` config. The text-only config needs a separate `--subset text_only` conversion with a template that has no `<image>` placeholder and `--modalities text`. Merge the resulting DatasetDicts manually (see Step 4 merge snippet).
- **Empty string `""` for absent images silently drops rows.** In pre-downloaded JSON, use `None` (JSON `null`) for rows without images — not `""`. An empty string is a valid string that `_value_to_pil` tries (and fails) to decode as base64, causing `encode_failed` and dropping the row.
- **Video rows — use `--mode local` or `--mode hf --hf-video`.** Videos go into `media` as paths or URLs; the template should use `<video>` placeholders. Plain `--mode hf` raises `ValueError` for video benchmarks because the standard HF schema uses `Sequence(Image())` which cannot carry video bytes. To distribute video datasets via HF, pass `--hf-video` to store video paths as strings in the Arrow `media` column; upload video files to the repo separately.
- **Missing media is silently skipped.** `convert.py` reports a `skipped` count in `convert_summary.json`. Double-check it isn't unexpectedly large — `skipped > 0` almost always means a reference resolution bug, not genuinely missing data.
- **`--limit N` counts post-explode rows, not source rows.** With `--explode questions`, `--limit 5` keeps 5 sub-rows total, which may be fewer than 5 source rows.
- **Local `data.json` is loaded fully into RAM at eval time.** Simple-MMEval's `LocalJSONDataset` uses `json.load`. For benchmarks beyond ~500k rows or ~500MB JSON, prefer `--mode hf` (Arrow-backed, lazy) or shard the source upfront.
- **Don't pass `--token` on the command line for production pushes.** It lands in shell history. Prefer `export HF_TOKEN=...` and let `push_to_hf.py` pick it up via the env var.
- **HF cache not cleaned up.** After converting a large dataset, its parquet files remain in `~/.cache/huggingface/hub/`. Run `python3 scripts/cleanup.py --hf-cache <repo_id>` after push to reclaim the space.
- **`<image>` placeholder count must equal row-level media count.** Reminder: `<image>` and `<video>` (no digit, no space) are the only canonical media placeholders. Strings like `<image1>` / `<image2>` (no space before digit, used in MathVision question text) and `<image 1>` / `<image_1>` (with space or underscore, used in MMMU / OlympiadBench) are dataset-native textual labels — keep them verbatim inside the question field and let the template prepend exactly one real `<image>` placeholder per actual media file. Replacing `<image_N>` / `<image N>` with `<image>` in MMMU-style requires the dedicated `convert_indexed_multimage.py` path so the media list lines up.
- **Switching prompt on per-row metadata, not on split name.** When a dataset's official prompt varies by language, subject, problem type, etc. (OlympiadBench's 18 splits being the prime example), the discriminating fields must live on the per-row message dict (e.g., `subject`, `language`, `is_theorem_proving`, `answer_type`). Jinja templates render against the message dict — they do NOT receive the split name. If the source has the field as a column, map it via `--map`; if not, re-process the source to add it before uploading. Encoding "switch on split name" by parsing the source file path inside the converter and writing the result into a message field is the cleanest workaround.
- **Harness trailers vs benchmark policy.** lmms-eval / VLMEvalKit default short-answer trailers (`Answer the question using a single word or phrase.`) are *harness conventions*, not benchmark policy. For datasets on the judge-scored allow-list in `jinja-templates.md` (MM-Vet, LLaVA-Bench-in-the-Wild, MMHal-Bench, WildVision-Bench, VibeEval) the trailer hurts scoring; use T2 (bare question). For datasets whose `question` already inlines an answer-format cue (RealWorldQA, MathVista `query`, MathVerse `query_wo`, CharXiv per-category instructions, MEGA-Bench task scaffold, OCRBench per-category instructions), use T2 to avoid duplicating the cue.
- **Local machine paths in published metadata.json.** `validate.py --audit` catches these automatically. If authoring metadata by hand, ensure no `source.path` key contains an absolute local path before pushing.
- **Copy-paste `source.url` across subsets.** When a dataset has multiple subsets (e.g., `reasoning` + `descriptive`, `main` + `text_only`), it's easy to copy a subset block and forget to update its `source.url` keys. Each subset's `source.url` must reference **only that subset's** uploaded split names — not a sibling subset's keys. Always cross-check against `get_dataset_split_names` output after push.
- **No-fabrication contract.** Rows are dropped (not patched) when they break this contract:
  - `id` is `None` / missing → counted as `missing_required:id`.
  - Video URL has no recognizable extension → counted as `unknown_video_ext` (the converter refuses to assume `.mp4`).
  - An image fails to decode (broken base64, missing file, HTTP error, …) → counted as `encode_failed`.

  Optional schema fields with `None` source values keep their empty representation (`""` / `{}` / `[]`) — that's a legal "no value", not fabrication. List `options` are normalized to `{A: …, B: …}` per the documented label transform.
- **Three different "id" concepts — keep them distinct.** Some upstreams use the same field name for things that mean different things; the converter and validator both treat them separately:
  - **Source id** (`source_id` on the message dict) — the upstream identifier of an *entity* (a chart, document, image, exam paper). Not necessarily row-unique. Preserved verbatim so downstream consumers can join back to the upstream source.
  - **Split-local id** — an id that is only unique within one split. Mm-eval datasets ship a single `default` config; a split-local id is acceptable only if the dataset has exactly one split, otherwise re-namespace at conversion time.
  - **Globally unique row id** (`id` on the row) — required by Simple-MMEval's `result.json` semantics. Must be unique across the *entire* uploaded dataset, not just one split.
- **`convert.py` auto-disambiguates duplicate row ids by default.** When the source's per-row id field is not row-unique (e.g., ChartNet ships multiple QA paraphrases per chart UUID; many doc-VQA datasets ship multiple Q&A per page id), the converter rewrites each duplicate to `{source_id}_q{k}` (1-indexed by source-stream order) and adds `source_id` to the per-row message dict. The action is recorded in `convert_summary.json:duplicate_ids_resolved`. Disable with `--no-auto-disambiguate-ids` if you genuinely want the converter to fail on collisions instead.
- **`validate.py --audit` is the gate.** It (a) requires `id` to be unique across every (subset, split), (b) when `source_id` is present on the rows, also requires that rows sharing a source_id have distinct row ids, and (c) prints a one-line note when the composite-id convention is active so you can tell at a glance whether the dataset uses 1-row-per-source or multi-row-per-source.
- **CircularEval-compressed image references in VLMEvalKit TSVs.** VLMEvalKit-hosted TSVs (`opencompass.openxlab.space/utils/VLMEval/*.tsv`, e.g. `MMBench_DEV_EN_V11.tsv`) frequently ship a CircularEval-compressed schema: rows where the `image` cell is a short integer string (typically <50 chars) reference another row's `index` column instead of carrying their own base64. Skipping the resolution silently drops those rows (`encode_failed: base64 decode failed: <integer>`). Pre-process the TSV with `base_imgs = {row['index']: row['image'] for row in df if len(row['image']) >= 500}` and replace `image` cells where `len < 500` with `base_imgs[row['image']]` before feeding to `convert.py`. A 70%+ `encode_failed` rate on a VLMEvalKit TSV almost always means this resolution step was skipped.
- **Expired upstream TLS certs.** Some canonical mirrors (notably `opencompass.openxlab.space` as of 2026-05-28) present an expired TLS cert. `requests`/`urllib` will reject the connection by default. Workaround: stream with `ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE` and re-verify the downloaded file by MD5 against the upstream-published hash. Never blindly disable TLS verification for any host where you can't independently verify the file via a published checksum.

- **Video datasets: always check `convert_summary.json` for skipped rows.** YouTube takedowns, expired links, gated access, and download failures silently produce `encode_failed` / `unknown_video_ext` skips (the two grouped prefixes in `skip_reason_counts`). A 30% skip rate on ActivityNet-QA is plausible (YouTube takedowns); a 30% skip rate on MVBench (bundled on HF) is a bug.
- **Video datasets: don't confuse `<video>` with textual references to "the video".** Many video QA questions contain phrases like "In the video, what does the person do?". These are question text, not media placeholders. The canonical `<video>` placeholder (angle brackets, lowercase, no space) is the only string that triggers media binding. Never replace textual "video" references with `<video>`.
- **Video datasets: include `video_storage` in metadata.** The `validate.py --audit` check flags video subsets missing the `video_storage` block. Add it manually after conversion — the converter does not auto-populate it yet.
- **Video datasets: process one dataset at a time.** Video datasets are large (often 10–400+ GB). Never convert multiple video datasets in parallel. Check disk space with `df -h` before starting, and clean up intermediate files with `cleanup.py` after each push.
- **Video datasets: use `--workers 1` for video conversion.** Video materialization (download/copy) is I/O-bound and network-limited. Parallel workers don't speed up downloads and may cause rate limiting. Use `--workers 1` unless you know the source supports parallel reads.
- **Video datasets: verify with `--verify-video`.** For large downloads, some files may be truncated or corrupted. Pass `--verify-video` to check each file after materialization. Install PyAV (`pip install av`) for deep probing (codec, duration, frame count).
- **Video placeholder `<video>` count must equal video file count per row.** The same rule as images: one `<video>` placeholder per video file in the `media` list. For benchmarks with one video per question (most common), the template starts with a single `<video>`.

## Reference files

- `references/mmeval-format.md` — full schema details, with valid/invalid examples.
- `references/metadata-json.md` — authoring `metadata.json`, `--metadata-json` seed workflow, and what the converter refreshes vs preserves.
- `references/jinja-templates.md` — ready-made templates per benchmark style (T1–T10, including video templates T6/T8/T9/T10).
- `references/video-datasets.md` — survey of popular video evaluation datasets and the mm-eval video storage/metadata design.

## When *not* to use this skill

- The user wants to add a new **model** to Simple-MMEval — that's a different task (see Simple-MMEval's CONTRIBUTING.md).
- The dataset is already in mm-eval format on the `mm-eval/` HF org — point Simple-MMEval at it directly with `--dataset mmeval_hf@mm-eval/<name>` (ensure it ships a root `metadata.json` in the new layout).
- **Pure text-only benchmarks** can use this skill, but the template must not contain any `<image>` / `<video>` placeholder, and `--map` must omit `image` / `images` / `media`.
- **Circular evaluation** (each MCQ replicated with rotated option labels) is *not* applied here — convert first, then run a second pass.
- **System messages** are not part of the mm-eval message schema (`role` ∈ `{user, assistant}`). Fold any system instruction into the first user message.
- **HF + video benchmarks** — convert with `--mode local` (primary) or `--mode hf --hf-video` (for HF distribution with video paths as strings). The standard HF image format (`Sequence(Image())`) does not carry video bytes.
