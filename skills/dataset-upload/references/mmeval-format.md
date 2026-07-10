# mm-eval data format reference

Simple-MMEval reads two input shapes; both ultimately produce per-sample dicts with `id`, `media`, and `messages`.

## Local JSON shape (`--dataset local@json --infile <data.json> --img_dir <media_dir>`)

`data.json` is a JSON array. Each entry:

```json
{
  "id": "vw_val_00000000",
  "media": ["vw_val_00000000.png"],
  "messages": [
    {
      "role": "user",
      "question": "<image> What does this say?",
      "answer": "",
      "options": {},
      "choices": [],
      "prompt": "<image> What does this say?\nAnswer the question using a single word or phrase.",
      "hint": ""
    }
  ]
}
```

- `media` entries are **basenames relative to `--img_dir`**.
- `prompt` is the rendered prompt; if present, no template is applied at runtime.
- `answer`/`options`/`choices`/`hint` are optional but conventional — keep them for round-trippability.
- The runner adds `eval-id` (an integer index) when iterating.

## HF dataset shape (`--dataset mmeval_hf@<user>/<repo>` [+ `--subset <name>` when the manifest has multiple subsets])

### `default` config
Arrow table(s) on the Hub. When converted with an explicit `--split`, this is a named `DatasetDict` split (`val`, `test`, …). When converted from local JSON/TSV/CSV **without** `--split`, the on-disk artifact is a flat `Dataset` (Hub push lands on split `train`). Columns:

| Column | Type | Notes |
|---|---|---|
| `id` | string | unique within the split |
| `media` | `Sequence(Image())` for image datasets; `Sequence(Value('string'))` for video datasets | Always emitted as a sequence. For image datasets: PIL Image objects (bytes embedded in Arrow). For video datasets: relative file paths as strings (e.g., `videos/vid_001.mp4`); actual video files stored separately in the HF repo. |
| `messages` | string (JSON) | `json.dumps([{role, question, ...}], ensure_ascii=False)` — JSON text so each HF row mirrors local `data.json`'s `messages[0]` payload; consumers don't branch on mode. |

### `metadata.json` (repo root)
Not a second dataset config — a single JSON file downloaded at runtime (`hf_hub_download(..., filename="metadata.json")`). It documents subsets, modalities, source provenance, per-field mappings, and the Jinja `prompt_template` used to re-render HF rows. See [`metadata-json.md`](metadata-json.md) for the full schema and authoring workflow.

### How prompts are built

Per row, per message:

1. If `messages[i].prompt` is set, use it as-is.
2. Else render the template with the message dict as context.
3. Replace each `<image>` / `<video>` placeholder with the next media item from `media` (consumed in order across messages).

The number of placeholders rendered must equal the number of media for that message. Mismatches raise immediately.

## Required canonical fields per message

`role` ∈ {`user`, `assistant`}. The base template expects:

- `question` — string
- `options` — dict (`{}` for non-MCQ)
- `hint` — string (`""` if absent)

Any other key is allowed and visible to your custom template (`choices`, `category`, `image_id`, etc.).

## Local vs HF prompt rendering

- **Local mode**: prompts are rendered at conversion time and frozen into `messages[i].prompt`. The runner uses them as-is; the template is *not* re-applied. This keeps local artifacts deterministic and self-contained.
- **HF mode**: only the message dict is shipped; the template lives in `metadata.json → subsets[<subset>].prompt_template` and is re-rendered per row at eval time. This lets you fix a template and re-push without re-encoding images.

## Video format details

### Local mode (`--dataset local@json`)

Video files live under `--img_dir` (the same directory as images). The `media` array in `data.json` contains video filenames (basenames, no directory prefix):

```json
{
  "id": "vmme_test_001_q0",
  "media": ["vmme_test_001_0.mp4"],
  "messages": [
    {
      "role": "user",
      "question": "What is happening in this video?",
      "answer": "A",
      "options": {"A": "A person is cooking", "B": "A person is dancing"},
      "prompt": "<video>What is happening in this video?\nAnswer with the option's letter from the given choices directly.",
      "hint": ""
    }
  ]
}
```

The dataloader returns the video file path as a string (no decoding). Model inference backends handle frame extraction.

### HF video mode (`--dataset mmeval_hf@<user>/<repo>`)

The Arrow dataset uses `Sequence(Value('string'))` for the `media` column instead of `Sequence(Image())`. Each entry is a relative path to a video file stored in the HF repo:

```
media: ["videos/vmme_test_001_0.mp4"]
```

The dataloader downloads video files from the HF repo via `hf_hub_download()` on first access, then passes the local cache path to the model.

### Mixed image+video datasets

For datasets with both image and video rows (modality `multi_image_video_interleave`), the `media` list may contain both image objects and video path strings. The dataloader distinguishes them by type: PIL Image objects are images; strings with video extensions are videos.

## Validation checklist

- [ ] `id` is a string and unique within the split.
- [ ] `media` is a list (single-image still becomes `[img]`; single-video becomes `["path.mp4"]`).
- [ ] In HF mode, `messages` is a JSON-encoded string serialized with `ensure_ascii=False` (CJK content otherwise inflates 3-5×).
- [ ] In local mode, every `media[i]` is a **basename** relative to `--img_dir` — no `..`, no absolute paths, no URLs.
- [ ] Rendered prompt's placeholder count matches `len(media)` per message. Count `<video>` placeholders for video media.
- [ ] `metadata.json` `mapping_from_source.source.url` is a `{split_name: URL, …}` dict; keys correspond to actual uploaded split names. No `links`, `repo`, or `path` keys in `source`.
- [ ] For video datasets: every video file referenced in `media` exists, has a recognized extension, and has size > 0.
- [ ] For video datasets: `metadata.json` includes a `video_storage` block (see `metadata-json.md`).
