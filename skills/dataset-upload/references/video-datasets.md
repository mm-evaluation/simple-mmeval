# Video multimodal evaluation datasets — survey and mm-eval design reference

This document surveys popular video evaluation benchmarks and defines the mm-eval conventions for storing, uploading, validating, and loading video datasets.

## 1. Survey of video evaluation datasets

### 1.1 Dataset catalog

| Dataset | HF ID | Videos | QA pairs | Format | QA type | Duration | Videos bundled on HF |
|---------|-------|--------|----------|--------|---------|----------|---------------------|
| Video-MME | `lmms-lab/Video-MME` | 900 | 2,700 | MP4 (YouTube) | MCQ (4 options) | 11 s – 1 hr | No |
| Video-MME v2 | `MME-Benchmarks/Video-MME-v2` | 800 | 3,200 | MP4 (1080p) | MCQ (3–8 options) | Short–long | Yes (zipped, 177 GB) |
| MVBench | `OpenGVLab/MVBench` | 4,000 | 4,000 | WebM | MCQ (3 options) | 5–35 s | Mostly (17 GB); 320 NTU RGB+D gated |
| EgoSchema | `lmms-lab/egoschema` | ~5,000 | 5,031 | MP4 (Ego4D) | MCQ (5 options) | 3 min each | No (Kaggle/Wasabi/Drive) |
| ActivityNet-QA | `lmms-lab/ActivityNetQA` | 5,800 | 58,000 | MP4 (YouTube) | Open-ended | avg 180 s | No |
| NExT-QA | `lmms-lab/NExTQA` | 5,440 | 52,044 | MP4 (VidOR) | MCQ (5) + open | avg 44 s | No |
| TGIF-QA | (no official HF) | ~72 K GIFs | 165,165 | GIF | Mixed | avg 3.1 s | Partial (base TGIF only) |
| MSVD-QA | `morpheushoc/msvd-qa` | 1,970 | 50,505 | AVI | Open-ended | avg 10 s | Yes (community, 10.9 GB) |
| MSRVTT-QA | `morpheushoc/msrvtt-qa` | 10,000 | 243,000 | MP4 | Open-ended | avg 15 s | Auth-gated |
| PerceptionTest | `lmms-lab/PerceptionTest` | 11,600 | 38,060 | MP4 + audio | MCQ (3 options) | avg 23 s | No (annotations only) |
| LongVideoBench | `longvideobench/LongVideoBench` | 3,763 | 6,678 | MP4 | MCQ (4 options) | 8 s – 1 hr | Yes (tar archives, 162 GB, gated) |
| MLVU | `MLVU/MVLU` | ~1,730 | 3,102 | MP4 | MCQ + generation | 3 min – 2 hr | Yes (430 GB, gated) |
| TempCompass | `lmms-lab/TempCompass` | 410 | 7,540 | MP4 | MCQ + YN + caption | < 30 s | No (ref by video_id) |

### 1.2 Common patterns observed across datasets

**Video format**: MP4 dominates (>80% of benchmarks). WebM (MVBench), AVI (MSVD-QA), and GIF (TGIF-QA) appear occasionally. Converter and dataloader must handle at least `.mp4`, `.webm`, `.avi`, `.mkv`, `.mov`, `.gif`.

**Video distribution models**:
- *YouTube-sourced* (Video-MME, ActivityNet-QA, MSRVTT): dataset provides YouTube IDs/URLs; videos must be downloaded externally. Subject to takedowns.
- *Bundled on HF* (Video-MME v2, MVBench, LongVideoBench, MLVU, MSVD-QA): videos stored directly in the HF repo, sometimes as zip/tar archives.
- *External cloud storage* (EgoSchema: Kaggle/Wasabi/Drive; NExT-QA: Drive): separate download step with potentially expiring links.
- *Official project page only* (PerceptionTest: ZIP downloads, TempCompass: Drive).

**Question/answer format**:
- MCQ is most common: 3 options (MVBench, PerceptionTest), 4 options (Video-MME, LongVideoBench), 5 options (EgoSchema, NExT-QA), variable 3-8 (Video-MME v2).
- Open-ended / free-form: ActivityNet-QA, MSVD-QA, MSRVTT-QA (answer from 1K vocabulary).
- Mixed: TGIF-QA (MCQ + counting + open-ended), TempCompass (MCQ + yes/no + captioning + caption matching), MLVU (MCQ + generation).

**Split/config conventions**:
- Most provide `test` split only (Video-MME, EgoSchema, TempCompass).
- Some have `train`/`val`/`test` (ActivityNet-QA, TGIF-QA) or task-based configs/subsets (MVBench: 20 configs, TempCompass: 4 configs).
- Answer labels may be withheld on the test split (EgoSchema, PerceptionTest).

**Video ID mapping**:
- Some datasets use integer IDs (`video_0`, `video_6358`), others use UUIDs (EgoSchema), YouTube IDs (Video-MME), or content-hash filenames.
- Mapping from annotation to video file varies: direct filename match, ID-to-filename mapping file (`map_vid_vidorID.json` for NExT-QA), or convention (`video{N}.mp4`).

**Duration distribution**:
- Short clips (< 1 min): MVBench, TGIF-QA, MSVD-QA, TempCompass, PerceptionTest.
- Medium (1–15 min): EgoSchema, ActivityNet-QA, NExT-QA, MLVU (avg 15 min).
- Long-form (> 15 min): Video-MME (up to 1 hr), LongVideoBench (up to 1 hr), MLVU (up to 2+ hr).

**Frame sampling**:
- Explicitly specified in few datasets. MVBench uses TSN sampling (16 frames). LongVideoBench provides a `max_num_frames` parameter. Most leave it to the evaluator.
- Long videos benefit from denser sampling or subtitle-aligned frames (Video-MME).

**Prompt format**:
- MCQ datasets typically provide a question + candidate list. The prompt format varies: `(A) opt` vs `A. opt` vs plain list.
- Video-MME provides a `post_prompt` field: "Answer with the option's letter from the given choices directly."
- Most datasets do not ship an official model-input prompt, leaving it to the evaluator.

### 1.3 Known difficulties

- **YouTube takedowns**: Video-MME, ActivityNet-QA, MSRVTT lose videos over time.
- **Gated access**: LongVideoBench (CC-BY-NC-SA), MLVU (CC-BY-NC-SA), EgoSchema (Ego4D license), MSRVTT-QA (auth required). Check gating before automating downloads.
- **Large total size**: MLVU (430 GB), Video-MME v2 (177 GB), LongVideoBench (162 GB), ActivityNet-QA (130 GB). Disk and bandwidth constraints are real.
- **Archive formats**: Video-MME v2 stores 40 numbered zip files; LongVideoBench uses split tar archives. Extraction strategy matters.
- **Missing ground truth**: EgoSchema (public subset: 500 answers only), PerceptionTest test split (challenge server submission).

---

## 2. mm-eval video storage standard

### 2.1 Storage layout

Video datasets in mm-eval use **local mode** (`--mode local`) as the primary format. Videos are stored as individual files under the `media/` directory, the same location used for images.

```
<artifact>/
├── data.json           # annotation array
├── media/              # video files (and images for mixed datasets)
│   ├── vid_001.mp4
│   ├── vid_002.mp4
│   └── ...
└── metadata.json       # manifest
```

For HF distribution of video datasets, mm-eval supports a **video-aware HF mode** (`--mode hf --hf-video`):

```
<hf-repo>/
├── data/               # default config parquet shards (Arrow)
├── videos/             # video files (flat or sharded)
│   ├── vid_001.mp4
│   └── ...
├── metadata.json       # manifest (with video_storage block)
└── .gitattributes
```

In HF video mode, the Arrow dataset stores:
- `id`: string
- `media`: `Sequence(Value('string'))` — video filenames relative to the repo root (e.g., `videos/vid_001.mp4`)
- `messages`: string (JSON) — same as image format

The dataloader resolves video paths by downloading them from the HF repo via `hf_hub_download()`.

### 2.2 Video file conventions

- **Filename**: `{id}_{media_idx}.{ext}` in converted artifacts (mirrors image naming).
- **Extensions**: Must be a recognized video extension (see `VIDEO_EXTS`). Files without extensions are rejected.
- **Codec**: No constraint enforced; MP4/H.264 recommended for broadest compatibility.
- **Resolution**: Preserve source resolution. Do not re-encode unless the user explicitly requests it.
- **Duration**: No hard limit, but datasets with videos > 30 min should document expected inference time.

### 2.3 When archives are acceptable

Archives (zip/tar) are acceptable for **distribution only** — they reduce HF upload overhead for repos with thousands of small files. Rules:

1. Archives must be **per-split or per-shard**, not monolithic. A single 400 GB zip is unacceptable.
2. The `metadata.json` must record `video_storage.archive_format` (`zip` or `tar`) and `video_storage.archive_files` (list of archive names).
3. The converter or a setup script must extract archives into the `media/` or `videos/` directory before evaluation.
4. Inner paths within archives must match the filenames in `data.json` / the Arrow `media` column.
5. Random access within archives is not supported by the dataloader; always extract first.
6. After extraction, the archive files may be deleted to save space (document this in the skill).

### 2.4 Video download and ID mapping

When converting a video dataset, the converter must:

1. **Preserve official video IDs** as the source identifier. Use the original video ID (YouTube ID, dataset-native ID, filename stem) as the base for the mm-eval row `id`.
2. **Create deterministic mm-eval IDs** when source IDs are not row-unique (e.g., multiple QA per video): use composite IDs `{video_id}_q{k}` (same convention as image datasets).
3. **Map each data entry** to its video file unambiguously: store the video filename in the `media` list on the data entry.
4. **Handle missing/unavailable videos** explicitly:
   - Log each missing video with its ID and reason (takedown, gated, download failure).
   - Skip the row; `convert.py` records it under `encode_failed:{video_id}` in `convert_summary.json`'s `skip_reason_counts` (videos with an unrecognized extension are grouped under `unknown_video_ext` instead).
   - Never silently drop rows — the skip count must be inspectable.
5. **Verify downloaded videos**:
   - File exists and size > 0.
   - Extension matches a known video format.
   - (Optional, when `av` / `decord` / `opencv-python` is available) Probe the file: check it opens, has > 0 frames, and the duration is non-zero.

### 2.5 Metadata schema for video datasets

The `metadata.json` per-subset block gains an optional `video_storage` object:

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
        "origin": "official",
        "reference": "Video-MME official post_prompt (MME-Benchmarks/Video-MME eval code)"
      },
      "video_storage": {
        "format": "files",
        "media_root": "media",
        "archive_format": null,
        "archive_files": [],
        "notes": "Videos downloaded from YouTube via yt-dlp"
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
          "duration_category": { "from": "duration" },
          "task_type_source": { "from": "task_type" }
        }
      }
    }
  }
}
```

`video_storage` fields:

| Field | Type | Description |
|-------|------|-------------|
| `format` | string | `"files"` (loose files) or `"archives"` (zip/tar that must be extracted first) |
| `media_root` | string | Directory containing video files, relative to repo root. Default: `"media"` (local) or `"videos"` (HF). |
| `archive_format` | string or null | `"zip"`, `"tar"`, `"tar.gz"`, or `null` if `format == "files"`. |
| `archive_files` | list of string | Archive filenames when `format == "archives"`. Empty otherwise. |
| `notes` | string | Free-text notes about video sourcing, licensing, download instructions. |

### 2.6 Prompt and placeholder conventions

- **`<video>`** is the canonical placeholder for video media in prompts. One `<video>` per video file.
- **`<image>`** and **`<video>`** must not be mixed in a single placeholder-count validation unless the modality is `multi_image_video_interleave`.
- Do not confuse dataset-native textual markers (e.g., `[video]`, `{video}`, or question text mentioning "the video") with actual media placeholders.
- When the dataset provides an official prompt (e.g., Video-MME's `post_prompt`), use it verbatim.
- When no official prompt exists, use template T6 (video MCQ) or T8 (video open-ended VQA) from `jinja-templates.md`.

### 2.7 Frame sampling

Frame sampling is the responsibility of the **model inference backend**, not the dataloader or converter. The dataloader passes the video file path (or URL) to the model, which applies its own frame sampling strategy.

However, the metadata should document expected frame-sampling behavior when specified by the benchmark:

```json
"video_storage": {
  "notes": "Frame sampling: TSN sampling, 16 frames recommended (MVBench paper)"
}
```

The dataloader provides a hook point for frame sampling via `load_media()`: when it encounters a video path, it returns the path as-is. Model backends that need frames can decode them at inference time.

---

## 3. Dataloader design for video

### 3.1 Current state

The existing dataloader already supports video:
- `BaseDataset._is_video()` checks extensions.
- `BaseDataset.load_media()` returns video paths as strings (no decoding).
- `_process_messages()` handles `<video>` placeholders by passing the path through.
- `LocalJSONDataset` resolves video paths via `media_dir`.

### 3.2 Required enhancements

1. **Video path resolution for HF datasets**: `MMEvalHFDataset` must resolve video filenames from the HF repo. When `media` contains string paths (not PIL images), download the video file via `hf_hub_download()` and return the local path.

2. **Video integrity validation**: Add `validate_video()` method that checks file existence, size > 0, and optionally probes with `av` or `decord`.

3. **Archive-aware path resolution**: When `video_storage.format == "archives"`, the dataloader should check whether videos are extracted and raise a clear error if not, pointing the user at the extraction step.

4. **Lazy loading**: Videos must never be loaded into memory by the dataloader. The `load_media()` method already returns paths for video — this contract must be preserved.

5. **Media count validation**: The existing placeholder/media count check in `_process_messages()` already covers video. No change needed.

6. **Clear error messages**: When a video file is missing or unreadable, the error must include the video path, the row ID, and the expected location.

### 3.3 Backward compatibility

All changes must be backward-compatible with existing image datasets:
- `Sequence(Image())` continues to work for image datasets.
- `Sequence(Value('string'))` is used only for video datasets.
- The dataloader auto-detects the media type from the column schema or from the file extension.
- No existing image dataset behavior changes.
