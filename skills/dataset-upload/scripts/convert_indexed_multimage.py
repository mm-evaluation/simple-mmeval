#!/usr/bin/env python3
"""Convert MMMU-style indexed multi-image benchmarks (MMMU/MMMU, MMMU/MMMU-Pro)
into Simple-MMEval HF format.

Source: ``MMMU/MMMU`` (default) or ``MMMU/MMMU-Pro`` via ``--hf`` — one HF config
per subject, with ``image_1``…``image_7`` columns and ``question`` text using
``<image N>`` references.

Output (per split): an mm-eval artifact at ``<out>/<split>/`` consisting of
``hf_dataset/`` (DatasetDict for the ``default`` config) plus a top-level
``metadata.json`` manifest whose single ``main`` subset carries the Jinja
template that mirrors the official MMMU paper / eval-code prompt format
verbatim:

  Multiple-choice (MMMU/configs/llava1.5.yaml + utils/data_utils.py):
      {question}\\n\\n(A) opt1\\n(B) opt2\\n…\\n\\n\\nAnswer with the option's letter from the given choices directly.

  Short-answer:
      {question}\\n\\nAnswer the question using a single word or phrase.

Per row we:
  1. Collect ``<image N>`` refs from the question and options in textual order.
  2. Build a media list with one entry per reference (``media[k] = image_{refs[k]}``).
  3. Normalize the question and option strings: ``<image N>`` → ``<image>``.
  4. Store messages with ``role=user``, ``question`` (normalized), ``options``
     (letter-keyed dict, normalized), and ``answer``. The Jinja template
     ASSEMBLES the prompt at runtime — no pre-rendered ``prompt`` field is
     stored on the message.

Subjects are loaded in parallel with one HF config each. Each per-split
``<out>/<split>/`` directory is directly consumable by ``merge_splits.py``
(which expects ``hf_dataset/`` + ``metadata.json``) and ``push_to_hf.py``.
Because ``metadata.json`` has a single subset (``main``), runtime selection
is just ``--dataset mmeval_hf@<repo>`` with no ``--subset`` needed.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.stdout.reconfigure(line_buffering=True)

IMG_REF_RE = re.compile(r"<image (\d+)>")
IMG_COLUMNS = [f"image_{i}" for i in range(1, 8)]

# All 30 MMMU subject configs (the entire upstream).
MMMU_SUBJECTS = [
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art",
    "Art_Theory", "Basic_Medical_Science", "Biology", "Chemistry",
    "Clinical_Medicine", "Computer_Science", "Design",
    "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics",
    "Energy_and_Power", "Finance", "Geography", "History", "Literature",
    "Manage", "Marketing", "Materials", "Math", "Mechanical_Engineering",
    "Music", "Pharmacy", "Physics", "Psychology", "Public_Health", "Sociology",
]

# Jinja template that produces output IDENTICAL to the official MMMU eval
# code (MMMU/utils/data_utils.py + MMMU/configs/llava1.5.yaml). The empty
# task_instructions case is the default in the official config.
MMMU_TEMPLATE = (
    "{{ question }}"
    "{% if options %}\n\n"
    "{% for k, v in options.items() %}({{ k }}) {{ v }}\n{% endfor %}"
    "\nAnswer with the option's letter from the given choices directly."
    "{% else %}\n\n"
    "Answer the question using a single word or phrase."
    "{% endif %}"
)


def parse_options(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    try:
        v = ast.literal_eval(raw)
        if isinstance(v, list):
            return [str(x) for x in v]
    except Exception:
        pass
    return []


def normalize_image_refs(s: str) -> str:
    return IMG_REF_RE.sub("<image>", s or "")


def write_image(img, out_dir: Path, stem: str, idx: int,
                image_format: str, jpeg_quality: int) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    if image_format == "jpeg":
        if img.mode != "RGB":
            img = img.convert("RGB")
        name = f"{stem}_{idx}.jpg"
        img.save(out_dir / name, format="JPEG", quality=jpeg_quality, optimize=False)
    else:
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        name = f"{stem}_{idx}.png"
        img.save(out_dir / name, format="PNG", optimize=False, compress_level=1)
    return name


def process_row(row: Dict[str, Any], stage_dir: Path, image_format: str,
                jpeg_quality: int) -> Optional[Dict[str, Any]]:
    rid = row["id"]
    stem = rid.replace("/", "_")
    options_raw = parse_options(row.get("options"))
    question_raw = row.get("question") or ""
    qtype = row.get("question_type") or ""

    refs = [int(m) for m in IMG_REF_RE.findall(question_raw)]
    for o in options_raw:
        refs.extend(int(m) for m in IMG_REF_RE.findall(o))

    images = [row.get(c) for c in IMG_COLUMNS]
    if not refs:
        media_pils = [im for im in images if im is not None]
    else:
        media_pils = []
        for r in refs:
            if r < 1 or r > len(images) or images[r - 1] is None:
                return None
            media_pils.append(images[r - 1])

    saved: List[str] = []
    for i, pil in enumerate(media_pils):
        try:
            saved.append(write_image(pil, stage_dir, stem, i, image_format, jpeg_quality))
        except Exception:
            return None

    question = normalize_image_refs(question_raw)
    if qtype == "multiple-choice":
        options_norm = {chr(ord("A") + i): normalize_image_refs(v)
                        for i, v in enumerate(options_raw)}
    else:
        options_norm = {}

    # Sanity: total <image> tokens the template will emit must equal media count.
    total_tokens = (
        question.count("<image>")
        + sum(v.count("<image>") for v in options_norm.values())
    )
    if total_tokens != len(saved):
        return None

    msg = {
        "role": "user",
        "question": question,
        "answer": row.get("answer") or "",
        "options": options_norm,
        "hint": "",
        "choices": [],
        "question_type": qtype,
        "subfield": row.get("subfield") or "",
        "img_type": row.get("img_type") or row.get("image_type") or "",
        "topic_difficulty": row.get("topic_difficulty") or "",
        "explanation": row.get("explanation") or "",
    }
    return {
        "id": rid,
        "media_basenames": saved,
        "messages_json": json.dumps([msg], ensure_ascii=False),
    }


def load_split_for_subjects(hf: str, subject: str, split: str) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    try:
        ds = load_dataset(hf, subject, split=split)
    except Exception as e:
        print(f"  [{subject}/{split}] load failed: {e}", flush=True)
        return []
    return [dict(r) for r in ds]


def convert_split(args, split: str) -> Dict[str, Any]:
    from datasets import Dataset, DatasetDict, Sequence, Image, Value, Features

    print(f"\n=== {split} ===", flush=True)

    out_dir = Path(args.out) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = out_dir / "_hf_images"
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Load all subjects in parallel for this split.
    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(8, len(args.subjects))) as pool:
        futures = {pool.submit(load_split_for_subjects, args.hf, s, split): s
                   for s in args.subjects}
        for fut in futures:
            sub_rows = fut.result()
            rows.extend(sub_rows)
            print(f"  loaded {futures[fut]}/{split}: {len(sub_rows)} rows", flush=True)
    print(f"total loaded: {len(rows)} rows in {time.time()-t0:.1f}s", flush=True)

    if args.limit:
        rows = rows[:args.limit]

    n = len(rows)
    results: List[Optional[Dict[str, Any]]] = [None] * n

    def worker(i: int) -> None:
        results[i] = process_row(rows[i], stage_dir, args.image_format, args.jpeg_quality)

    t1 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, _ in enumerate(pool.map(worker, range(n), chunksize=8), start=1):
            if i % 500 == 0 or i == n:
                el = time.time() - t1
                print(f"  processed {i}/{n} ({i/el:.1f} rows/s)", flush=True)

    kept = [r for r in results if r is not None]
    skipped = n - len(kept)
    print(f"kept {len(kept)} skipped {skipped}", flush=True)

    kept.sort(key=lambda r: r["id"])
    ids = [r["id"] for r in kept]
    media_col = [
        [{"path": None, "bytes": (stage_dir / b).read_bytes()} for b in r["media_basenames"]]
        for r in kept
    ]
    messages_col = [r["messages_json"] for r in kept]

    features = Features({
        "id": Value("string"),
        "media": Sequence(Image()),
        "messages": Value("string"),
    })
    out_ds = Dataset.from_dict(
        {"id": ids, "media": media_col, "messages": messages_col},
        features=features,
    )
    DatasetDict({split: out_ds}).save_to_disk(str(out_dir / "hf_dataset"))

    media_min = min(media_counts) if (media_counts := [len(m) for m in media_col]) else 0
    media_max = max(media_counts) if media_counts else 0
    metadata = {
        "name": Path(args.hf).name,
        "release_date": args.release_date,
        "subsets": {
            "main": {
                "language": ["en"],
                "modalities": ["multi_image_interleave"],
                "task_type": "multiple_choice_vqa",
                "prompt_template": MMMU_TEMPLATE,
                "prompt_template_source": {
                    "origin": args.template_source_origin,
                    "reference": args.template_source_ref,
                    "notes": args.template_source_notes,
                },
                "mapping_from_source": {
                    "source": {
                        "format": "huggingface",
                        "url": {split: f"https://huggingface.co/datasets/{args.hf}"},
                    },
                    "id": {"from": "id"},
                    "question": {"from": "question"},
                    "options": {"from": "options", "optional": True},
                    "answer": {"from": "answer", "optional": True},
                    "media": {
                        "from": "image_1..image_7",
                        "type": "list",
                        "min_items": media_min,
                        "max_items": media_max,
                    },
                },
            }
        },
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return {
        "split": split,
        "rows": len(kept),
        "skipped": skipped,
        "elapsed_s": round(time.time() - t0, 2),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf", default="MMMU/MMMU")
    p.add_argument("--subjects", nargs="+", default=MMMU_SUBJECTS)
    p.add_argument("--splits", nargs="+", default=["dev", "validation", "test"])
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--image-format", choices=["png", "jpeg"], default="jpeg")
    p.add_argument("--jpeg-quality", type=int, default=92)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--release-date", default=None,
                   help="metadata.json release_date (default: today UTC date)")
    p.add_argument("--template-source-origin", default="official",
                   help="prompt_template_source.origin (default: official — MMMU_TEMPLATE is "
                        "byte-for-byte the official MMMU eval-code prompt)")
    p.add_argument("--template-source-ref",
                   default="https://github.com/MMMU-Benchmark/MMMU "
                           "(utils/data_utils.py + configs/llava1.5.yaml)",
                   help="prompt_template_source.reference")
    p.add_argument("--template-source-notes", default="",
                   help="Optional prompt_template_source.notes")
    args = p.parse_args()
    if args.release_date is None:
        from datetime import date as _date
        args.release_date = _date.today().isoformat()

    summary = {"hf": args.hf, "subjects": args.subjects, "splits": []}
    for s in args.splits:
        summary["splits"].append(convert_split(args, s))

    with open(Path(args.out) / "convert_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
