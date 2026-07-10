#!/usr/bin/env python3
"""Convert any source dataset into Simple-MMEval-runnable format.

Output modes:
  - local: writes <out>/data.json + <out>/media/  (use with --dataset local@json)
  - hf:    writes <out>/hf_dataset/ + <out>/metadata.json  (push_to_hub-ready)
  - both:  emits both in a single source pass.

Source: HuggingFace (--hf, requires --split), local JSON (--json), TSV (--tsv),
or CSV (--csv). Column mapping is ``--map`` or loaded from ``--metadata-json``.

Column mapping is given as a list of `canonical=source` pairs:
  --map id=question_id question=question image=image answer=answers

Canonical keys understood by this script:
  id, question, answer, hint, options, choices,
  image | images | media   (single media field — single value or list).

Any other `--map foo=bar` pairs are passed through to the message dict;
pass-through values that are None are dropped (so the Jinja template never
sees the literal string "None").

No-fabrication contract:
  - row with id=None / missing → skipped, counted as `missing_required:id`
    (no `row_{i}` fallback);
  - video URL with no extension → skipped, counted as `unknown_video_ext`
    (no `.mp4` fallback);
  - HF mode + video media → ValueError pointing the user at --mode local.

Examples
--------

# 1. VizWiz-VQA val -> local JSON + media
python3 convert.py \\
    --hf lmms-lab/VizWiz-VQA --split val \\
    --map id=question_id question=question image=image answer=answers \\
    --template '<image>{{ question }}
Answer the question using a single word or phrase.' \\
    --mode local --out /tmp/vizwiz_val_local --workers 16

# 2. VizWiz-VQA val -> HF dataset (push_to_hub ready)
python3 convert.py \\
    --hf lmms-lab/VizWiz-VQA --split val \\
    --map id=question_id question=question image=image answer=answers \\
    --template '<image>{{ question }}
Answer the question using a single word or phrase.' \\
    --mode hf --out /tmp/vizwiz_val_hf
"""
from __future__ import annotations

import argparse
import base64
import copy
import csv
import io
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests
from jinja2 import Environment, Template
from PIL import Image

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv",
              ".mpeg", ".mpg", ".m4v", ".3gp", ".3g2", ".ts", ".mts", ".vob"}

# Schema fields handled explicitly by build_message; anything else in colmap
# is treated as a user-defined pass-through.
SCHEMA_KEYS = {"id", "image", "images", "media",
               "question", "answer", "hint", "options", "choices"}

# Skip-reason keys that carry a free-form suffix after ":" — bucket counts by prefix.
_GROUPED_REASON_PREFIXES = ("encode_failed", "unknown_video_ext")

MODALITY_ORDER = (
    "single_image_start", "single_video_start",
    "multi_image_start", "multi_image_interleave",
    "multi_video_interleave", "multi_image_video_interleave", "text",
)


def _skip_category(reason: str) -> str:
    """Group encode_failed:... / unknown_video_ext:... by prefix; keep stable
    categories like missing_required:id as-is."""
    for prefix in _GROUPED_REASON_PREFIXES:
        if reason == prefix or reason.startswith(f"{prefix}:"):
            return prefix
    return reason


# Force unbuffered stdout so background invocations stream their progress.
sys.stdout.reconfigure(line_buffering=True)


class EncodeFailed(Exception):
    """A media item could not be encoded; the row is skipped with this reason."""
    category = "encode_failed"


class UnknownVideoExt(EncodeFailed):
    """Video URL/path has no extension; we refuse to assume one."""
    category = "unknown_video_ext"


def is_video_path(path: str) -> bool:
    return os.path.splitext(path.split("?")[0])[-1].lower() in VIDEO_EXTS


def video_path_of(value: Any) -> Optional[str]:
    """Return the video path/URL string if `value` looks like a video, else None."""
    if isinstance(value, str) and is_video_path(value):
        return value
    if isinstance(value, dict):
        path = value.get("path")
        if isinstance(path, str) and is_video_path(path):
            return path
    return None


def parse_map_pairs(pairs: Optional[List[str]], *, require_id: bool = True,
                    require_question: bool = True) -> Dict[str, str]:
    """Parse ``canonical=source`` pairs.

    - ``require_id=False`` skips the ``id`` requirement (e.g. overrides on top of
      ``--metadata-json``, or when ``--auto-id`` will synthesize a row id).
    - ``require_question=False`` skips the ``question`` requirement — captioning
      tasks have synthetic prompts that do not consume ``question`` at render time.
    """
    out: Dict[str, str] = {}
    if not pairs:
        missing = []
        if require_id: missing.append("id=<source-field>")
        if require_question: missing.append("question=<source-field>")
        if missing:
            raise ValueError("--map must include " + " and ".join(missing))
        return out
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--map entry '{p}' must be canonical=source")
        k, v = p.split("=", 1)
        k, v = k.strip(), v.strip()
        if not k:
            raise ValueError(f"--map entry '{p}' has empty canonical key")
        if not v:
            raise ValueError(f"--map entry '{p}' has empty source field")
        if k in out:
            raise ValueError(f"--map has duplicate canonical key '{k}'")
        out[k] = v
    if require_id and "id" not in out:
        raise ValueError("--map must include id=<source-field>")
    if require_question and "question" not in out:
        raise ValueError("--map must include question=<source-field>")
    return out


def parse_map(pairs: List[str], *, task_type: Optional[str] = None,
              has_auto_id: bool = False) -> Dict[str, str]:
    """Convenience wrapper. Captioning tasks don't require a `question` mapping.
    `--auto-id` callers synthesize their own ids, so `id` mapping is optional."""
    return parse_map_pairs(
        pairs,
        require_id=not has_auto_id,
        require_question=(task_type != "captioning"),
    )


def load_metadata_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _subset_block(meta: Dict[str, Any], subset: str) -> Tuple[str, Dict[str, Any]]:
    """Resolve subset name (single-subset fallback) and return ``(resolved_name, block)``."""
    subsets = meta.get("subsets") or {}
    resolved = subset
    if resolved not in subsets:
        if len(subsets) == 1:
            resolved = next(iter(subsets))
        else:
            raise ValueError(f"subset {subset!r} not in metadata subsets {list(subsets)}")
    block = subsets[resolved]
    if not isinstance(block, dict):
        raise ValueError(f"subsets[{resolved!r}] must be an object")
    return resolved, block


def colmap_from_mapping(
    mapping: Dict[str, Any],
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """Build colmap + optional MCQ choice_specs from mapping_from_source."""
    colmap: Dict[str, str] = {}
    choice_specs: List[Dict[str, Any]] = []
    for key, val in mapping.items():
        if key in ("source", "extra"):
            continue
        if key == "choices" and isinstance(val, list):
            choice_specs = [x for x in val if isinstance(x, dict) and "from" in x and "key" in x]
            continue
        if key == "choices" and isinstance(val, dict) and "from" in val:
            src = val["from"]
            if isinstance(src, str) and src.strip():
                colmap["choices"] = src
            continue
        if key == "media" and isinstance(val, dict) and "from" in val:
            src = val["from"]
            if not isinstance(src, str) or not src.strip():
                raise ValueError("mapping_from_source.media.from must be a non-empty string")
            colmap["image"] = src
            continue
        if isinstance(val, dict) and "from" in val:
            src = val["from"]
            if not isinstance(src, str) or not src.strip():
                raise ValueError(f"mapping_from_source.{key}.from must be a non-empty string")
            colmap[key] = src
            continue
    extra = mapping.get("extra")
    if isinstance(extra, dict):
        for _ek, ev in extra.items():
            if isinstance(ev, dict) and "from" in ev:
                colmap[str(_ek)] = str(ev["from"])
    if "id" not in colmap or "question" not in colmap:
        raise ValueError("metadata mapping_from_source must include id and question (.from)")
    return colmap, choice_specs


def merge_colmap(base: Dict[str, str], override: Dict[str, str]) -> Dict[str, str]:
    out = dict(base)
    out.update(override)
    return out


def load_template(template: Optional[str]) -> Optional[str]:
    if not template:
        return None
    if os.path.exists(template):
        with open(template, encoding="utf-8") as f:
            return f.read()
    return template


def build_jinja_env() -> Environment:
    env = Environment()
    env.globals.update({"zip": zip, "enumerate": enumerate, "len": len, "range": range,
                        "list": list, "dict": dict, "str": str, "int": int, "float": float,
                        "bool": bool, "sum": sum, "max": max, "min": min})
    return env


def _value_to_pil(value: Any) -> Image.Image:
    """Decode an image-like value into a PIL.Image; raises EncodeFailed on any failure.

    Does not handle videos — caller is expected to route those via video_path_of()."""
    if value is None or value == "":
        raise EncodeFailed("empty media value")
    if hasattr(value, "size") and hasattr(value, "mode"):
        return value
    if isinstance(value, dict):
        if value.get("bytes"):
            try:
                return Image.open(io.BytesIO(value["bytes"]))
            except Exception as e:
                raise EncodeFailed(f"image dict bytes decode failed: {e!s}") from e
        if value.get("path"):
            try:
                return Image.open(value["path"])
            except Exception as e:
                raise EncodeFailed(f"image dict path open failed: {value['path']!r}: {e!s}") from e
        raise EncodeFailed(f"image dict missing bytes/path: keys={list(value.keys())[:6]}")
    if isinstance(value, str):
        if os.path.exists(value):
            try:
                return Image.open(value)
            except Exception as e:
                raise EncodeFailed(f"image file open failed: {value[:80]!r}: {e!s}") from e
        if value.startswith(("http://", "https://")):
            try:
                r = requests.get(value, timeout=30)
                r.raise_for_status()
                return Image.open(io.BytesIO(r.content))
            except Exception as e:
                raise EncodeFailed(f"http image fetch failed: {value[:80]!r}: {e!s}") from e
        try:
            return Image.open(io.BytesIO(base64.b64decode(value)))
        except Exception as e:
            raise EncodeFailed(f"base64 decode failed: {value[:60]}") from e
    raise EncodeFailed(f"unsupported image value type {type(value).__name__}")


def encode_image_bytes(value: Any, image_format: str, jpeg_quality: int) -> Tuple[bytes, str]:
    """Encode an image-like value into (bytes, ext). Pure in-memory; never touches disk."""
    img = _value_to_pil(value)
    buf = io.BytesIO()
    if image_format == "jpeg":
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=False)
        return buf.getvalue(), ".jpg"
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")
    img.save(buf, format="PNG", optimize=False, compress_level=1)
    return buf.getvalue(), ".png"


def materialize_video(path_or_url: str, out_dir: Path, stem: str, idx: int,
                      verify: bool = False) -> str:
    """Copy/download a video to local disk; returns the basename.

    When ``verify=True``, checks that the materialized file is non-empty and
    (when PyAV is available) can be opened and contains at least one video stream.
    """
    ext = os.path.splitext(path_or_url.split("?")[0])[-1].lower()
    if not ext:
        raise UnknownVideoExt(f"video has no extension: {path_or_url[:80]}")
    if ext not in VIDEO_EXTS:
        raise UnknownVideoExt(f"unrecognized video extension {ext!r}: {path_or_url[:80]}")
    name = f"{stem}_{idx}{ext}"
    target = out_dir / name
    try:
        if path_or_url.startswith(("http://", "https://")):
            with requests.get(path_or_url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(target, "wb") as f:
                    for chunk in r.iter_content(65536):
                        f.write(chunk)
        else:
            if not os.path.isfile(path_or_url):
                raise FileNotFoundError(f"video source file not found: {path_or_url!r}")
            shutil.copyfile(path_or_url, target)
    except EncodeFailed:
        raise
    except Exception as e:
        raise EncodeFailed(f"video materialize failed: {path_or_url[:80]!r}: {e!s}") from e
    if verify:
        _verify_video_file(target)
    return name


def _verify_video_file(path: Path) -> None:
    """Quick integrity check: file is non-empty and decodable."""
    if not path.exists():
        raise EncodeFailed(f"video file missing after write: {path}")
    if path.stat().st_size == 0:
        raise EncodeFailed(f"video file is empty (0 bytes): {path}")
    try:
        import av
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise EncodeFailed(f"video has no video stream: {path}")
    except ImportError:
        pass
    except EncodeFailed:
        raise
    except Exception as e:
        raise EncodeFailed(f"video probe failed for {path}: {e}") from e


def _media_path_str(m: Any) -> Optional[str]:
    """HTTP(S) URL or local filesystem path string from a media cell."""
    if isinstance(m, str):
        return m
    if isinstance(m, dict):
        p = m.get("path")
        return p if isinstance(p, str) else None
    return None


def options_from_choice_specs(row: Dict[str, Any], specs: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for s in specs:
        key = s.get("key")
        src = s.get("from")
        if not isinstance(key, str) or not isinstance(src, str):
            continue
        val = row.get(src)
        if val is None or (isinstance(val, str) and not val.strip()):
            continue
        out[key] = val
    return out


def build_message(
    row: Dict[str, Any],
    colmap: Dict[str, str],
    answer_join: str,
    answer_as_list: bool,
    choice_specs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build the canonical message dict. Empty schema fields keep their empty
    representations (`""` / `{}` / `[]`); pass-through fields with None values
    are dropped to avoid the Jinja template rendering the literal string "None"."""
    msg: Dict[str, Any] = {"role": "user"}
    # `question` is optional for captioning-style tasks; default to "" when
    # the mapping omits it.
    msg["question"] = (row.get(colmap["question"], "") or "") if "question" in colmap else ""
    if "answer" in colmap:
        ans = row.get(colmap["answer"])
        if isinstance(ans, list) and not answer_as_list:
            ans = answer_join.join(str(a) for a in ans) if ans else ""
        msg["answer"] = ans if ans is not None else ""
    else:
        msg["answer"] = ""
    msg["hint"] = (row.get(colmap["hint"]) or "") if "hint" in colmap else ""

    if choice_specs:
        msg["options"] = options_from_choice_specs(row, choice_specs)
        if "options" in colmap:
            extra = row.get(colmap["options"])
            if isinstance(extra, dict):
                msg["options"] = {**msg["options"], **extra}
            elif isinstance(extra, list):
                d = {chr(ord("A") + i): v for i, v in enumerate(extra)}
                msg["options"] = {**msg["options"], **d}
    elif "options" in colmap:
        opts = row.get(colmap["options"]) or {}
        if isinstance(opts, list):
            opts = {chr(ord("A") + i): v for i, v in enumerate(opts)}
        msg["options"] = opts
    else:
        msg["options"] = {}

    if "choices" in colmap:
        ch = row.get(colmap["choices"]) or []
        msg["choices"] = list(ch) if not isinstance(ch, list) else ch
    else:
        msg["choices"] = []
    for k, src in colmap.items():
        if k in SCHEMA_KEYS:
            continue
        val = row.get(src)
        if val is not None:
            msg[k] = val
    return msg


def _row_media_profile(media_raw: List[Any], expects_video: bool) -> Dict[str, Any]:
    """Per-row modality signal: n_media, n_video_items, n_image_items."""
    n_vid = n_img = 0
    for m in media_raw:
        v = video_path_of(m)
        if v is None and expects_video:
            ps = _media_path_str(m)
            if ps and (ps.startswith("http") or os.path.exists(ps)):
                ext_raw = os.path.splitext(ps.split("?")[0])[-1].lower()
                if not ext_raw:
                    v = ps
        if v is not None:
            n_vid += 1
        elif m not in (None, ""):
            n_img += 1
    return {"n_media": len(media_raw), "n_vid": n_vid, "n_img": n_img}


def infer_modalities(signals: List[Dict[str, Any]]) -> List[str]:
    """Infer modalities from per-row media profiles (id-valid rows only)."""
    if not signals:
        return ["text"]
    any_text = any(s["n_img"] + s["n_vid"] == 0 for s in signals)
    any_multi_img = any(s["n_img"] >= 2 and s["n_vid"] == 0 for s in signals)
    any_vid = any(s["n_vid"] > 0 for s in signals)
    any_img = any(s["n_img"] > 0 for s in signals)
    row_mixed = any(s["n_vid"] > 0 and s["n_img"] > 0 for s in signals)
    rows_only_vid = [s for s in signals if s["n_vid"] > 0 and s["n_img"] == 0]
    rows_only_img = [s for s in signals if s["n_img"] > 0 and s["n_vid"] == 0]
    dataset_mixed_vid_img = bool(rows_only_vid and rows_only_img)

    modes: List[str] = []
    if row_mixed or dataset_mixed_vid_img:
        modes.append("multi_image_video_interleave")
    elif any_vid:
        modes.append("single_video_start")
    elif any_multi_img:
        modes.append("multi_image_start")
    elif any_img:
        modes.append("single_image_start")
    if any_text:
        modes.append("text")
    if not modes:
        modes = ["text"]
    seen = set()
    ordered: List[str] = []
    for tag in MODALITY_ORDER:
        if tag in modes and tag not in seen:
            ordered.append(tag)
            seen.add(tag)
    return ordered


def infer_source_format(args: Any) -> str:
    if args.hf:
        return "huggingface"
    if getattr(args, "tsv", None):
        return "tsv"
    if getattr(args, "csv", None):
        return "csv"
    return "json"


def build_mapping_from_colmap(
    colmap: Dict[str, str],
    choice_specs: List[Dict[str, Any]],
    media_min: int,
    media_max: int,
    args: Any,
) -> Dict[str, Any]:
    """Serialize colmap into mapping_from_source (excluding top-level source block).

    ``id.from`` uses the original source column (``args._original_id_field``) so
    ``--explode`` does not leak the synthetic ``_unique_id`` field into metadata.
    """
    m: Dict[str, Any] = {}
    media_src = None
    for k in ("images", "media", "image"):
        if k in colmap:
            media_src = colmap[k]
            break
    if media_src:
        m["media"] = {
            "from": media_src,
            "type": "list",
            "min_items": media_min,
            "max_items": media_max,
        }
    orig_id = getattr(args, "_original_id_field", None) or colmap.get("id")
    if orig_id:
        id_ent: Dict[str, Any] = {"from": orig_id}
        if getattr(args, "explode", None):
            id_ent["explode_from"] = args.explode
            id_ent["id_template"] = getattr(args, "id_template", "{id}_q{idx}")
        m["id"] = id_ent
    for key in ("question", "answer", "hint", "options"):
        if key in colmap:
            ent: Dict[str, Any] = {"from": colmap[key]}
            if key in ("answer", "hint", "options"):
                ent["optional"] = True
            if key == "options":
                ent["note"] = "list source values are normalized to {A,B,...} dict"
            m[key] = ent
    if choice_specs:
        m["choices"] = choice_specs
    elif "choices" in colmap:
        m["choices"] = {"from": colmap["choices"]}
    extra = {k: {"from": v} for k, v in colmap.items()
             if k not in SCHEMA_KEYS and k not in m}
    if extra:
        m["extra"] = extra
    return m


def _inferred_task_type(colmap: Dict[str, str], choice_specs: List[Dict[str, Any]]) -> str:
    return (
        "multiple_choice_vqa"
        if (choice_specs or "options" in colmap or "choices" in colmap)
        else "vqa"
    )


def _prompt_template_source_from_args(args: Any, seed_block: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build the per-subset prompt_template_source provenance block."""
    seed_source: Dict[str, Any] = {}
    if seed_block and isinstance(seed_block.get("prompt_template_source"), dict):
        seed_source = copy.deepcopy(seed_block["prompt_template_source"])

    origin = getattr(args, "template_source_origin", None)
    reference = getattr(args, "template_source_ref", None)
    notes = getattr(args, "template_source_notes", None)

    if origin is None and reference is None and notes is None and seed_source:
        return seed_source

    out = seed_source if seed_source else {
        "origin": "unspecified",
        "reference": "",
        "notes": "TODO: pass --template-source-origin/-ref (or seed prompt_template_source); "
                 "this stub is rejected by validate.py --audit",
    }
    if origin is not None:
        out["origin"] = origin
    if reference is not None:
        out["reference"] = reference
    if notes is not None:
        out["notes"] = notes
    out.setdefault("origin", "unspecified")
    out.setdefault("reference", "")
    out.setdefault("notes", "")
    return out


def build_metadata_object(
    args: Any,
    colmap: Dict[str, str],
    choice_specs: List[Dict[str, Any]],
    template_str: Optional[str],
    modalities: List[str],
    media_min: int,
    media_max: int,
    seed: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble full metadata.json dict.

    With ``--metadata-json``, the seed subset is deep-copied and only
    ``release_date``, ``modalities``, ``source.*`` (format/url),
    and ``media.min_items`` / ``max_items`` are refreshed; other mapping and
    subset fields stay unless the CLI explicitly overrides language, task_type,
    or ``--template``.
    """
    src_path = getattr(args, "json", None) or getattr(args, "tsv", None) or getattr(args, "csv", None)
    name = getattr(args, "dataset_name", None) or (args.hf or (Path(str(src_path)).stem if src_path else "dataset"))
    subset = getattr(args, "subset", "main")
    release = getattr(args, "release_date", None) or date.today().isoformat()
    inferred_tt = _inferred_task_type(colmap, choice_specs)

    if seed and isinstance(seed, dict):
        root = copy.deepcopy(seed)
        root.setdefault("name", name)
        prev_rd = str(root.get("release_date") or "")
        root["release_date"] = max(prev_rd, release) if prev_rd else release
        subs_all = root.setdefault("subsets", {})
        seed_block = copy.deepcopy(subs_all.get(subset) or {})
        if not seed_block:
            raise ValueError(f"seed metadata has no subset {subset!r}")

        mfs = copy.deepcopy(seed_block.get("mapping_from_source") or {})
        if not isinstance(mfs, dict):
            raise ValueError("seed mapping_from_source must be an object")

        src_seed = copy.deepcopy(mfs.get("source") or {})
        if not isinstance(src_seed, dict):
            src_seed = {}
        fmt = (
            getattr(args, "source_format", None)
            or src_seed.get("format")
            or infer_source_format(args)
        )
        src_seed["format"] = fmt
        # Build url dict (migrate legacy links/string-url if present).
        url_dict: Dict[str, str] = {}
        existing_url = src_seed.get("url")
        if isinstance(existing_url, dict):
            url_dict.update(existing_url)
        for k, v in (src_seed.get("links") or {}).items():
            url_dict.setdefault(k, v)
        split_key = args.split or "train"
        if args.hf:
            url_dict[split_key] = (
                getattr(args, "source_url", None)
                or (isinstance(existing_url, str) and existing_url)
                or f"https://huggingface.co/datasets/{args.hf}"
            )
        elif getattr(args, "source_url", None):
            url_dict[split_key] = args.source_url
        # Remove deprecated fields.
        for deprecated in ("links", "repo", "path"):
            src_seed.pop(deprecated, None)
        src_seed["url"] = url_dict
        mfs["source"] = src_seed

        if mfs.get("media") and isinstance(mfs["media"], dict):
            mfs["media"] = dict(mfs["media"])
            mfs["media"]["min_items"] = media_min
            mfs["media"]["max_items"] = media_max
        else:
            partial = build_mapping_from_colmap(colmap, choice_specs, media_min, media_max, args)
            if "media" in partial:
                mfs["media"] = partial["media"]

        # Pass-through --map keys (e.g. category=cat) merge into extra without clobbering seed extras.
        _reserved = SCHEMA_KEYS | {
            "id", "question", "answer", "hint", "options", "choices", "media", "source", "extra",
        }
        ex_map = copy.deepcopy(mfs.get("extra") or {})
        if not isinstance(ex_map, dict):
            ex_map = {}
        for k, src in colmap.items():
            if k in _reserved:
                continue
            ex_map[k] = {"from": src}
        if ex_map:
            mfs["extra"] = ex_map

        if args.language is not None:
            languages = list(args.language)
        else:
            languages = list(seed_block.get("language") or ["en"])

        if getattr(args, "task_type", None) is not None:
            task_type = args.task_type
        else:
            task_type = str(seed_block.get("task_type") or inferred_tt)

        if getattr(args, "template", None) is not None:
            prompt_t = template_str or ""
        else:
            prompt_t = str(seed_block.get("prompt_template") or (template_str or ""))

        merged_block = {
            "language": languages,
            "modalities": modalities,
            "task_type": task_type,
            "prompt_template": prompt_t,
            "prompt_template_source": _prompt_template_source_from_args(args, seed_block),
            "mapping_from_source": mfs,
        }
        for k, v in seed_block.items():
            if k not in merged_block:
                merged_block[k] = copy.deepcopy(v)
        subs_all[subset] = merged_block
        return root

    fmt = getattr(args, "source_format", None) or infer_source_format(args)
    source: Dict[str, Any] = {"format": fmt, "url": {}}
    split_key = args.split or "train"
    if args.hf:
        source["url"][split_key] = (
            getattr(args, "source_url", None)
            or f"https://huggingface.co/datasets/{args.hf}"
        )
    else:
        if getattr(args, "source_url", None):
            source["url"][split_key] = args.source_url

    mapping_body = build_mapping_from_colmap(colmap, choice_specs, media_min, media_max, args)
    mapping_body["source"] = source

    languages = list(args.language) if args.language is not None else ["en"]
    task_type = args.task_type if getattr(args, "task_type", None) is not None else inferred_tt

    subset_obj = {
        "language": languages,
        "modalities": modalities,
        "task_type": task_type,
        "prompt_template": template_str or "",
        "prompt_template_source": _prompt_template_source_from_args(args),
        "mapping_from_source": mapping_body,
    }

    return {
        "name": name,
        "release_date": release,
        "subsets": {subset: subset_obj},
    }


def get_media_field(row: Dict[str, Any], colmap: Dict[str, str]) -> List[Any]:
    for key in ("images", "media", "image"):
        if key in colmap:
            raw = row.get(colmap[key])
            if raw is None:
                return []
            return list(raw) if isinstance(raw, list) else [raw]
    return []


def _check_disk_space(path: str, warn_gb: float = 5.0) -> None:
    """Warn if the target partition has less than warn_gb GB free."""
    try:
        free_gb = shutil.disk_usage(path).free / 1e9
        if free_gb < warn_gb:
            print(
                f"WARNING: only {free_gb:.1f} GB free on the partition containing {path!r}.\n"
                f"  Tips to save space:\n"
                f"    • Use --image-format jpeg (5-10x smaller than PNG at quality 92)\n"
                f"    • Process one split at a time; delete per-split dirs after merging\n"
                f"    • Run: python3 cleanup.py --hf-cache <repo_id>  to free parquet cache\n"
                f"    • Use --work-dir <path-on-larger-partition> to redirect all cache/temp paths"
            )
    except Exception:
        pass


def _setup_work_dir(work_dir: Path) -> None:
    """Create per-dataset work dir and redirect all temp/cache env vars into it.

    Sets HF_HOME, HF_DATASETS_CACHE, HUGGINGFACE_HUB_CACHE, TRANSFORMERS_CACHE,
    TMPDIR, TEMP, and TMP to subdirectories under ``work_dir``.  Must be called
    before any datasets/huggingface_hub imports so the libraries pick up the new
    paths.
    """
    import tempfile as _tempfile
    hf_home = work_dir / "hf"
    tmp_dir = work_dir / "tmp"
    hf_home.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env_update = {
        "HF_HOME": str(hf_home),
        "HF_DATASETS_CACHE": str(hf_home / "datasets"),
        "HUGGINGFACE_HUB_CACHE": str(hf_home / "hub"),
        "TRANSFORMERS_CACHE": str(hf_home),
    }
    # Linux Unix-domain socket paths are capped at 108 bytes. multiprocess.Manager
    # (used internally by datasets.save_to_disk → iflatmap_unordered) builds a
    # socket path under $TMPDIR; if the project root + work-dir name already
    # consumes >~80 chars, the socket bind fails with `OSError: AF_UNIX path too
    # long`, killing the save. Only route TMPDIR into the work-dir when the
    # resulting socket path will fit; otherwise leave TMPDIR pointing at the
    # system default (usually /tmp) so multiprocess works. HF caches still go
    # to work_dir regardless — they don't use Unix sockets.
    # multiprocess Manager builds a socket path like `<TMPDIR>/pymp-<uuid>/listener-<uuid>`,
    # which appends ~55 chars. The kernel limit for AF_UNIX paths is 108 bytes, so
    # leave a generous margin: ~50 chars max for TMPDIR. When tmp_dir exceeds that,
    # we leave TMPDIR pointing at the system default (which is typically /tmp).
    AF_UNIX_BUDGET = 50
    if len(str(tmp_dir)) <= AF_UNIX_BUDGET:
        env_update.update({
            "TMPDIR": str(tmp_dir),
            "TEMP": str(tmp_dir),
            "TMP": str(tmp_dir),
        })
        _tempfile.tempdir = str(tmp_dir)
    else:
        print(f"[work-dir] TMPDIR path too long ({len(str(tmp_dir))} chars); "
              f"leaving TMPDIR={os.environ.get('TMPDIR', '/tmp')} to avoid AF_UNIX bind failure")
    os.environ.update(env_update)
    try:
        stat = shutil.disk_usage(work_dir)
        free_gb = stat.free / 1e9
        total_gb = stat.total / 1e9
        print(f"[work-dir] {work_dir}  ({free_gb:.1f}/{total_gb:.1f} GB free/total)")
        if free_gb < 5.0:
            print(f"WARNING: only {free_gb:.1f} GB free on partition containing work-dir.")
    except Exception:
        print(f"[work-dir] {work_dir}")


def iter_source(args) -> Iterator[Tuple[Dict[str, Any], int]]:
    """Iterate (row, global_index). When `--explode <field>` is set, each
    element of that list-valued field becomes its own row with its dict keys
    merged onto the outer row, and `_unique_id` set from the outer id.
    Outer id None → `_unique_id = None` so process_one skips with reason."""
    if args.hf:
        from datasets import load_dataset
        # Redirect HF dataset cache when --hf-cache-dir is set.  Must be set
        # before load_dataset is called so the datasets library picks it up.
        hf_cache = getattr(args, "hf_cache_dir", None)
        if hf_cache:
            os.environ["HF_DATASETS_CACHE"] = hf_cache
        # Stream when the user only wants a small sample; avoids fully
        # downloading multi-GB benchmarks just to inspect a few rows.
        streaming = args.limit is not None
        # Optional --hf-config <name> for multi-config repos (e.g. lmms-lab/DocVQA
        # ships configs "DocVQA" + "InfographicVQA"). When omitted, load_dataset
        # uses the repo's default config — works for single-config repos.
        hf_config = getattr(args, "hf_config", None)
        if hf_config:
            base = load_dataset(args.hf, hf_config, split=args.split, streaming=streaming)
        else:
            base = load_dataset(args.hf, split=args.split, streaming=streaming)
        raw_iter = enumerate(base)
    elif args.json:
        with open(args.json, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("--json top-level must be a JSON array")
        raw_iter = enumerate(data)
    elif getattr(args, "tsv", None):
        with open(args.tsv, encoding="utf-8", newline="") as f:
            data = list(csv.DictReader(f, delimiter="\t"))
        raw_iter = enumerate(data)
    elif getattr(args, "csv", None):
        with open(args.csv, encoding="utf-8", newline="") as f:
            data = list(csv.DictReader(f))
        raw_iter = enumerate(data)
    else:
        raise ValueError("no source (internal)")

    # Optional row filter — a Python expression evaluated with `row` in scope.
    # Use sparingly; mixed-modality benchmarks (e.g. SEEDBench with image+video
    # rows) need a filter to isolate the modality this converter supports.
    filter_expr = getattr(args, "filter_expr", None)
    filter_fn = None
    if filter_expr:
        compiled = compile(filter_expr, "<--filter-expr>", "eval")
        filter_fn = lambda row: bool(eval(compiled, {"__builtins__": {}}, {"row": row}))

    explode = getattr(args, "explode", None)
    id_field = getattr(args, "_original_id_field", None)

    global_idx = 0
    for _, row in raw_iter:
        if filter_fn is not None and not filter_fn(row):
            continue
        if not explode:
            yield row, global_idx
            global_idx += 1
            continue
        sub_rows = row.get(explode) or []
        outer_id = row.get(id_field) if id_field else None
        for j, sub in enumerate(sub_rows):
            merged = dict(row)
            if isinstance(sub, dict):
                merged.update(sub)
            else:
                merged["_explode_value"] = sub
            merged["_explode_idx"] = j
            merged["_outer_id"] = outer_id
            merged["_unique_id"] = (
                args.id_template.format(id=outer_id, idx=j, outer=outer_id)
                if outer_id is not None else None
            )
            yield merged, global_idx
            global_idx += 1


# --- conversion state -------------------------------------------------------

@dataclass
class StreamState:
    local_rows: List[Dict[str, Any]] = field(default_factory=list)
    hf_ids: List[str] = field(default_factory=list)
    hf_media: List[List[Dict[str, Any]]] = field(default_factory=list)
    hf_messages: List[str] = field(default_factory=list)
    # Parallel to local_rows / hf_ids: the source-stream index that produced
    # each surviving row.  Used to make duplicate-id disambiguation
    # deterministic (occurrence order = source order, independent of which
    # thread completed first).
    local_source_idx: List[int] = field(default_factory=list)
    hf_source_idx: List[int] = field(default_factory=list)
    skipped_samples: List[Dict[str, Any]] = field(default_factory=list)
    skip_reason_counts: Dict[str, int] = field(default_factory=dict)
    seen: int = 0
    row_signals: List[Dict[str, Any]] = field(default_factory=list)
    media_counts_kept: List[int] = field(default_factory=list)
    duplicate_ids_resolved: Dict[str, Any] = field(default_factory=dict)

    def record(
        self,
        idx: int,
        stem: Optional[str],
        local: Optional[Dict[str, Any]],
        hf: Optional[Tuple[str, List[Dict[str, Any]], str]],
        reason: Optional[str],
        signal: Optional[Dict[str, Any]],
        n_media_kept: Optional[int],
    ) -> None:
        self.seen += 1
        if signal is not None:
            self.row_signals.append(signal)
        if reason is not None:
            if len(self.skipped_samples) < 50:
                self.skipped_samples.append({"index": idx, "id": stem, "reason": reason})
            cat = _skip_category(reason)
            self.skip_reason_counts[cat] = self.skip_reason_counts.get(cat, 0) + 1
            return
        if n_media_kept is not None:
            self.media_counts_kept.append(n_media_kept)
        if local is not None:
            self.local_rows.append(local)
            self.local_source_idx.append(idx)
        if hf is not None:
            self.hf_ids.append(hf[0])
            self.hf_media.append(hf[1])
            self.hf_messages.append(hf[2])
            self.hf_source_idx.append(idx)

    @property
    def skipped(self) -> int:
        return sum(self.skip_reason_counts.values())


def _resolve_duplicate_row_ids(state: "ConvertState") -> None:
    """Disambiguate non-unique row ids deterministically.

    Some upstreams use a per-entity id (e.g., per-chart UUID) that is not
    unique once multiple samples per entity ship in the same split (multiple
    QA paraphrases per chart, multiple sub-questions per document, etc.).
    Walk surviving rows in stable source-stream order, count per-id
    occurrences (1-indexed), and rewrite `id` to `{source_id}_q{k}` for the
    kth occurrence.  Move the original id into a `source_id` field on the
    per-row message dict and on the local row dict so consumers can still
    join back to the upstream entity.

    If every id is already row-unique, this is a no-op.
    """
    from collections import Counter
    # Tally — use whichever output mode has data (both lists carry the same
    # ids when --mode both).
    ids_seq: List[str] = state.hf_ids if state.hf_ids else [r["id"] for r in state.local_rows]
    counts = Counter(ids_seq)
    dup_ids = [i for i, c in counts.items() if c > 1]
    if not dup_ids:
        return

    # Build a stable per-(id, source_idx) occurrence index.  Sort by
    # (original_id, source_idx) so the kth occurrence of `<X>` is always the
    # one that came first in the source stream, regardless of thread order.
    def _occ_map(ids: List[str], src_idx: List[int]) -> List[int]:
        order = sorted(range(len(ids)), key=lambda i: (ids[i], src_idx[i] if i < len(src_idx) else 0))
        occ_seen: Dict[str, int] = {}
        out = [0] * len(ids)
        for i in order:
            occ_seen[ids[i]] = occ_seen.get(ids[i], 0) + 1
            out[i] = occ_seen[ids[i]]
        return out

    if state.hf_ids:
        occ = _occ_map(state.hf_ids, state.hf_source_idx)
        new_hf_ids: List[str] = []
        new_hf_messages: List[str] = []
        for i, old_id in enumerate(state.hf_ids):
            new_id = f"{old_id}_q{occ[i]}" if counts[old_id] > 1 else old_id
            new_hf_ids.append(new_id)
            # Inject source_id into the message dict only when the id changed.
            if new_id != old_id:
                msgs = json.loads(state.hf_messages[i])
                if msgs and isinstance(msgs[0], dict):
                    msgs[0]["source_id"] = old_id
                new_hf_messages.append(json.dumps(msgs, ensure_ascii=False))
            else:
                new_hf_messages.append(state.hf_messages[i])
        state.hf_ids = new_hf_ids
        state.hf_messages = new_hf_messages

    if state.local_rows:
        local_ids = [r["id"] for r in state.local_rows]
        occ = _occ_map(local_ids, state.local_source_idx)
        for i, row in enumerate(state.local_rows):
            old_id = row["id"]
            if counts[old_id] > 1:
                row["id"] = f"{old_id}_q{occ[i]}"
                # mmeval LocalJSONDataset doesn't re-render `messages[0]` at
                # eval time when a per-row `prompt` is present; still inject
                # source_id so downstream tooling can read it.
                msgs = row.get("messages") or []
                if msgs and isinstance(msgs[0], dict):
                    msgs[0]["source_id"] = old_id

    # Record what happened for the convert_summary.json so the operator sees it.
    state.duplicate_ids_resolved = {
        "source_id_field_added": "source_id",
        "id_template": "{source_id}_q{occurrence_index}",
        "n_duplicated_source_ids": len(dup_ids),
        "max_occurrences": max(counts.values()),
        "examples": [{"source_id": sid, "occurrences": counts[sid]} for sid in dup_ids[:5]],
    }
    print(f"[ids] disambiguated {len(dup_ids)} source ids with multiple occurrences "
          f"(max {max(counts.values())}); original id preserved as `source_id`.")


# --- main per-row pipeline --------------------------------------------------

def stream_convert(
    args: Any,
    template_str: Optional[str],
    colmap: Dict[str, str],
    choice_specs: List[Dict[str, Any]],
    metadata_seed: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _check_disk_space(str(out))
    do_local = args.mode in {"local", "both"}
    do_hf = args.mode in {"hf", "both"}

    media_dir: Optional[Path] = None
    if do_local:
        media_dir = out / "media"
        media_dir.mkdir(parents=True, exist_ok=True)

    jinja_template: Optional[Template] = None
    if template_str is not None:
        # Compile once; render is hot-path.
        jinja_template = build_jinja_env().from_string(template_str)

    # Extensionless video URLs/paths only enter the video pipeline when the
    # template contains `<video>` — avoids misclassifying image URLs without
    # extensions when the prompt is image-only.
    expects_video = bool(template_str and "<video>" in template_str)

    state = StreamState()
    t0 = time.time()
    last_log = t0

    def maybe_log() -> None:
        nonlocal last_log
        now = time.time()
        if now - last_log > 5:
            print(f"  …processed {state.seen} rows ({state.seen/(now-t0):.1f}/s)")
            last_log = now

    auto_id_tpl = getattr(args, "auto_id", None)
    split_name = args.split or "data"

    def process_one(row: Dict[str, Any], i: int):
        raw_id = row.get(colmap["id"])
        if raw_id is None:
            if auto_id_tpl:
                # Synthesize a deterministic positional id when the source has no id
                # field. Documented as `auto_id_template` in the metadata.json so
                # downstream consumers can re-derive the row identity.
                raw_id = auto_id_tpl.format(idx=i, split=split_name)
            else:
                return i, None, None, None, "missing_required:id", None, None
        stem = str(raw_id).replace("/", "_")

        msg = build_message(row, colmap, args.answer_join, args.answer_list, choice_specs or None)
        media_raw = get_media_field(row, colmap)

        if (args.json or getattr(args, "tsv", None) or getattr(args, "csv", None)) and args.media_dir:
            resolved = []
            for m in media_raw:
                if isinstance(m, str) and not os.path.isabs(m) and not m.startswith(("http://","https://")):
                    cand = os.path.join(args.media_dir, m)
                    if os.path.exists(cand):
                        m = cand
                resolved.append(m)
            media_raw = resolved

        signal = _row_media_profile(media_raw, expects_video)

        saved_basenames: List[str] = []
        hf_media_items: List[Dict[str, Any]] = []
        for m_idx, m in enumerate(media_raw):
            video = video_path_of(m)
            if video is None and expects_video:
                ps = _media_path_str(m)
                if ps and (ps.startswith("http") or os.path.exists(ps)):
                    ext_raw = os.path.splitext(ps.split("?")[0])[-1].lower()
                    if not ext_raw:
                        video = ps
            if video is not None:
                if do_hf and not getattr(args, "hf_video", False):
                    raise ValueError(
                        f"HF mode does not support video media (sample id={stem}); "
                        f"use --mode local for video benchmarks, or pass --hf-video "
                        f"to store video paths (not bytes) in the Arrow dataset."
                    )
                if do_hf and getattr(args, "hf_video", False):
                    hf_media_items.append(video)
                try:
                    verify_vid = getattr(args, "verify_video", False)
                    saved_basenames.append(materialize_video(video, media_dir, stem, m_idx,
                                                            verify=verify_vid))
                except EncodeFailed as e:
                    return i, stem, None, None, f"{e.category}:{e}", signal, None
                continue
            try:
                data, ext = encode_image_bytes(m, args.image_format, args.jpeg_quality)
            except EncodeFailed as e:
                return i, stem, None, None, f"{e.category}:{e}", signal, None
            name = f"{stem}_{m_idx}{ext}"
            if do_local:
                (media_dir / name).write_bytes(data)
                saved_basenames.append(name)
            if do_hf:
                hf_media_items.append({"path": None, "bytes": data})

        local_entry = None
        hf_entry = None
        if do_local:
            local_msg = dict(msg)
            if jinja_template is not None:
                prompt = jinja_template.render(**local_msg)
                if args.prompt_prefix and args.prompt_prefix not in prompt:
                    prompt = args.prompt_prefix + prompt
                local_msg["prompt"] = prompt
            local_entry = {"id": stem, "media": saved_basenames, "messages": [local_msg]}
        if do_hf:
            hf_entry = (stem, hf_media_items, json.dumps([msg], ensure_ascii=False))
        n_kept = len(media_raw)
        return i, stem, local_entry, hf_entry, None, signal, n_kept

    if args.workers and args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures: List = []
            queue_cap = args.workers * 4
            for row, i in iter_source(args):
                if args.limit is not None and i >= args.limit:
                    break
                futures.append(pool.submit(process_one, row, i))
                if len(futures) >= queue_cap:
                    state.record(*futures.pop(0).result())
                    maybe_log()
            for fut in futures:
                state.record(*fut.result())
    else:
        for row, i in iter_source(args):
            if args.limit is not None and i >= args.limit:
                break
            state.record(*process_one(row, i))
            maybe_log()

    # Globally disambiguate row ids before writing artifacts.  Some source
    # datasets carry a per-CHART or per-DOCUMENT id that is not unique at the
    # row granularity once the dataset ships multiple samples per source
    # entity (e.g., ChartNet has multiple QA paraphrases per chart_id).  In
    # those cases we walk surviving rows in stable source-stream order, count
    # per-id occurrences (1-indexed), and rewrite the row id to
    # `{source_id}_q{k}`.  The original id is preserved as `source_id` on
    # the per-row message dict so downstream consumers can still join back
    # to the upstream source.  Opt out with --no-auto-disambiguate-ids when
    # the upstream id is already row-unique and you want the converter to
    # hard-fail on collisions instead.
    if getattr(args, "auto_disambiguate_ids", True):
        _resolve_duplicate_row_ids(state)

    summary: Dict[str, Any] = {
        "mode": args.mode,
        "out": str(out),
        "rows": state.seen - state.skipped,
        "skipped": state.skipped,
        "skip_reason_counts": state.skip_reason_counts,
        "skipped_samples": state.skipped_samples,
        "duplicate_ids_resolved": state.duplicate_ids_resolved,
        "elapsed_s": round(time.time() - t0, 2),
    }

    if do_local:
        state.local_rows.sort(key=lambda r: r["id"])
        data_path = out / "data.json"
        dump_kw: Dict[str, Any] = {"ensure_ascii": False, "indent": args.json_indent}
        if args.json_indent is None:
            dump_kw["separators"] = (",", ":")
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(state.local_rows, f, **dump_kw)
        summary["local"] = {"data_path": str(data_path), "media_dir": str(media_dir),
                            "rows": len(state.local_rows)}
        print(f"[local] wrote {len(state.local_rows)} rows -> {data_path}")

    if do_hf:
        if not state.hf_ids:
            print("[hf] SKIP: no rows survived; nothing to save under hf_dataset/. "
                  "Check convert_summary.json's skip_reason_counts.")
        else:
            from datasets import Dataset, DatasetDict, Features, Sequence, Value, Image as HFImage
            order = sorted(range(len(state.hf_ids)), key=lambda i: state.hf_ids[i])
            hf_ids = [state.hf_ids[i] for i in order]
            hf_media = [state.hf_media[i] for i in order]
            hf_messages = [state.hf_messages[i] for i in order]

            hf_video_mode = getattr(args, "hf_video", False)
            if hf_video_mode:
                features = Features({
                    "id": Value("string"),
                    "media": Sequence(Value("string")),
                    "messages": Value("string"),
                })
                print("[hf] video mode: media column stores video paths as strings")
            else:
                features = Features({
                    "id": Value("string"),
                    "media": Sequence(HFImage()),
                    "messages": Value("string"),
                })
            ds = Dataset.from_dict({"id": hf_ids, "media": hf_media, "messages": hf_messages},
                                   features=features)
            hf_path = out / "hf_dataset"
            if args.split:
                DatasetDict({args.split: ds}).save_to_disk(str(hf_path), num_proc=args.save_num_proc)
            else:
                ds.save_to_disk(str(hf_path), num_proc=args.save_num_proc)
            summary["hf"] = {"hf_dataset": str(hf_path), "rows": len(hf_ids),
                             "video_mode": hf_video_mode}
            print(f"[hf] wrote {len(hf_ids)} rows -> {hf_path}")

    seed_modalities: Optional[List[str]] = None
    if metadata_seed and isinstance(metadata_seed, dict):
        sub = (metadata_seed.get("subsets") or {}).get(getattr(args, "subset", "main"))
        if isinstance(sub, dict):
            sm = sub.get("modalities")
            if isinstance(sm, list) and sm:
                seed_modalities = list(sm)
    cli_modalities = list(getattr(args, "modalities", None) or [])
    if cli_modalities:
        modalities = cli_modalities
    elif seed_modalities:
        modalities = seed_modalities
    else:
        modalities = infer_modalities(state.row_signals)
    if state.media_counts_kept:
        media_min = min(state.media_counts_kept)
        media_max = max(state.media_counts_kept)
    else:
        media_min = media_max = 0

    meta_obj = build_metadata_object(
        args, colmap, choice_specs, template_str, modalities, media_min, media_max, metadata_seed,
    )
    meta_path = out / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_obj, f, ensure_ascii=False, indent=2)
    summary["metadata"] = str(meta_path)
    print(f"[meta] wrote {meta_path}")

    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf", help="HuggingFace dataset repo id")
    src.add_argument("--json", help="Local JSON file path (list of records)")
    src.add_argument("--tsv", help="Local TSV path (header row + tab-separated values)")
    src.add_argument("--csv", help="Local CSV path (header row)")
    p.add_argument("--split", default=None, help="HF split (required for --hf). Optional for local files.")
    p.add_argument("--hf-config", default=None,
                   help="HF config name for multi-config repos (e.g. lmms-lab/DocVQA "
                        "configs DocVQA / InfographicVQA). Omit for single-config repos.")
    p.add_argument("--media-dir", help="Local media dir for local file sources")
    p.add_argument("--map", nargs="*", default=None,
                   help="Column mapping: canonical=source pairs (required unless --metadata-json)")
    p.add_argument("--metadata-json", help="Authoring template; can supply mapping_from_source + prompt_template")
    p.add_argument("--name", dest="dataset_name", help="Dataset display name in metadata.json (default: inferred)")
    p.add_argument("--subset", default="main", help="Subset key inside metadata.json (default: main)")
    p.add_argument("--language", nargs="+", default=None,
                   help="ISO-ish language tags for metadata.json (default: seed or en)")
    p.add_argument("--task-type", default=None, help="Task type string in metadata.json (default: inferred)")
    p.add_argument("--modalities", nargs="*", default=None,
                   help="Override modalities list; default is inferred from data")
    p.add_argument("--release-date", default=None, help="metadata.json release_date (default: today UTC date)")
    p.add_argument("--source-url", default=None, help="Original dataset URL for metadata.json source.url / links")
    p.add_argument("--source-format", default=None,
                   choices=("huggingface", "tsv", "csv", "json"),
                   help="Force metadata.json mapping_from_source.source.format")
    p.add_argument("--template-source-origin", default=None,
                   choices=("official", "source_column", "fallback"),
                   help="prompt_template_source.origin provenance value")
    p.add_argument("--template-source-ref", default=None,
                   help="prompt_template_source.reference (URL/path+lines, source column, or fallback T-id)")
    p.add_argument("--template-source-notes", default=None,
                   help="Optional prompt_template_source.notes text")
    p.add_argument("--template", help="Jinja template string or file path")
    p.add_argument("--prompt-prefix", default="",
                   help="If template lacks an <image>/<video> placeholder, optionally prepend this")
    p.add_argument("--answer-join", default=" ",
                   help="Join string when source answer is a list (default: ' ')")
    p.add_argument("--answer-list", action="store_true",
                   help="Keep answer as a list (do not join)")
    p.add_argument("--mode", choices=["local", "hf", "both"], required=True,
                   help="Output mode")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of rows (for testing). With --hf, triggers streaming load.")
    p.add_argument("--workers", type=int, default=8,
                   help="Number of worker threads for image encoding (default 8)")
    p.add_argument("--image-format", choices=["png", "jpeg"], default="jpeg",
                   help="Image encoding for --mode local and HF Arrow storage. "
                        "jpeg (default) is 5-10x smaller than png at quality 92 and is "
                        "appropriate for almost all eval benchmarks. Use png only when "
                        "lossless fidelity is required (e.g. OCR benchmarks with fine text).")
    p.add_argument("--jpeg-quality", type=int, default=92, help="JPEG quality 1-100")
    p.add_argument("--hf-cache-dir", default=None,
                   help="Override only the HuggingFace datasets download cache directory. "
                        "Prefer --work-dir for a full redirect of all cache/temp paths.")
    p.add_argument("--work-dir", default=None,
                   help="Per-dataset work directory for HF caches and temp files. "
                        "Sets HF_HOME, HF_DATASETS_CACHE, HUGGINGFACE_HUB_CACHE, "
                        "TRANSFORMERS_CACHE, TMPDIR/TEMP/TMP to subdirs here. "
                        "Default: <project-root>/.tmp/conversions/<out-basename>. "
                        "Pass an explicit path to override, or --work-dir '' to disable.")
    p.add_argument("--explode", default=None,
                   help="Source list-valued field to explode into one row per element "
                        "(e.g. 'questions' for nested-MCQ datasets). The inner dict's "
                        "keys are merged onto each row; outer keys remain as fallback.")
    p.add_argument("--auto-id", default=None,
                   help="Format string used to synthesize a positional row id "
                        "when the source has no id field (e.g. '{split}_{idx:06d}'). "
                        "Available placeholders: {idx} (0-based row index), {split} "
                        "(--split value or 'data'). Without this flag, rows without "
                        "an id are skipped as 'missing_required:id' per the strict "
                        "no-fabrication contract; pass it only for sources you have "
                        "verified do not expose any natural id field.")
    p.add_argument("--id-template", default="{id}_q{idx}",
                   help="Format string for post-explode ids (default '{id}_q{idx}'). "
                        "Available fields: {id} (outer), {outer}, {idx}.")
    p.add_argument("--filter-expr", default=None,
                   help="Python expression eval'd per row (with `row` in scope) to "
                        "keep only matching rows. e.g. \"row['data_type']=='image'\" "
                        "for SEEDBench's image/video mix. Sandboxed (no builtins).")
    p.add_argument("--no-auto-disambiguate-ids", dest="auto_disambiguate_ids",
                   action="store_false",
                   help="By default, when the per-row id from the source is not unique "
                        "across surviving rows (e.g., a per-chart id when the dataset "
                        "ships multiple QA samples per chart), the converter rewrites "
                        "each duplicate id to '{source_id}_q{k}' (1-indexed by source "
                        "stream order) and preserves the original id in a `source_id` "
                        "field on the per-row message dict. Pass --no-auto-disambiguate-ids "
                        "to disable that and fail hard on id collisions instead.")
    p.set_defaults(auto_disambiguate_ids=True)
    p.add_argument("--hf-video", action="store_true",
                   help="Enable HF video mode: store video paths as strings in the "
                        "Arrow media column (Sequence(Value('string'))) instead of "
                        "raising an error. Video files are materialized to media/ "
                        "and must be uploaded separately to the HF repo. Use this "
                        "when converting video benchmarks for HF distribution.")
    p.add_argument("--verify-video", action="store_true",
                   help="Verify each materialized video file: check non-empty size "
                        "and (when PyAV is installed) probe codec, duration, and "
                        "frame count. Slower but catches broken/truncated downloads.")
    p.add_argument("--save-num-proc", type=int, default=1,
                   help="num_proc forwarded to Dataset.save_to_disk / DatasetDict.save_to_disk (default 1)")
    p.add_argument("--json-indent", type=int, default=None,
                   help="Indent for data.json (default: compact, no whitespace). "
                        "Use --json-indent 2 for human-readable output.")
    args = p.parse_args()

    # Set up per-dataset work dir (temp files + HF cache) before any dataset imports.
    if args.work_dir != "":  # empty string explicitly disables
        if args.work_dir:
            _wd = Path(args.work_dir)
        else:
            _project_root = Path(__file__).resolve().parents[3]
            _wd = _project_root / ".tmp" / "conversions" / Path(args.out).name
        _setup_work_dir(_wd)

    if args.hf and not args.split:
        p.error("--hf requires an explicit --split")

    metadata_seed: Optional[Dict[str, Any]] = None
    choice_specs: List[Dict[str, Any]] = []
    if args.metadata_json:
        metadata_seed = load_metadata_json(Path(args.metadata_json))
        resolved_subset, block = _subset_block(metadata_seed, args.subset)
        args.subset = resolved_subset
        mfs = block.get("mapping_from_source")
        if not isinstance(mfs, dict):
            raise ValueError("metadata subset must contain mapping_from_source object")
        base_map, choice_specs = colmap_from_mapping(mfs)
        if args.map is not None:
            base_map = merge_colmap(
                base_map,
                parse_map_pairs(args.map, require_id=False, require_question=False),
            )
        colmap = base_map
        template_str = load_template(args.template)
        if not template_str and block.get("prompt_template"):
            template_str = str(block["prompt_template"])
    else:
        if not args.map:
            p.error("either --map or --metadata-json is required")
        colmap = parse_map(
            args.map,
            task_type=getattr(args, "task_type", None),
            has_auto_id=bool(getattr(args, "auto_id", None)),
        )
        template_str = load_template(args.template)

    args._original_id_field = colmap.get("id")
    if args.explode:
        colmap = dict(colmap)
        colmap["id"] = "_unique_id"

    if args.release_date is None:
        args.release_date = date.today().isoformat()

    src = args.hf or args.json or getattr(args, "tsv", None) or getattr(args, "csv", None)
    print(f"Source: {src} (split={args.split!r})")
    print(f"Column map: {colmap}")
    print(f"Template: {template_str!r}")

    summary = stream_convert(args, template_str, colmap, choice_specs, metadata_seed)
    summary_path = Path(args.out) / "convert_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nDone in {summary['elapsed_s']}s — kept {summary['rows']} rows, "
          f"skipped {summary['skipped']} (reasons: {summary['skip_reason_counts']}).")
    print(f"Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
