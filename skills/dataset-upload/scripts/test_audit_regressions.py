#!/usr/bin/env python3
"""Regression checks for audit plan (metadata-json seed, explode, merge, modalities).

Run from anywhere:
  python3 scripts/test_audit_regressions.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

MODALITY_ORDER = (
    "single_image_start", "single_video_start",
    "multi_image_start", "multi_image_interleave",
    "multi_video_interleave", "multi_image_video_interleave", "text",
)


def _sort_modalities_check(mods: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for tag in MODALITY_ORDER:
        if tag in mods and tag not in seen:
            out.append(tag)
            seen.add(tag)
    for tag in mods:
        if tag not in seen:
            out.append(tag)
            seen.add(tag)
    return out


def _run_convert(cwd: Path, args: list[str]) -> None:
    cmd = [sys.executable, str(cwd / "scripts" / "convert.py"), *args]
    r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if r.returncode != 0:
        raise AssertionError(f"convert failed: {cmd}\n{r.stdout}\n{r.stderr}")


def _run_merge(cwd: Path, inputs: list[Path], out: Path) -> None:
    cmd = [
        sys.executable,
        str(cwd / "scripts" / "merge_splits.py"),
        "--inputs",
        *[str(p) for p in inputs],
        "--out",
        str(out),
    ]
    r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if r.returncode != 0:
        raise AssertionError(f"merge_splits failed: {cmd}\n{r.stdout}\n{r.stderr}")


def test_seed_preserve_subset_resolve_and_links(tmp_skill: Path, tmp: Path) -> None:
    media = tmp / "media"
    media.mkdir()
    for name in ("a.png", "b.png"):
        Image.new("RGB", (1, 1), color="white").save(media / name)

    data = [
        {
            "qid": "1",
            "text": "hello",
            "img": [str(media / "a.png"), str(media / "b.png")],
            "opt_a": "A1",
            "note_col": "n1",
        }
    ]
    data_path = tmp / "rows.json"
    data_path.write_text(json.dumps(data), encoding="utf-8")

    seed = {
        "name": "SeedTest",
        "release_date": "2020-01-01",
        "subsets": {
            "bench_only": {
                "language": ["zh"],
                "modalities": ["single_image_start"],
                "task_type": "vqa",
                "prompt_template": "<image>{{ question }}",
                "prompt_template_source": {
                    "origin": "source_column",
                    "reference": "text",
                    "notes": "Seed provenance should be preserved.",
                },
                "mapping_from_source": {
                    "source": {"format": "json", "url": {}},
                    "id": {"from": "qid"},
                    "question": {"from": "text"},
                    "media": {"from": "img", "type": "list", "min_items": 9, "max_items": 99},
                    "choices": [{"key": "A", "from": "opt_a", "optional": False}],
                    "extra": {"note": {"from": "note_col", "optional": True}},
                },
            }
        },
    }
    seed_path = tmp / "seed.json"
    seed_path.write_text(json.dumps(seed, indent=2), encoding="utf-8")

    out = tmp / "out_seed"
    _run_convert(
        tmp_skill,
        [
            "--json",
            str(data_path),
            "--metadata-json",
            str(seed_path),
            "--media-dir",
            str(media),
            "--split",
            "dev",
            "--source-url",
            "https://example.com/dev",
            "--mode",
            "hf",
            "--out",
            str(out),
            "--workers",
            "1",
        ],
    )

    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert "main" not in meta.get("subsets", {}), "spurious main subset"
    blk = meta["subsets"]["bench_only"]
    assert blk["language"] == ["zh"]
    assert blk["task_type"] == "vqa"
    assert blk["prompt_template_source"]["origin"] == "source_column"
    assert blk["prompt_template_source"]["reference"] == "text"
    ch = blk["mapping_from_source"]["choices"]
    assert isinstance(ch, list) and ch[0].get("key") == "A" and ch[0].get("from") == "opt_a"
    assert ch[0].get("optional") is False
    assert "extra" in blk["mapping_from_source"]
    media_map = blk["mapping_from_source"]["media"]
    assert media_map["min_items"] == 2 and media_map["max_items"] == 2
    url = blk["mapping_from_source"]["source"]["url"]
    assert isinstance(url, dict) and url.get("dev") == "https://example.com/dev"


def test_map_override_only_with_metadata_json(tmp_skill: Path, tmp: Path) -> None:
    media = tmp / "m2"
    media.mkdir()
    Image.new("RGB", (1, 1), color="white").save(media / "x.png")
    data_path = tmp / "r2.json"
    data_path.write_text(
        json.dumps([{"qid": "1", "text": "q", "img": str(media / "x.png"), "cat": "z"}]),
        encoding="utf-8",
    )
    seed_path = tmp / "seed2.json"
    seed_path.write_text(
        json.dumps(
            {
                "name": "M2",
                "release_date": "2020-01-01",
                "subsets": {
                    "main": {
                        "language": ["en"],
                        "modalities": ["single_image_start", "text"],
                        "task_type": "vqa",
                        "prompt_template": "<image>{{ question }}",
                        "mapping_from_source": {
                            "source": {"format": "json", "url": {}},
                            "id": {"from": "qid"},
                            "question": {"from": "text"},
                            "media": {"from": "img", "type": "list", "min_items": 1, "max_items": 1},
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp / "out_map"
    _run_convert(
        tmp_skill,
        [
            "--json",
            str(data_path),
            "--metadata-json",
            str(seed_path),
            "--media-dir",
            str(media),
            "--map",
            "category=cat",
            "--mode",
            "hf",
            "--out",
            str(out),
            "--workers",
            "1",
        ],
    )
    blk = json.loads((out / "metadata.json").read_text(encoding="utf-8"))["subsets"]["main"]
    extra = blk["mapping_from_source"].get("extra") or {}
    assert "category" in extra
    assert extra["category"]["from"] == "cat"


def test_explode_id_metadata(tmp_skill: Path, tmp: Path) -> None:
    data_path = tmp / "ex.json"
    data_path.write_text(
        json.dumps(
            [{"outer_id": "10", "subs": [{"text": "inner q", "answer": "x"}]}]
        ),
        encoding="utf-8",
    )
    out = tmp / "out_ex"
    _run_convert(
        tmp_skill,
        [
            "--json",
            str(data_path),
            "--map",
            "id=outer_id",
            "question=text",
            "answer=answer",
            "--explode",
            "subs",
            "--mode",
            "local",
            "--out",
            str(out),
            "--workers",
            "1",
        ],
    )
    blk = json.loads((out / "metadata.json").read_text(encoding="utf-8"))["subsets"]["main"]
    idm = blk["mapping_from_source"]["id"]
    assert idm["from"] == "outer_id"
    assert idm.get("explode_from") == "subs"
    assert "id_template" in idm


def test_multi_image_text_modalities(tmp_skill: Path, tmp: Path) -> None:
    media = tmp / "m3"
    media.mkdir()
    Image.new("RGB", (1, 1), color="white").save(media / "p1.png")
    Image.new("RGB", (1, 1), color="white").save(media / "p2.png")
    data_path = tmp / "r3.json"
    data_path.write_text(
        json.dumps(
            [
                {"id": "1", "q": "a", "img": [str(media / "p1.png"), str(media / "p2.png")]},
                {"id": "2", "q": "textonly", "img": []},
            ]
        ),
        encoding="utf-8",
    )
    out = tmp / "out_mod"
    _run_convert(
        tmp_skill,
        [
            "--json",
            str(data_path),
            "--map",
            "id=id",
            "question=q",
            "image=img",
            "--mode",
            "hf",
            "--out",
            str(out),
            "--template-source-origin",
            "fallback",
            "--template-source-ref",
            "T4",
            "--template-source-notes",
            "regression test fallback provenance",
            "--workers",
            "1",
        ],
    )
    blk = json.loads((out / "metadata.json").read_text(encoding="utf-8"))["subsets"]["main"]
    mods = blk["modalities"]
    assert mods == _sort_modalities_check(mods)
    assert mods == ["multi_image_start", "text"]
    pts = blk["prompt_template_source"]
    assert pts["origin"] == "fallback"
    assert pts["reference"] == "T4"
    assert pts["notes"] == "regression test fallback provenance"


def test_merge_auto_subset_non_main(tmp_skill: Path, tmp: Path) -> None:
    media = tmp / "m4"
    media.mkdir()
    Image.new("RGB", (1, 1), color="white").save(media / "z.png")
    # Reuse the same path between runs so metadata source.path / media stats match merge's equality check.
    shared_data = tmp / "src" / "data.json"
    shared_data.parent.mkdir(parents=True, exist_ok=True)

    def one_convert(name: str, rid: str, split: str) -> Path:
        shared_data.write_text(
            json.dumps([{"id": rid, "question": "q", "image": str(media / "z.png")}]),
            encoding="utf-8",
        )
        seed = {
            "name": "Z",
            "release_date": "2020-01-01",
            "subsets": {
                "special": {
                    "language": ["en"],
                    "modalities": ["single_image_start", "text"],
                    "task_type": "vqa",
                    "prompt_template": "<image>{{ question }}",
                    "mapping_from_source": {
                        "source": {"format": "json", "url": {}},
                        "id": {"from": "id"},
                        "question": {"from": "question"},
                        "media": {"from": "image", "type": "list", "min_items": 1, "max_items": 1},
                    },
                }
            },
        }
        sp = tmp / f"seed_{name}.json"
        sp.write_text(json.dumps(seed), encoding="utf-8")
        outp = tmp / f"out_{name}"
        _run_convert(
            tmp_skill,
            [
                "--json",
                str(shared_data),
                "--metadata-json",
                str(sp),
                "--media-dir",
                str(media),
                "--split",
                split,
                "--mode",
                "hf",
                "--out",
                str(outp),
                "--workers",
                "1",
            ],
        )
        return outp

    a = one_convert("a", "1", "train")
    b = one_convert("b", "2", "val")
    merged = tmp / "merged"
    _run_merge(tmp_skill, [a, b], merged)
    meta = json.loads((merged / "metadata.json").read_text(encoding="utf-8"))
    assert set(meta["subsets"].keys()) == {"special"}
    mods = meta["subsets"]["special"]["modalities"]
    assert mods == _sort_modalities_check(mods)


def main() -> int:
    skill = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_seed_preserve_subset_resolve_and_links(skill, tmp)
        test_map_override_only_with_metadata_json(skill, tmp)
        test_explode_id_metadata(skill, tmp)
        test_multi_image_text_modalities(skill, tmp)
        # merge uses separate sub-temps to avoid name collisions
        with tempfile.TemporaryDirectory() as td2:
            test_merge_auto_subset_non_main(skill, Path(td2))
    print("audit regressions: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
