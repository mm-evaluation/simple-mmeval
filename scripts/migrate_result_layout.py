#!/usr/bin/env python3
"""Migrate old-layout result.json files to the canonical FLAT layout.

Canonical layout (the one contract the scorer accepts): grading fields —
`answer`, `question_type`, `reference_response`, `source_id`, `dataset_meta` —
are FLAT top-level sample fields alongside `id`/`media`/`messages`; messages[0]
carries render inputs only (question/options/choices/hint/prompt); the
prediction stays at messages[-1].response.

The script handles files from every era:
  * message-embedded layout (grading fields inside messages[0]): each grading field
    found in messages[0] is lifted to the sample top level and removed from the
    message. A value already present at the top level is NEVER overwritten (the
    stale messages[0] copy is dropped either way).
  * flat layout (`answer`/`question_type` already at the sample top
    level): already canonical for those fields — reported as such, no-op.
  * canonical files: no-op (idempotent).

Existing result.json files — expensive GPU artifacts — stay scoreable afterwards.

Usage:
    python3 scripts/migrate_result_layout.py <result.json> [more.json ...]
    python3 scripts/migrate_result_layout.py --dry-run <result.json>
    python3 scripts/migrate_result_layout.py --out migrated.json <result.json>

Default is in-place rewrite (a `.bak` copy is written first). --dry-run reports
what would change and touches nothing. --out writes to a new file (single input
only). Exit code 0 on success, 1 on any error.
"""
import argparse
import json
import shutil
import sys

GRADING_FIELDS = ("answer", "question_type", "reference_response", "source_id", "dataset_meta")


def migrate_samples(samples):
    """Returns (migrated_count, already_canonical_count). Mutates samples."""
    migrated = 0
    canonical = 0
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        messages = sample.get("messages")
        if not (isinstance(messages, list) and messages and isinstance(messages[0], dict)):
            continue
        msg = messages[0]
        moved = False
        for field in GRADING_FIELDS:
            if field in msg:
                if field not in sample:
                    sample[field] = msg[field]
                del msg[field]  # drop the stale messages[0] copy either way
                moved = True
        if moved:
            migrated += 1
        else:
            canonical += 1
    return migrated, canonical


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="result.json files to migrate")
    parser.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    parser.add_argument("--out", default=None, help="write to this path instead of in place (single input only)")
    args = parser.parse_args(argv)

    if args.out and len(args.files) != 1:
        parser.error("--out accepts exactly one input file")

    failed = False
    for path in args.files:
        try:
            with open(path, "r") as f:
                samples = json.load(f)
            if not isinstance(samples, list):
                raise ValueError("not a JSON list of samples")
            migrated, canonical = migrate_samples(samples)
            if args.dry_run:
                print(f"[dry-run] {path}: would migrate {migrated} sample(s); {canonical} already canonical")
                continue
            if migrated == 0:
                print(f"[skip] {path}: already canonical ({canonical} samples)")
                continue
            out_path = args.out or path
            if out_path == path:
                shutil.copyfile(path, f"{path}.bak")
            with open(out_path, "w") as f:
                json.dump(samples, f, indent=4)
            print(f"[ok] {path} -> {out_path}: migrated {migrated} sample(s)"
                  + ("" if args.out else f" (backup: {path}.bak)"))
        except Exception as exc:
            print(f"[error] {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
