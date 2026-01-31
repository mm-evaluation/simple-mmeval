#!/usr/bin/env python3
"""
Task Count Verification and Cache Recovery Script

This script:
1. Checks if task counts in work_dirs match the source JSON
2. If mismatch detected, converts result.json back to cache.db
   so the inference program can resume from where it left off

Usage:
    python check_and_recover_cache.py \
        --source-json /path/to/source.json \
        --work-dir /path/to/work_dirs/dataset_name

    # Or with defaults for cogcontrol3.6:
    python check_and_recover_cache.py
"""

import os
import sys
import json
import argparse
import sqlite3
from typing import Dict, Any, List, Tuple

# Add parent directory to path so we can import from mmeval
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmeval.utils.sqlitkv import SQLiteKVStore


def load_source_json(json_path: str) -> List[Dict[str, Any]]:
    """Load source JSON file and return list of tasks."""
    print(f"Loading source JSON: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  Total tasks in source: {len(data)}")
    return data


def load_result_json(result_path: str) -> List[Dict[str, Any]]:
    """Load result.json file from work directory."""
    if not os.path.exists(result_path):
        return []
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def get_work_subdirs(work_dir: str) -> List[str]:
    """Get all subdirectories in the work directory."""
    if not os.path.exists(work_dir):
        print(f"Work directory does not exist: {work_dir}")
        return []
    
    subdirs = []
    for name in sorted(os.listdir(work_dir)):
        subdir_path = os.path.join(work_dir, name)
        if os.path.isdir(subdir_path):
            subdirs.append(subdir_path)
    return subdirs


def check_cache_db(cache_path: str) -> int:
    """Check how many entries are in cache.db, return -1 if doesn't exist."""
    if not os.path.exists(cache_path):
        return -1
    try:
        store = SQLiteKVStore(cache_path)
        keys = store.keys()
        return len(keys)
    except Exception as e:
        print(f"  Error reading cache.db: {e}")
        return -1


def convert_result_to_cache(result_json_path: str, cache_db_path: str, backup: bool = True) -> int:
    """
    Convert result.json to cache.db format.
    
    The cache.db uses SQLiteKVStore with:
    - key: str(eval-id)
    - value: the complete sample dict (with response)
    
    Returns the number of items written.
    """
    import shutil
    
    # Load result.json
    results = load_result_json(result_json_path)
    if not results:
        print(f"  No results found in {result_json_path}")
        return 0
    
    # Backup result.json and then delete it
    if backup and os.path.exists(result_json_path):
        result_backup_path = result_json_path + ".backup"
        print(f"  Backing up result.json to {result_backup_path}")
        shutil.copy2(result_json_path, result_backup_path)
        print(f"  Deleting original result.json")
        os.remove(result_json_path)
    
    # Backup existing cache.db if it exists
    if backup and os.path.exists(cache_db_path):
        backup_path = cache_db_path + ".backup"
        print(f"  Backing up existing cache.db to {backup_path}")
        shutil.copy2(cache_db_path, backup_path)
        # Also backup WAL and SHM files if they exist
        for ext in ['-wal', '-shm']:
            wal_path = cache_db_path + ext
            if os.path.exists(wal_path):
                shutil.copy2(wal_path, backup_path + ext)
    
    # Remove existing cache.db to start fresh
    if os.path.exists(cache_db_path):
        os.remove(cache_db_path)
        for ext in ['-wal', '-shm']:
            wal_path = cache_db_path + ext
            if os.path.exists(wal_path):
                os.remove(wal_path)
    
    # Create new cache.db and populate it
    store = SQLiteKVStore(cache_db_path)
    
    # Prepare batch items: (str(eval-id), sample_dict)
    items = []
    for sample in results:
        if 'eval-id' not in sample:
            print(f"  Warning: sample missing eval-id, skipping")
            continue
        eval_id = str(sample['eval-id'])
        items.append((eval_id, sample))
    
    # Batch insert for efficiency
    if items:
        store.batch_put(items)
    
    print(f"  Wrote {len(items)} items to cache.db")
    return len(items)


def analyze_work_dir(subdir: str, expected_count: int) -> Tuple[str, int, int, bool]:
    """
    Analyze a single work subdirectory.
    
    Returns: (name, result_count, cache_count, needs_recovery)
    """
    name = os.path.basename(subdir)
    result_path = os.path.join(subdir, 'result.json')
    cache_path = os.path.join(subdir, 'cache.db')
    
    result_data = load_result_json(result_path)
    result_count = len(result_data)
    cache_count = check_cache_db(cache_path)
    
    # Needs recovery if:
    # 1. Result count < expected (incomplete) AND
    # 2. Cache doesn't exist or cache count doesn't match result count
    is_incomplete = result_count < expected_count and result_count > 0
    cache_missing = cache_count == -1
    cache_mismatch = cache_count != -1 and cache_count != result_count
    
    needs_recovery = is_incomplete and (cache_missing or cache_mismatch)
    
    return name, result_count, cache_count, needs_recovery


def main():
    parser = argparse.ArgumentParser(
        description='Check task counts and recover cache.db from result.json'
    )
    parser.add_argument(
        '--source-json',
        default='/data/ztw/data/cogcontrol4/cogcontrol4_frame/p0/cogcontrol4_total_p0.json',
        help='Path to source JSON file with all tasks'
    )
    parser.add_argument(
        '--work-dir',
        default='/data/ztw/simple-mmeval-infer-data/work_dirs/cogcontrol4',
        help='Path to work directory containing model subdirectories'
    )
    parser.add_argument(
        '--recover',
        action='store_true',
        help='Actually perform recovery (convert result.json to cache.db). Without this flag, only reports status.'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not backup existing cache.db before overwriting'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Only process specific model subdirectory (e.g., "Qwen2.5-VL-32B-Instruct-cogcontrol3.6")'
    )
    
    args = parser.parse_args()
    
    # Load source JSON to get expected task count
    source_data = load_source_json(args.source_json)
    expected_count = len(source_data)
    
    # Get work subdirectories
    subdirs = get_work_subdirs(args.work_dir)
    if not subdirs:
        print("No subdirectories found in work directory")
        return 1
    
    # Filter by model if specified
    if args.model:
        subdirs = [s for s in subdirs if args.model in os.path.basename(s)]
        if not subdirs:
            print(f"No subdirectory matching '{args.model}' found")
            return 1
    
    print(f"\nAnalyzing {len(subdirs)} work directories...")
    print("=" * 80)
    
    # Analyze all directories
    complete = []
    incomplete = []
    needs_recovery = []
    
    for subdir in subdirs:
        name, result_count, cache_count, needs_rec = analyze_work_dir(subdir, expected_count)
        
        cache_status = f"{cache_count}" if cache_count >= 0 else "missing"
        status_icon = "✓" if result_count == expected_count else "○"
        
        if result_count == expected_count:
            complete.append(name)
        else:
            incomplete.append((name, result_count))
            if needs_rec:
                needs_recovery.append(subdir)
        
        print(f"{status_icon} {name}")
        print(f"    result.json: {result_count}/{expected_count} tasks")
        print(f"    cache.db: {cache_status}")
        if needs_rec:
            print(f"    → NEEDS RECOVERY")
    
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Complete: {len(complete)}/{len(subdirs)}")
    print(f"  Incomplete: {len(incomplete)}/{len(subdirs)}")
    print(f"  Need recovery: {len(needs_recovery)}")
    
    if incomplete:
        print(f"\nIncomplete models:")
        for name, count in sorted(incomplete, key=lambda x: x[1]):
            remaining = expected_count - count
            print(f"  - {name}: {count}/{expected_count} ({remaining} remaining)")
    
    # Perform recovery if requested
    if args.recover and needs_recovery:
        print(f"\n{'=' * 80}")
        print("RECOVERY MODE")
        print("=" * 80)
        
        for subdir in needs_recovery:
            name = os.path.basename(subdir)
            result_path = os.path.join(subdir, 'result.json')
            cache_path = os.path.join(subdir, 'cache.db')
            
            print(f"\nRecovering: {name}")
            written = convert_result_to_cache(
                result_path, 
                cache_path, 
                backup=not args.no_backup
            )
            
            # Verify the recovery
            verify_count = check_cache_db(cache_path)
            if verify_count == written:
                print(f"  ✓ Recovery successful: {written} items in cache.db")
            else:
                print(f"  ✗ Recovery verification failed: expected {written}, got {verify_count}")
        
        print(f"\n{'=' * 80}")
        print(f"Recovery complete. {len(needs_recovery)} directories processed.")
        print("You can now resume inference for these models.")
    
    elif needs_recovery and not args.recover:
        print(f"\n⚠ {len(needs_recovery)} directories need recovery.")
        print("Run with --recover flag to convert result.json to cache.db")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
