#!/usr/bin/env python3
"""
Script to move mismatch model directories based on check results.
"""

import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple


def load_check_results(result_check_dir: Path) -> List[Dict]:
    """Load all check result JSON files."""
    all_mismatches = []
    
    for json_file in sorted(result_check_dir.glob("*_check.json")):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            dataset = data.get('dataset')
            mismatches = data.get('mismatches', [])
            
            print(f"Loaded {json_file.name}: {len(mismatches)} mismatches from dataset '{dataset}'")
            
            # Add dataset info to each mismatch if not present
            for mismatch in mismatches:
                if 'dataset' not in mismatch:
                    mismatch['dataset'] = dataset
                all_mismatches.append(mismatch)
                
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return all_mismatches


def get_unique_work_dirs(mismatches: List[Dict]) -> List[Tuple[str, str]]:
    """Extract unique (dataset, work_dir) pairs from mismatches."""
    unique_dirs = set()
    
    for mismatch in mismatches:
        dataset = mismatch.get('dataset')
        work_dir = mismatch.get('work_dir')
        if dataset and work_dir:
            unique_dirs.add((dataset, work_dir))
    
    return sorted(list(unique_dirs))


def move_mismatch_directories(dry_run: bool = True):
    """Move mismatch directories from work_dirs to mismatch_work_dirs."""
    base_dir = Path("/data/ztw/simple-mmeval-infer-data")
    result_check_dir = base_dir / "result_check"
    work_dirs_base = base_dir / "work_dirs"
    mismatch_base = base_dir / "mismatch_work_dirs"
    
    # Load all check results
    print("="*70)
    print("LOADING CHECK RESULTS")
    print("="*70)
    all_mismatches = load_check_results(result_check_dir)
    
    if not all_mismatches:
        print("No mismatches found in any check result files.")
        return
    
    # Get unique work directories
    unique_dirs = get_unique_work_dirs(all_mismatches)
    
    print(f"\n{'='*70}")
    print(f"FOUND {len(unique_dirs)} UNIQUE WORK DIRECTORIES WITH MISMATCHES")
    print("="*70)
    
    # Prepare move operations
    move_operations = []
    missing_sources = []
    existing_targets = []
    
    for dataset, work_dir in unique_dirs:
        source = work_dirs_base / dataset / work_dir
        target = mismatch_base / dataset / work_dir
        
        # Check if source exists
        if not source.exists():
            missing_sources.append((dataset, work_dir, str(source)))
            continue
        
        # Check if target already exists
        if target.exists():
            existing_targets.append((dataset, work_dir, str(target)))
            continue
        
        move_operations.append({
            'dataset': dataset,
            'work_dir': work_dir,
            'source': source,
            'target': target
        })
    
    # Display summary
    print(f"\n{'='*70}")
    print("MOVE OPERATIONS SUMMARY")
    print("="*70)
    print(f"Total directories to move: {len(move_operations)}")
    print(f"Source directories not found: {len(missing_sources)}")
    print(f"Target directories already exist: {len(existing_targets)}")
    
    # Show missing sources
    if missing_sources:
        print(f"\n⚠️  WARNING: {len(missing_sources)} source directories not found:")
        for dataset, work_dir, source in missing_sources[:10]:
            print(f"  - {dataset}/{work_dir}")
        if len(missing_sources) > 10:
            print(f"  ... and {len(missing_sources) - 10} more")
    
    # Show existing targets
    if existing_targets:
        print(f"\n⚠️  WARNING: {len(existing_targets)} target directories already exist (will skip):")
        for dataset, work_dir, target in existing_targets[:10]:
            print(f"  - {dataset}/{work_dir}")
        if len(existing_targets) > 10:
            print(f"  ... and {len(existing_targets) - 10} more")
    
    # Show operations to be performed
    if move_operations:
        print(f"\n{'='*70}")
        print(f"DIRECTORIES TO BE MOVED ({len(move_operations)} total)")
        print("="*70)
        
        # Group by dataset
        by_dataset = {}
        for op in move_operations:
            dataset = op['dataset']
            if dataset not in by_dataset:
                by_dataset[dataset] = []
            by_dataset[dataset].append(op)
        
        for dataset, ops in sorted(by_dataset.items()):
            print(f"\nDataset: {dataset} ({len(ops)} directories)")
            for op in ops[:20]:  # Show first 20 per dataset
                print(f"  ✓ {op['work_dir']}")
            if len(ops) > 20:
                print(f"  ... and {len(ops) - 20} more")
    
    # Perform move if not dry-run
    if not dry_run and move_operations:
        print(f"\n{'='*70}")
        print("EXECUTING MOVE OPERATIONS")
        print("="*70)
        
        moved_count = 0
        failed_count = 0
        
        for op in move_operations:
            try:
                # Create target parent directory
                op['target'].parent.mkdir(parents=True, exist_ok=True)
                
                # Move directory
                shutil.move(str(op['source']), str(op['target']))
                moved_count += 1
                print(f"✓ Moved: {op['dataset']}/{op['work_dir']}")
                
            except Exception as e:
                failed_count += 1
                print(f"✗ Failed to move {op['dataset']}/{op['work_dir']}: {e}")
        
        print(f"\n{'='*70}")
        print("MOVE OPERATIONS COMPLETE")
        print("="*70)
        print(f"Successfully moved: {moved_count}")
        print(f"Failed: {failed_count}")
        
        # Save move log
        log_file = base_dir / f"move_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_moved': moved_count,
                'total_failed': failed_count,
                'operations': [
                    {
                        'dataset': op['dataset'],
                        'work_dir': op['work_dir'],
                        'source': str(op['source']),
                        'target': str(op['target'])
                    }
                    for op in move_operations
                ]
            }, f, indent=2)
        print(f"\nMove log saved to: {log_file}")
    
    elif dry_run:
        print(f"\n{'='*70}")
        print("DRY RUN MODE - NO CHANGES MADE")
        print("="*70)
        print("Run with --execute flag to actually move the directories.")
        print("Command: python3 move_mismatches.py --execute")


def main():
    parser = argparse.ArgumentParser(
        description="Move mismatch model directories to mismatch_work_dirs"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute the move operations (default is dry-run)"
    )
    
    args = parser.parse_args()
    
    move_mismatch_directories(dry_run=not args.execute)


if __name__ == "__main__":
    main()

