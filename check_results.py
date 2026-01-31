#!/usr/bin/env python3
"""
Result checker script to verify model inference results match original data task counts.
"""

import json
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Import MODEL_CONFIGS from batch_infer.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from batch_infer import MODEL_CONFIGS


# Modality to data file mapping (from batch_infer.py)
MODALITY_MAPPING = {
    "single_image_start": ("frame", "single_image"),
    "single_video_start": ("video", "single_video"),
    "multi_video_interleave": ("video", "total"),
    "multi_image_video_interleave": ("video", "total"),
    "multi_image_interleave": ("frame", "total"),
    "multi_image_start": ("frame", "single_video"),
}


def get_modality_paths(dataset: str, modality: str, prompt: str) -> str:
    """Get original data file path based on modality mapping."""
    if modality not in MODALITY_MAPPING:
        return None
    
    subdir_type, file_type = MODALITY_MAPPING[modality]
    infile_path = f"/data/ztw/data/{dataset}/{dataset}_{subdir_type}/{prompt}/{dataset}_{file_type}_{prompt}.json"
    return infile_path


def extract_model_name_from_dir(dir_name: str, dataset: str) -> str:
    """Extract model name from directory name.
    
    Example: 'InternVL2-1B-conservation-all' -> 'InternVL2-1B'
    """
    # Remove dataset suffix with dash
    suffix = f"-{dataset.replace('_', '-')}"
    if dir_name.endswith(suffix):
        return dir_name[:-len(suffix)]
    return dir_name


def count_tasks_in_json(json_file: Path) -> Optional[int]:
    """Count number of tasks in a JSON file."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            return len(data)
        return None
    except Exception as e:
        print(f"  Error reading {json_file}: {e}")
        return None


def detect_prompts(dataset: str) -> List[str]:
    """Detect available prompts by scanning original data directories."""
    prompts = set()
    data_base = Path(f"/data/ztw/data/{dataset}")
    
    if not data_base.exists():
        return []
    
    # Scan both frame and video subdirectories
    for subdir in data_base.iterdir():
        if subdir.is_dir() and (subdir.name.startswith(f"{dataset}_frame") or 
                                 subdir.name.startswith(f"{dataset}_video")):
            for prompt_dir in subdir.iterdir():
                if prompt_dir.is_dir() and prompt_dir.name.startswith('p'):
                    prompts.add(prompt_dir.name)
    
    return sorted(list(prompts))


def check_dataset(dataset: str) -> Dict:
    """Check all models for a given dataset."""
    base_dir = Path("/data/ztw/simple-mmeval-infer-data")
    work_dir = base_dir / "work_dirs" / dataset
    
    if not work_dir.exists():
        print(f"Work directory not found: {work_dir}")
        return {
            "dataset": dataset,
            "check_time": datetime.now().isoformat(),
            "error": f"Work directory not found: {work_dir}",
            "mismatches": [],
            "summary": {
                "total_models_checked": 0,
                "models_with_mismatches": 0,
                "total_mismatches": 0
            }
        }
    
    # Detect available prompts
    prompts = detect_prompts(dataset)
    if not prompts:
        print(f"No prompts found for dataset: {dataset}")
        prompts = ["p0"]  # Default fallback
    
    print(f"Checking dataset: {dataset}")
    print(f"Available prompts: {prompts}")
    print(f"Scanning work directory: {work_dir}")
    
    mismatches = []
    models_checked = set()
    models_with_mismatches = set()
    
    # Scan work_dirs for model directories
    for model_dir in sorted(work_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        
        # Extract model name
        model_name = extract_model_name_from_dir(model_dir.name, dataset)
        models_checked.add(model_name)
        
        print(f"\nChecking model: {model_name} (dir: {model_dir.name})")
        
        # Check if result.json exists
        result_file = model_dir / "result.json"
        if not result_file.exists():
            print(f"  ⚠️  No result.json found, skipping")
            continue
        
        # Count tasks in result.json
        result_count = count_tasks_in_json(result_file)
        if result_count is None:
            print(f"  ⚠️  Could not read result.json, skipping")
            continue
        
        print(f"  Result tasks: {result_count}")
        
        # Get model's supported modalities
        if model_name in MODEL_CONFIGS:
            modalities = MODEL_CONFIGS[model_name].get("modalities", [])
        else:
            print(f"  ⚠️  Model not in MODEL_CONFIGS, skipping modality check")
            continue
        
        print(f"  Supported modalities: {modalities}")
        
        # Check each (modality, prompt) combination
        for modality in modalities:
            for prompt in prompts:
                original_file = get_modality_paths(dataset, modality, prompt)
                
                if original_file is None:
                    continue
                
                original_path = Path(original_file)
                if not original_path.exists():
                    print(f"  ⚠️  Original file not found: {original_file}")
                    continue
                
                # Count tasks in original file
                original_count = count_tasks_in_json(original_path)
                if original_count is None:
                    continue
                
                # Compare counts
                if original_count != result_count:
                    difference = result_count - original_count
                    print(f"  ❌ MISMATCH: {modality}/{prompt} - Original: {original_count}, Result: {result_count}, Diff: {difference:+d}")
                    
                    mismatches.append({
                        "model": model_name,
                        "work_dir": model_dir.name,
                        "modality": modality,
                        "prompt": prompt,
                        "original_file": str(original_file),
                        "original_count": original_count,
                        "result_count": result_count,
                        "difference": difference
                    })
                    models_with_mismatches.add(model_name)
                else:
                    print(f"  ✅ OK: {modality}/{prompt} - {original_count} tasks")
    
    # Generate summary
    summary = {
        "total_models_checked": len(models_checked),
        "models_with_mismatches": len(models_with_mismatches),
        "total_mismatches": len(mismatches)
    }
    
    result = {
        "dataset": dataset,
        "check_time": datetime.now().isoformat(),
        "prompts_checked": prompts,
        "mismatches": mismatches,
        "summary": summary
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Check if model inference results match original data task counts"
    )
    parser.add_argument("dataset", help="Dataset name (e.g., conservation_all)")
    args = parser.parse_args()
    
    # Run check
    result = check_dataset(args.dataset)
    
    # Save results
    result_check_dir = Path("/data/ztw/simple-mmeval-infer-data/result_check")
    result_check_dir.mkdir(exist_ok=True)
    
    output_file = result_check_dir / f"{args.dataset}_check.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n{'='*60}")
    print("CHECK SUMMARY")
    print(f"{'='*60}")
    print(f"Dataset: {result['dataset']}")
    print(f"Total models checked: {result['summary']['total_models_checked']}")
    print(f"Models with mismatches: {result['summary']['models_with_mismatches']}")
    print(f"Total mismatches: {result['summary']['total_mismatches']}")
    print(f"\nResults saved to: {output_file}")
    
    if result['summary']['total_mismatches'] > 0:
        print(f"\n⚠️  Found {result['summary']['total_mismatches']} mismatch(es)!")
        sys.exit(1)
    else:
        print(f"\n✅ All checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()

