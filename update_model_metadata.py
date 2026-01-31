#!/usr/bin/env python3
"""
Maintenance script to fetch HuggingFace metadata for all models in MODEL_CONFIGS
and generate an updated version with additional metadata fields.

Usage:
    python update_model_metadata.py

Output:
    - MODEL_CONFIGS_updated.py: Enhanced MODEL_CONFIGS with HF metadata
    - Console output: Summary and list of failed models
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import re
import time
import json

# Add current directory to Python path to import batch_infer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import MODEL_CONFIGS from batch_infer
from batch_infer import MODEL_CONFIGS

# Import HuggingFace Hub API
try:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
except ImportError:
    print("ERROR: huggingface_hub is not installed. Please run: pip install huggingface_hub")
    sys.exit(1)


def extract_parameter_count(info) -> Optional[int]:
    """
    Extract parameter count from HuggingFace API model info.
    
    Only extracts from real API data, does NOT infer from model names.
    
    Tries multiple API data sources:
    1. info.safetensors.total - SafeTensors metadata
    2. info.cardData - Model card YAML metadata
    3. info.config - Model configuration
    
    Returns:
        Parameter count as integer, or None if unavailable in API data
    """
    # Method 1: SafeTensors metadata (most reliable)
    if hasattr(info, 'safetensors') and info.safetensors:
        # Check object attribute
        if hasattr(info.safetensors, 'total'):
            total_val = info.safetensors.total
            if total_val is not None and isinstance(total_val, (int, float)):
                return int(total_val)
        # Check dict key
        if isinstance(info.safetensors, dict) and 'total' in info.safetensors:
            total_val = info.safetensors['total']
            if total_val is not None and isinstance(total_val, (int, float)):
                return int(total_val)
    
    # Method 2: Model card YAML front-matter
    if hasattr(info, 'cardData') and info.cardData:
        card_data = info.cardData
        # Check all possible field names in cardData
        field_names = [
            'parameters', 'parameter_count', 'num_parameters',
            'model_parameters', 'params', 'n_params', 
            'total_params', 'num_params', 'model_params',
            'model_size'  # Sometimes used for parameter count
        ]
        
        for field in field_names:
            # Check as object attribute
            if hasattr(card_data, field):
                val = getattr(card_data, field)
                if val and isinstance(val, (int, float)):
                    return int(val)
                # Also try as string (might be "8B" or "8 billion")
                if isinstance(val, str):
                    val_parsed = parse_parameter_string(val)
                    if val_parsed:
                        return val_parsed
            
            # Check as dict key
            if isinstance(card_data, dict) and field in card_data:
                val = card_data[field]
                if val and isinstance(val, (int, float)):
                    return int(val)
                # Also try as string
                if isinstance(val, str):
                    val_parsed = parse_parameter_string(val)
                    if val_parsed:
                        return val_parsed
    
    # Method 3: Model config (if available)
    if hasattr(info, 'config') and info.config:
        config = info.config
        # Some models store parameter info in config
        for field in ['num_parameters', 'n_params', 'total_params']:
            if isinstance(config, dict) and field in config:
                val = config[field]
                if val and isinstance(val, (int, float)):
                    return int(val)
    
    # If no real data found, return None (do NOT infer from model name)
    return None


def parse_parameter_string(s: str) -> Optional[int]:
    """
    Parse parameter count from string like "8B", "8 billion", "8000000000".
    
    Args:
        s: String potentially containing parameter count
        
    Returns:
        Parameter count as integer, or None
    """
    s = s.strip().lower()
    
    # Try direct number
    if s.isdigit():
        return int(s)
    
    # Try with billion suffix: "8b", "8 billion", "8.5b"
    import re
    match = re.search(r'(\d+\.?\d*)\s*(?:b|billion)', s)
    if match:
        billions = float(match.group(1))
        return int(billions * 1_000_000_000)
    
    # Try with million suffix: "500m", "500 million"
    match = re.search(r'(\d+\.?\d*)\s*(?:m|million)', s)
    if match:
        millions = float(match.group(1))
        return int(millions * 1_000_000)
    
    return None


def extract_training_dataset(info) -> Optional[Any]:
    """
    Extract training dataset information from model card data.
    
    Checks multiple field names where dataset info might be stored.
    
    Args:
        info: HuggingFace model info object
        
    Returns:
        Dataset information (string, list, or dict), or None if not found
    """
    if not hasattr(info, 'cardData') or not info.cardData:
        return None
    
    card_data = info.cardData
    
    # Try to get dataset information from various field names
    # Following the same approach as get_hf_information.py
    if isinstance(card_data, dict):
        dataset_info = (
            card_data.get("datasets") or
            card_data.get("dataset") or
            card_data.get("dataset_name")
        )
        return dataset_info
    
    # If cardData is an object (not dict), try attributes
    for field_name in ['datasets', 'dataset', 'dataset_name']:
        if hasattr(card_data, field_name):
            val = getattr(card_data, field_name)
            if val:
                return val
    
    return None


def extract_model_size_bytes(info) -> Optional[int]:
    """Extract total model size in bytes from safetensors metadata."""
    if hasattr(info, 'safetensors') and info.safetensors:
        total_size = 0
        try:
            # safetensors is a dict with parameters info
            if isinstance(info.safetensors, dict):
                if 'total' in info.safetensors:
                    # This might be parameter count, not size
                    pass
                # Try to get file sizes
                for key, value in info.safetensors.items():
                    if isinstance(value, dict) and 'size' in value:
                        size_val = value['size']
                        if size_val is not None and isinstance(size_val, (int, float)):
                            total_size += int(size_val)
        except Exception:
            pass
        
        if total_size > 0:
            return total_size
    
    # Alternative: check siblings for model weight files
    if hasattr(info, 'siblings') and info.siblings:
        total_size = 0
        # Support multiple model file formats
        model_extensions = (
            '.safetensors',  # SafeTensors format (most common)
            '.bin',          # PyTorch binary
            '.pt',           # PyTorch
            '.pth',          # PyTorch
            '.h5',           # TensorFlow/Keras
            '.ckpt',         # TensorFlow checkpoint
            '.pb',           # TensorFlow SavedModel
        )
        
        for sibling in info.siblings:
            # sibling is a dict-like object
            if hasattr(sibling, 'rfilename') and hasattr(sibling, 'size'):
                # Check if file is a model weight file
                if sibling.rfilename.endswith(model_extensions):
                    size_val = sibling.size
                    if size_val is not None and isinstance(size_val, (int, float)):
                        total_size += int(size_val)
        
        if total_size > 0:
            return total_size
    
    return None


def fetch_model_metadata(hf_path: str, api: HfApi, max_retries: int = 3, base_delay: float = 2.0) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Fetch metadata for a model from HuggingFace Hub with retry logic.
    
    Args:
        hf_path: HuggingFace model path (e.g., "meta-llama/Llama-3-8B")
        api: HfApi instance
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        
    Returns:
        Tuple of (metadata_dict, error_message)
        If successful, metadata_dict contains the fields and error_message is None
        If failed, metadata_dict is None and error_message contains the error
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # Request files_metadata to get siblings information for model size calculation
            info = api.model_info(hf_path, files_metadata=True)
            
            # Extract metadata fields
            metadata = {
                "created_at": info.created_at.date().isoformat() if hasattr(info, 'created_at') and info.created_at else None,
                "last_modified": info.lastModified.date().isoformat() if hasattr(info, 'lastModified') and info.lastModified else None,
                "downloads": info.downloads if hasattr(info, 'downloads') else None,
                "likes": info.likes if hasattr(info, 'likes') else None,
                "license": info.cardData.get('license') if hasattr(info, 'cardData') and info.cardData else None,
                "tags": info.tags[:10] if hasattr(info, 'tags') and info.tags else None,  # Limit to 10 tags to keep concise
                "pipeline_tag": info.pipeline_tag if hasattr(info, 'pipeline_tag') else None,
                "library_name": info.library_name if hasattr(info, 'library_name') else None,
                "model_size_bytes": extract_model_size_bytes(info),
                "parameter_count": extract_parameter_count(info),
                "training_dataset": extract_training_dataset(info),
            }
            
            return metadata, None
            
        except RepositoryNotFoundError:
            return None, "Repository not found (deleted or never existed)"
        except HfHubHTTPError as e:
            if "401" in str(e) or "403" in str(e):
                return None, f"Access denied (private/gated model)"
            else:
                last_error = f"HTTP error: {str(e)}"
                # Retry for HTTP errors
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    time.sleep(delay)
                    continue
                return None, last_error
        except Exception as e:
            error_type = type(e).__name__
            last_error = f"{error_type}: {str(e)[:100]}"
            
            # Retry for connection errors
            if "Connection" in error_type or "Timeout" in error_type:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    time.sleep(delay)
                    continue
            
            return None, last_error
    
    return None, last_error


def update_all_model_configs(api: HfApi, request_delay: float = 1.0, 
                             output_file: str = "MODEL_CONFIGS_updated.json") -> Tuple[Dict[str, Dict], List[Tuple[str, str, str]]]:
    """
    Update all MODEL_CONFIGS with HuggingFace metadata.
    
    Args:
        api: HfApi instance
        request_delay: Delay in seconds between requests to avoid rate limiting
        output_file: Output JSON file path
        
    Returns:
        Tuple of (updated_configs, failed_models)
        - updated_configs: Dictionary with all models and their enhanced configs (only successful ones)
        - failed_models: List of (model_name, hf_path, error_message) tuples
    """
    output_path = Path(output_file)
    
    # Load existing output if available
    if output_path.exists():
        print(f"📂 Loading existing results from {output_file}...")
        with open(output_path, 'r', encoding='utf-8') as f:
            updated_configs = json.load(f)
            processed_models = set(updated_configs.keys())
        print(f"✓ Found {len(processed_models)} models already processed")
    else:
        updated_configs = {}
        processed_models = set()
    
    # Failed models list (only for this session)
    failed_models = []
    
    total_models = len(MODEL_CONFIGS)
    remaining_models = total_models - len(processed_models)
    # Initialize counters - ensure they are integers, not None
    success_count = sum(1 for m in updated_configs.values() if m.get('downloads') is not None) or 0
    fail_count = len(failed_models) if failed_models else 0
    
    print(f"\n📊 Processing Status:")
    print(f"   Total models: {total_models}")
    print(f"   Remaining: {remaining_models}")
    print(f"   Already processed: {len(processed_models)} (✓ {success_count} successful, ✗ {fail_count} failed)")
    print("=" * 80)
    
    start_time = time.time()
    current_batch_start = time.time()
    
    try:
        for idx, (model_name, config) in enumerate(MODEL_CONFIGS.items(), 1):
            # Skip already processed models
            if model_name in processed_models:
                continue
            
            hf_path = config.get("hf_path")
            
            if not hf_path:
                print(f"[{idx}/{total_models}] ⚠️  {model_name}: No hf_path defined, skipping")
                failed_models.append((model_name, "N/A", "No hf_path in config"))
                fail_count += 1
                # Do NOT add to updated_configs - failed models are excluded
                continue
            
            # Progress indicator
            progress_pct = (len(processed_models) / total_models) * 100
            print(f"\n[{idx}/{total_models}] ({progress_pct:.1f}%) 🔄 Fetching: {model_name}")
            print(f"   HF Path: {hf_path}")
            print(f"   Status: ", end="", flush=True)
            
            # Add delay before request (except for first request)
            if len(processed_models) > 0:
                time.sleep(request_delay)
            
            request_start = time.time()
            metadata, error = fetch_model_metadata(hf_path, api)
            request_duration = time.time() - request_start
            
            if error:
                fail_count += 1
                print(f"❌ FAILED ({request_duration:.1f}s)")
                print(f"   Error: {error}")
                failed_models.append((model_name, hf_path, error))
                # Do NOT add to updated_configs - failed models are excluded from JSON
            else:
                success_count += 1
                print(f"✅ SUCCESS ({request_duration:.1f}s)")
                # Show key metadata
                if metadata.get('downloads'):
                    print(f"   Downloads: {metadata['downloads']:,}")
                if metadata.get('likes'):
                    print(f"   Likes: {metadata['likes']:,}")
                # Merge original config with new metadata
                updated_config = config.copy()
                updated_config.update(metadata)
                updated_configs[model_name] = updated_config
                processed_models.add(model_name)
            
            # Calculate ETA
            elapsed = time.time() - start_time
            avg_time_per_model = elapsed / len(processed_models) if len(processed_models) > 0 else 0
            remaining_models = total_models - len(processed_models)
            eta_seconds = avg_time_per_model * remaining_models
            eta_minutes = eta_seconds / 60
            
            print(f"   Progress: ✓ {success_count} successful | ✗ {fail_count} failed | ⏱️  ETA: {eta_minutes:.1f} min")
            
            # Save progress every 10 models
            if len(processed_models) % 10 == 0:
                save_json_output(output_path, updated_configs)
                print(f"   💾 Auto-saved to JSON")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Saving progress...")
        save_json_output(output_path, updated_configs)
        print(f"✓ Progress saved to {output_file}")
        print(f"✓ Successful models: {len(updated_configs)}/{total_models}")
        print(f"✓ Run the script again to resume from this checkpoint.")
        sys.exit(1)
    
    # Save final output
    save_json_output(output_path, updated_configs)
    
    return updated_configs, failed_models


def save_json_output(output_path: Path, updated_configs: Dict[str, Dict]) -> None:
    """Save successfully processed models to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(updated_configs, f, indent=2, ensure_ascii=False)




def print_summary(updated_configs: Dict[str, Dict], failed_models: List[Tuple[str, str, str]]) -> None:
    """Print summary statistics and failed models list."""
    total = len(MODEL_CONFIGS)
    success = len(updated_configs)  # Only successfully fetched models are in updated_configs
    failed = len(failed_models)
    
    print("\n" + "=" * 80)
    print("📊 FINAL SUMMARY")
    print("=" * 80)
    
    # Overall statistics
    print(f"\n🔢 Overall Results:")
    print(f"   Total models: {total}")
    print(f"   ✅ Successfully fetched metadata: {success}")
    print(f"   ❌ Failed to fetch metadata: {failed}")
    print(f"   📝 Included in JSON: {success}")
    print(f"   🚫 Excluded from JSON: {failed}")
    
    if success > 0:
        success_rate = success/total*100
        print(f"   📈 Success rate: {success_rate:.1f}%")
        
        # Visual progress bar
        bar_length = 50
        filled = int(bar_length * success / total)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"   Progress: [{bar}] {success}/{total}")
    
    # Print statistics about metadata coverage
    has_downloads = sum(1 for c in updated_configs.values() if c.get('downloads') is not None)
    has_params = sum(1 for c in updated_configs.values() if c.get('parameter_count') is not None)
    has_size = sum(1 for c in updated_configs.values() if c.get('model_size_bytes') is not None)
    has_license = sum(1 for c in updated_configs.values() if c.get('license') is not None)
    has_created = sum(1 for c in updated_configs.values() if c.get('created_at') is not None)
    has_modified = sum(1 for c in updated_configs.values() if c.get('last_modified') is not None)
    
    print(f"\n📋 Metadata Field Coverage:")
    print(f"   Downloads:       {has_downloads:3d}/{total} ({has_downloads/total*100:5.1f}%)")
    print(f"   Parameter count: {has_params:3d}/{total} ({has_params/total*100:5.1f}%)")
    print(f"   Model size:      {has_size:3d}/{total} ({has_size/total*100:5.1f}%)")
    print(f"   License info:    {has_license:3d}/{total} ({has_license/total*100:5.1f}%)")
    print(f"   Created date:    {has_created:3d}/{total} ({has_created/total*100:5.1f}%)")
    print(f"   Modified date:   {has_modified:3d}/{total} ({has_modified/total*100:5.1f}%)")
    
    # Categorize failures
    if failed_models:
        print("\n" + "=" * 80)
        print("❌ FAILED MODELS DETAILS")
        print("=" * 80)
        
        # Group failures by error type
        error_types = {}
        for model_name, hf_path, error in failed_models:
            # Extract error type
            if "not found" in error.lower():
                error_type = "Repository not found"
            elif "access denied" in error.lower() or "private" in error.lower():
                error_type = "Access denied (private/gated)"
            elif "connection" in error.lower():
                error_type = "Connection error"
            elif "timeout" in error.lower():
                error_type = "Timeout"
            elif "no hf_path" in error.lower():
                error_type = "No HF path configured"
            else:
                error_type = "Other error"
            
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append((model_name, hf_path, error))
        
        print(f"\n📊 Failure breakdown by type:")
        for error_type, models in sorted(error_types.items()):
            print(f"   {error_type}: {len(models)} models")
        
        print(f"\n📝 Detailed failure list:")
        for idx, (model_name, hf_path, error) in enumerate(failed_models, 1):
            print(f"\n{idx}. ❌ {model_name}")
            print(f"   Path: {hf_path}")
            print(f"   Error: {error}")
    else:
        print(f"\n🎉 All models processed successfully! No failures.")


def main():
    """Main execution function."""
    print("=" * 80)
    print("🚀 HuggingFace Metadata Updater for MODEL_CONFIGS")
    print("=" * 80)
    print(f"📁 Source: batch_infer.py")
    print(f"🔢 Total models to process: {len(MODEL_CONFIGS)}")
    
    # Set HF_TOKEN directly in script
    hf_token = "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"
    
    # Also check environment variable (can override the hardcoded one)
    env_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if env_token:
        hf_token = env_token
        print(f"🔑 Authentication: ✓ Using HuggingFace token from environment")
    else:
        print(f"🔑 Authentication: ✓ Using HuggingFace token (hardcoded)")
    
    print(f"\n⚙️  Configuration:")
    print(f"   • Delay between requests: 1.0 seconds")
    print(f"   • Retry attempts: 3 times with exponential backoff")
    print(f"   • Progress auto-save: every 10 models")
    print(f"   • Resume support: Yes (via metadata_progress.json)")
    
    print("\n" + "=" * 80)
    print("Starting metadata fetch process...")
    print("=" * 80)
    
    overall_start_time = time.time()
    
    # Initialize HuggingFace API with token if available
    api = HfApi(token=hf_token) if hf_token else HfApi()
    
    # Update all model configs with rate limiting protection
    output_file = "MODEL_CONFIGS_updated.json"
    updated_configs, failed_models = update_all_model_configs(api, request_delay=1.0, output_file=output_file)
    
    overall_duration = time.time() - overall_start_time
    
    output_path = Path(__file__).parent / output_file
    
    # Print summary
    print_summary(updated_configs, failed_models)
    
    print("\n" + "=" * 80)
    print("✅ PROCESS COMPLETE!")
    print("=" * 80)
    print(f"⏱️  Total time: {overall_duration/60:.1f} minutes ({overall_duration:.1f} seconds)")
    print(f"📄 Output JSON file: {output_path}")
    print(f"   Absolute path: {output_path.absolute()}")
    print(f"📊 Models in JSON: {len(updated_configs)}/{len(MODEL_CONFIGS)}")
    
    print(f"\n📋 Next steps:")
    print(f"   1. Review the generated JSON file:")
    print(f"      {output_path.absolute()}")
    print(f"   2. Verify the metadata looks correct")
    print(f"   3. Use this JSON file to update batch_infer.py MODEL_CONFIGS")
    
    if failed_models:
        print(f"\n⚠️  Note: {len(failed_models)} models failed and were EXCLUDED from JSON")
        print(f"   • Failed models are NOT in the output JSON file")
        print(f"   • Re-run this script to automatically retry missing models")
        print(f"   • The script will detect which models are missing and retry only those")
    else:
        print(f"\n🎉 Perfect! All models successfully fetched metadata!")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

