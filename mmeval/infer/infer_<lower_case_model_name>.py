import os
import re
import sys
import json
import argparse
from tqdm import tqdm
from PIL import Image


def parse_media_placeholders(prompt: str) -> tuple[list[str], str]:
    """
    Extract media file paths from placeholders and clean prompt for model processing.
    
    Args:
        prompt: Input prompt containing media placeholders in format <type-placeholder: filename>
        
    Returns:
        tuple: (media_paths, cleaned_prompt) where media_paths is list of filenames 
               and cleaned_prompt has placeholder content removed
    """
    media_paths = re.findall(r'<[a-zA-Z]{5}-placeholder: (.*?)>', prompt)
    
    # Remove placeholder content, keep placeholder markers for model processing
    cleaned_prompt = re.sub(r'<(image|video)-placeholder: (.*?)>', r'<\1-placeholder>', prompt)
    
    return media_paths, cleaned_prompt


def load_media_files(media_paths: list[str], media_dir: str):
    """
    Load media files from filesystem based on model requirements.
    
    Args:
        media_paths: List of media filenames to load
        media_dir: Directory path containing media files
        
    Returns:
        list: File paths (str) or loaded media objects depending on implementation
    """
    
    media_files = []
    
    for filename in media_paths:
        filepath = os.path.join(media_dir, filename)
        
        if os.path.exists(filepath):
            # TODO: Choose one implementation option below:
            
            # === Option 1: Return file paths (for models accepting file paths) ===
            # media_files.append(filepath)
            
            # === Option 2: Load PIL Images (for models requiring image objects) ===
            # try:
            #     image = Image.open(filepath).convert("RGB")
            #     media_files.append(image)
            #     print(f"Loaded image from {filepath}")
            # except Exception as e:
            #     print(f"Error loading image {filepath}: {e}")
            #     # Optionally continue or raise exception based on your needs
            
            pass  # Remove this line after uncommenting an option above
        else:
            print(f"Warning: Media file not found: {filepath}")
    
    return media_files


def load_cached_results(cache_file: str) -> dict:
    """
    Load previously cached inference results from file.
    
    Args:
        cache_file: Path to cache file containing previous results
        
    Returns:
        dict: Mapping of sample IDs to cached outputs, empty dict if file not found
    """
    if not os.path.exists(cache_file):
        return {}
    
    try:
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
        return {item["id"]: item["output"] for item in cached_data if "output" in item}
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Invalid cache file format: {e}")
        return {}


def validate_and_clean_results(results: list, original_samples: list) -> tuple[list, bool]:
    """
    Validate result completeness and remove invalid entries.
    
    Args:
        results: List of inference results to validate
        original_samples: List of original input samples for comparison
        
    Returns:
        tuple: (cleaned_results, is_complete) where cleaned_results contains only 
               valid entries and is_complete indicates if all samples processed
    """
    # Get original IDs
    original_ids = {sample["id"] for sample in original_samples}
    
    # Clean results: remove entries without output
    valid_results = []
    invalid_ids = []
    
    for result in results:
        if "output" in result and result["output"]:
            valid_results.append(result)
        else:
            invalid_ids.append(result["id"])
    
    # Check for missing IDs
    result_ids = {result["id"] for result in valid_results}
    missing_ids = original_ids - result_ids
    
    # Report issues
    if invalid_ids or missing_ids:
        print(f"\n=== Data Validation Issues ===")
        if invalid_ids:
            print(f"Removed {len(invalid_ids)} entries without output:")
            print(f"IDs: {invalid_ids}")
        if missing_ids:
            print(f"Missing {len(missing_ids)} entries from results:")
            print(f"IDs: {sorted(missing_ids)}")
        print(f"Total missing: {len(invalid_ids) + len(missing_ids)}")
        print("===============================\n")
        
        return valid_results, False
    
    return valid_results, True


def save_results(results: list, original_samples: list, output_file: str, cache_file: str):
    """
    Save validated results to output file and manage cache cleanup.
    
    Args:
        results: List of inference results to save
        original_samples: List of original input samples for validation
        output_file: Path for final output JSON file
        cache_file: Path for temporary cache file
        
    Returns:
        bool: True if all results complete and saved successfully, False otherwise
    """
    # Validate and clean results
    clean_results, is_complete = validate_and_clean_results(results, original_samples)
    
    # Always save clean results to cache
    with open(cache_file, 'w') as f:
        json.dump(clean_results, f, indent=2, ensure_ascii=False)
    
    if is_complete:
        # Only save final JSON if all data is complete
        with open(output_file, 'w') as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
        
        # Clean up cache file after successful completion
        if os.path.exists(cache_file):
            os.remove(cache_file)
        
        print(f"✅ All {len(clean_results)} samples processed successfully")
        print(f"✅ Results saved to {output_file}")
        print(f"✅ Cache cleaned up")
        return True
    else:
        print(f"⚠️ Data incomplete - JSON file not saved")
        print(f"⚠️ Clean cache updated at {cache_file} ({len(clean_results)} valid entries)")
        return False


def main():
    """
    Main inference pipeline for MLLM evaluation.
    
    Processes input samples through multimodal language model, handles caching,
    and saves results with validation. Supports resumable execution via cache.
    """
    parser = argparse.ArgumentParser(description='MLLM Inference Script')
    parser.add_argument('--input', required=True, help='Input JSON file path')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--media_dir', required=True, help='Media files directory')
    parser.add_argument('--model_path', default="model_path", help='Model path')
    parser.add_argument('--max_tokens', type=int, default=512, help='Max output tokens')
    parser.add_argument('--temperature', type=float, default=0.0, help='Generation temperature')
    parser.add_argument('--use_cache', action='store_true', help='Use existing cached results')
    
    args = parser.parse_args()
    
    # Load input data
    try:
        with open(args.input, 'r') as f:
            samples = json.load(f)
        print(f"Loaded {len(samples)} samples from {args.input}")
    except Exception as e:
        print(f"Error loading input file: {e}")
        sys.exit(1)
    
    # Initialize model and processor
    try:
        # TODO: Initialize your model here
        # model = AutoModelForCausalLM.from_pretrained(args.model_path)
        # processor = AutoProcessor.from_pretrained(args.model_path)
        print(f"Model loaded from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Setup caching - only load if use_cache is True
    cache_file = f"{args.output}.cache"
    cached_outputs = load_cached_results(cache_file) if args.use_cache else {}
    
    if args.use_cache and cached_outputs:
        print(f"Loaded {len(cached_outputs)} cached results")
    
    results = []
    
    for sample in tqdm(samples, desc="Processing"):
        sample_id = sample["id"]
        
        # Use cached result if available and use_cache is enabled
        if args.use_cache and sample_id in cached_outputs:
            sample["output"] = cached_outputs[sample_id]
            results.append(sample)
            continue
        
        # Parse prompt and media
        media_paths, cleaned_prompt = parse_media_placeholders(sample["prompt"])
        media_files = load_media_files(media_paths, args.media_dir)
        
        # Model inference
        try:
            # TODO: Implement model inference
            # Example inference structure:
            # inputs = processor(text=cleaned_prompt, images=media_files, return_tensors="pt")
            # outputs = model.generate(**inputs, max_new_tokens=args.max_tokens, temperature=args.temperature)
            # output_text = processor.decode(outputs[0], skip_special_tokens=True)
            pass
            
        except Exception as e:
            print(f"Error during inference for sample {sample_id}: {e}")
            output_text = ""
        
        sample["output"] = output_text
        results.append(sample)
        
        # Always save progress to cache (regardless of use_cache flag)
        with open(cache_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save final results with validation
    success = save_results(results, samples, args.output, cache_file)
    
    if success:
        print("Inference completed successfully!")
    else:
        print("Inference completed with issues. Resume with --use_cache to continue.")
        sys.exit(1)


if __name__ == "__main__":
    main()