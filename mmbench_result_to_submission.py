#!/usr/bin/env python3
import os
import json
import pandas as pd
import glob
import re

def clean_text(text):
    """Clean text to remove illegal characters for Excel."""
    if pd.isna(text) or text is None:
        return ""
    text = str(text)
    # Remove control characters and other problematic characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    return text

def process_result_json(result_json_path, output_dir):
    """Process a single result.json file and save to Excel format."""
    # Get base name from the parent directory
    base_name = os.path.basename(os.path.dirname(result_json_path))
    output_xlsx = os.path.join(output_dir, f"result_{base_name}.xlsx")
    
    # Load result.json
    with open(result_json_path, 'r') as f:
        results = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Remove 'image' column if present
    if 'image' in df.columns:
        df = df.drop(columns=['image'])
    
    # Rename 'response' to 'prediction'
    if 'response' in df.columns:
        df = df.rename(columns={'response': 'prediction'})
    
    # Check required columns
    required = ['index', 'question', 'A', 'B', 'C', 'D', 'prediction']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"Warning: Missing columns in {base_name}: {missing}")
        return False
    
    # Clean text data to avoid Excel illegal character errors
    for col in required:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
    
    # Save to Excel
    df[required].to_excel(output_xlsx, index=False)
    print(f"Processed: {base_name} -> {output_xlsx}")
    return True

def main():
    """Process all result.json files in the sanity_check folder."""
    # Define paths
    sanity_check_dir = os.path.join("./work_dirs/sanity_check")
    
    if not os.path.exists(sanity_check_dir):
        print(f"Error: sanity_check directory not found: {sanity_check_dir}")
        return
    
    # Find all result.json files in sanity_check subdirectories
    pattern = os.path.join(sanity_check_dir, "*/result.json")
    result_files = glob.glob(pattern)
    
    if not result_files:
        print(f"No result.json files found in {sanity_check_dir}")
        return
    
    print(f"Found {len(result_files)} result.json files")
    
    # Process each file
    success_count = 0
    for result_file in result_files:
        if process_result_json(result_file, sanity_check_dir):
            success_count += 1
    
    print(f"\nCompleted: {success_count}/{len(result_files)} files processed successfully")

if __name__ == "__main__":
    main()
