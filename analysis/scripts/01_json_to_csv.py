import json
import csv
import os
import re
from pathlib import Path
from tqdm import tqdm

def get_processing_mode():
    """Ask user to select processing mode"""
    print("Select processing mode:")
    print("  1. Single folder (process one folder)")
    print("  2. Batch process (process all subfolders in a main folder)")
    
    while True:
        choice = input("\nEnter your choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            return int(choice)
        print("❌ Invalid choice. Please enter 1 or 2.")

def get_folder_path():
    """Get folder path from user"""
    folder_path = input("Enter the folder path containing JSON files: ").strip()
    # Remove quotes if user copies path with quotes
    folder_path = folder_path.strip('"').strip("'")
    
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder does not exist: {folder_path}")
    return folder_path

def get_main_folder_path():
    """Get main folder path for batch processing"""
    folder_path = input("Enter the main folder path containing subfolders: ").strip()
    # Remove quotes if user copies path with quotes
    folder_path = folder_path.strip('"').strip("'")
    
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder does not exist: {folder_path}")
    return folder_path

def find_subfolders_with_json(main_folder):
    """Find all subfolders containing JSON files"""
    main_path = Path(main_folder)
    subfolders = []
    
    print("\n🔍 Scanning for subfolders with JSON files...")
    
    # Check all subdirectories
    for item in main_path.iterdir():
        if item.is_dir():
            # Check if this folder has JSON files
            json_files = list(item.glob('*.json'))
            if json_files:
                subfolders.append(item)
                print(f"  ✓ Found: {item.name} ({len(json_files):,} JSON files)")
    
    return subfolders

def extract_base_name(json_files):
    """Extract base name from JSON files (removing timestamp parts)"""
    if not json_files:
        return "output"
    
    # Get first filename
    first_file = json_files[0].stem  # filename without extension
    
    # Try multiple timestamp patterns (from most specific to least specific)
    patterns = [
        r'_\d{8}_\d{6}_\d{6}$',      # _YYYYMMDD_HHMMSS_microseconds
        r'_\d{8}_\d{6}_\d+$',        # _YYYYMMDD_HHMMSS_anydigits
        r'_\d{8}_\d{6}$',            # _YYYYMMDD_HHMMSS
        r'_\d{14}_\d+$',             # _YYYYMMDDHHmmss_microseconds
        r'_\d{14}$',                 # _YYYYMMDDHHmmss
        r'_\d+_\d+_\d+$',            # _numbers_numbers_numbers (generic)
        r'_\d+_\d+$',                # _numbers_numbers (generic)
        r'_\d+$',                    # _numbers (generic timestamp)
    ]
    
    base_name = first_file
    for pattern in patterns:
        result = re.sub(pattern, '', first_file)
        if result != first_file:
            base_name = result
            break
    
    # If still no match, try to find common prefix among multiple files
    if base_name == first_file and len(json_files) > 1:
        names = [f.stem for f in json_files[:min(10, len(json_files))]]
        common = os.path.commonprefix(names).rstrip('_')
        if common:
            base_name = common
    
    return base_name if base_name else "output"

def find_max_predictions(json_files, sample_size=1000):
    """Quickly determine max prediction length by sampling files"""
    max_n = 0
    sample_files = json_files[::max(1, len(json_files)//sample_size)][:sample_size]
    
    for json_file in tqdm(sample_files, desc="  Scanning structure", ncols=80, leave=False):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                n = max(
                    len(data.get('actual_data', [])), 
                    len(data.get('prediction_results', []))
                )
                max_n = max(max_n, n)
        except:
            continue
    
    return max_n

def generate_headers(max_n):
    """Generate CSV headers with last_values first"""
    # Start with last_value columns
    headers = [
        'last_value_open',
        'last_value_high',
        'last_value_low',
        'last_value_close',
        'last_value_volume'
    ]
    
    # Add actual and predicted columns for each prediction step
    for i in range(1, max_n + 1):
        headers.extend([
            f'actual_open_{i}', f'predicted_open_{i}',
            f'actual_high_{i}', f'predicted_high_{i}',
            f'actual_low_{i}', f'predicted_low_{i}',
            f'actual_close_{i}', f'predicted_close_{i}',
            f'actual_volume_{i}', f'predicted_volume_{i}'
        ])
    return headers

def process_json_file(json_path, max_n):
    """Extract and format data from a single JSON file"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract last_values
        last_values = data.get('input_data_summary', {}).get('last_values', {})
        
        # Start row with last_values
        row = [
            last_values.get('open', ''),
            last_values.get('high', ''),
            last_values.get('low', ''),
            last_values.get('close', ''),
            last_values.get('volume', '')
        ]
        
        # Get actual and predicted data
        actual_data = data.get('actual_data', [])
        prediction_data = data.get('prediction_results', [])
        
        # Add actual and predicted data for each prediction step
        for i in range(max_n):
            if i < len(actual_data) and i < len(prediction_data):
                # Both actual and predicted exist
                row.extend([
                    actual_data[i].get('open', ''),
                    prediction_data[i].get('open', ''),
                    actual_data[i].get('high', ''),
                    prediction_data[i].get('high', ''),
                    actual_data[i].get('low', ''),
                    prediction_data[i].get('low', ''),
                    actual_data[i].get('close', ''),
                    prediction_data[i].get('close', ''),
                    actual_data[i].get('volume', ''),
                    prediction_data[i].get('volume', '')
                ])
            elif i < len(actual_data):
                # Only actual data exists
                row.extend([
                    actual_data[i].get('open', ''), '',
                    actual_data[i].get('high', ''), '',
                    actual_data[i].get('low', ''), '',
                    actual_data[i].get('close', ''), '',
                    actual_data[i].get('volume', ''), ''
                ])
            elif i < len(prediction_data):
                # Only predicted data exists
                row.extend([
                    '', prediction_data[i].get('open', ''),
                    '', prediction_data[i].get('high', ''),
                    '', prediction_data[i].get('low', ''),
                    '', prediction_data[i].get('close', ''),
                    '', prediction_data[i].get('volume', '')
                ])
            else:
                # Neither exists (padding)
                row.extend([''] * 10)
        
        return row
    
    except Exception as e:
        # Return empty row on error (5 last_value columns + max_n * 10)
        return [''] * (5 + max_n * 10)

def process_folder(folder_path, output_dir=None, folder_name=None):
    """Process a single folder containing JSON files"""
    
    if folder_name:
        print(f"\n{'='*70}")
        print(f"Processing folder: {folder_name}")
        print(f"{'='*70}")
    
    # Get all JSON files sorted by name
    json_files = sorted(Path(folder_path).glob('*.json'))
    
    if not json_files:
        print(f"  ⚠ No JSON files found in {folder_path}")
        return False
    
    print(f"  ✓ Found {len(json_files):,} JSON files")
    
    # Extract base name from filenames
    base_name = extract_base_name(json_files)
    print(f"  ✓ Detected base name: {base_name}")
    
    # Determine max prediction length
    print("  📊 Determining data structure...")
    max_n = find_max_predictions(json_files)
    print(f"  ✓ Max prediction length: {max_n}")
    
    # Generate headers
    headers = generate_headers(max_n)
    
    # Create output directory
    if output_dir is None:
        output_dir = Path('analysis') / 'data'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output CSV path
    output_csv = output_dir / f'{base_name}.csv'
    
    # Process files and write to CSV
    print("  🔄 Converting JSON files to CSV...")
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for json_file in tqdm(json_files, desc="  Processing", ncols=80, unit="files", leave=False):
            row = process_json_file(json_file, max_n)
            writer.writerow(row)
    
    print(f"  ✅ Complete! Output: {output_csv}\n")
    
    return True

def process_single_folder():
    """Process a single folder"""
    folder_path = get_folder_path()
    
    print("\nFinding JSON files...")
    json_files = sorted(Path(folder_path).glob('*.json'))
    
    if not json_files:
        print("❌ No JSON files found in the folder!")
        return
    
    print(f"✓ Found {len(json_files):,} JSON files\n")
    
    # Extract base name from filenames
    base_name = extract_base_name(json_files)
    print(f"✓ Detected base name: {base_name}")
    print(f"  (from: {json_files[0].name})\n")
    
    # Determine max prediction length
    print("Step 1: Determining data structure...")
    max_n = find_max_predictions(json_files)
    print(f"✓ Max prediction length: {max_n}\n")
    
    # Generate headers
    headers = generate_headers(max_n)
    
    # Create output directory
    output_dir = Path('analysis') / 'data'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output CSV path
    output_csv = output_dir / f'{base_name}.csv'
    
    # Process files and write to CSV
    print("Step 2: Converting JSON files to CSV...")
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for json_file in tqdm(json_files, desc="Processing", ncols=80, unit="files"):
            row = process_json_file(json_file, max_n)
            writer.writerow(row)
    
    print(f"\n{'='*60}")
    print(f"✓ Conversion complete!")
    print(f"✓ Total files processed: {len(json_files):,}")
    print(f"✓ Output file: {output_csv}")
    print(f"{'='*60}\n")

def process_batch_folders():
    """Process multiple folders in batch mode"""
    main_folder = get_main_folder_path()
    
    # Find all subfolders with JSON files
    subfolders = find_subfolders_with_json(main_folder)
    
    if not subfolders:
        print("\n❌ No subfolders with JSON files found!")
        return
    
    print(f"\n📦 Found {len(subfolders)} folder(s) to process")
    
    # Confirm processing
    confirm = input(f"\nProcess all {len(subfolders)} folder(s)? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Batch processing cancelled.")
        return
    
    # Create output directory
    output_dir = Path('analysis') / 'data'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each subfolder
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING - {len(subfolders)} FOLDERS")
    print(f"{'='*70}\n")
    
    successful = 0
    failed = 0
    
    for i, subfolder in enumerate(subfolders, 1):
        print(f"[{i}/{len(subfolders)}] ", end="")
        
        if process_folder(subfolder, output_dir, subfolder.name):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Successful: {successful}")
    if failed > 0:
        print(f"❌ Failed: {failed}")
    print(f"📁 Output directory: {output_dir}")
    print(f"{'='*70}\n")

def main():
    print("╔═══════════════════════════════════════╗")
    print("║   JSON to CSV Converter (Fast Mode)  ║")
    print("╚═══════════════════════════════════════╝\n")
    
    try:
        # Get processing mode
        mode = get_processing_mode()
        
        if mode == 1:
            # Single folder processing
            process_single_folder()
        else:
            # Batch folder processing
            process_batch_folders()
    
    except KeyboardInterrupt:
        print("\n\n⚠ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()