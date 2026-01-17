import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
import sys

def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")

def print_subheader(text):
    """Print formatted subheader"""
    print(f"\n{text}")
    print(f"{'-'*70}")

def print_metric(label, value, status=""):
    """Print formatted metric"""
    if status:
        print(f"  {label:<45} {value:>15} {status:>8}")
    else:
        print(f"  {label:<45} {value:>20}")

def get_processing_mode():
    """Ask user to select processing mode"""
    print("\nSelect validation mode:")
    print("  1. Single file (validate one CSV against one JSON folder)")
    print("  2. Batch validate (validate multiple CSVs against JSON folders)")
    
    while True:
        choice = input("\nEnter your choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            return int(choice)
        print("❌ Invalid choice. Please enter 1 or 2.")

def get_csv_path():
    """Get CSV file path from user"""
    csv_path = input("\nEnter the CSV file path (or press Enter for 'analysis/data/*.csv'): ").strip()
    csv_path = csv_path.strip('"').strip("'")
    
    if not csv_path:
        # Look for CSV in default location
        data_dir = Path('analysis/data')
        if data_dir.exists():
            csv_files = list(data_dir.glob('*.csv'))
            if len(csv_files) == 1:
                return csv_files[0]
            elif len(csv_files) > 1:
                print("\nMultiple CSV files found:")
                for i, f in enumerate(csv_files, 1):
                    print(f"  {i}. {f.name}")
                choice = int(input("\nSelect file number: ")) - 1
                return csv_files[choice]
            else:
                raise ValueError("No CSV files found in analysis/data/")
        else:
            raise ValueError("analysis/data/ directory not found")
    
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise ValueError(f"File does not exist: {csv_path}")
    
    return csv_path

def get_csv_folder():
    """Get folder containing CSV files for batch processing"""
    folder_path = input("\nEnter the folder path containing CSV files (or press Enter for 'analysis/data'): ").strip()
    folder_path = folder_path.strip('"').strip("'")
    
    if not folder_path:
        folder_path = 'analysis/data'
    
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    return folder_path

def get_json_folder():
    """Get JSON folder path from user"""
    folder_path = input("\nEnter the JSON source folder path: ").strip()
    folder_path = folder_path.strip('"').strip("'")
    
    if not folder_path:
        raise ValueError("JSON folder path is required")
    
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    return folder_path

def get_json_main_folder():
    """Get main folder containing JSON subfolders for batch processing"""
    folder_path = input("\nEnter the main folder path containing JSON subfolders: ").strip()
    folder_path = folder_path.strip('"').strip("'")
    
    if not folder_path:
        raise ValueError("JSON main folder path is required")
    
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    return folder_path

def match_csv_to_json_folder(csv_files, json_folders):
    """Match CSV files to their corresponding JSON folders by name"""
    matches = []
    unmatched_csv = []
    unmatched_json = []
    
    # Create a dict of json folders by their base name
    json_dict = {folder.name: folder for folder in json_folders}
    
    for csv_file in csv_files:
        csv_base = csv_file.stem  # filename without .csv
        
        # Try to find matching JSON folder
        matched = False
        for json_name, json_folder in json_dict.items():
            # Check if CSV name matches JSON folder name
            if csv_base == json_name or csv_base in json_name or json_name in csv_base:
                matches.append((csv_file, json_folder))
                json_dict.pop(json_name)  # Remove from dict to avoid duplicate matching
                matched = True
                break
        
        if not matched:
            unmatched_csv.append(csv_file)
    
    # Remaining JSON folders are unmatched
    unmatched_json = list(json_dict.values())
    
    return matches, unmatched_csv, unmatched_json

def analyze_json_folder(json_folder):
    """Analyze JSON folder to get expected structure"""
    json_files = sorted(json_folder.glob('*.json'))
    
    if not json_files:
        raise ValueError(f"No JSON files found in {json_folder}")
    
    total_json_files = len(json_files)
    max_n_actual = 0
    max_n_predicted = 0
    
    # Sample files to determine max n
    sample_size = min(1000, total_json_files)
    sample_files = json_files[::max(1, total_json_files//sample_size)][:sample_size]
    
    for json_file in sample_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                actual_len = len(data.get('actual_data', []))
                predicted_len = len(data.get('prediction_results', []))
                max_n_actual = max(max_n_actual, actual_len)
                max_n_predicted = max(max_n_predicted, predicted_len)
        except Exception as e:
            continue
    
    max_n = max(max_n_actual, max_n_predicted)
    
    return {
        'total_files': total_json_files,
        'max_n_actual': max_n_actual,
        'max_n_predicted': max_n_predicted,
        'max_n': max_n,
        'expected_columns': max_n * 10 + 5  # 5 last_value columns + (5 fields * 2 (actual + predicted) * n)
    }

def analyze_csv_quality(csv_path, json_info, show_full_report=True):
    """Perform comprehensive quality analysis on CSV file"""
    
    if show_full_report:
        print_header("CSV QUALITY ANALYSIS REPORT")
        print(f"\n📄 File: {csv_path}")
        print(f"📊 File Size: {csv_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Load CSV
    if show_full_report:
        print("\n⏳ Loading CSV file...")
    df = pd.read_csv(csv_path)
    
    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols
    
    # Parse column structure
    column_info = parse_column_structure(df.columns)
    
    # STRUCTURE VALIDATION
    if show_full_report:
        print_header("STRUCTURE VALIDATION (CSV vs JSON Source)")
    
    rows_match = total_rows == json_info['total_files']
    cols_match = total_cols == json_info['expected_columns']
    max_n_match = column_info['max_n'] == json_info['max_n']
    has_last_values = column_info['has_last_values']
    
    if show_full_report:
        print_metric("Expected JSON Files:", f"{json_info['total_files']:,}")
        print_metric("Actual CSV Rows:", f"{total_rows:,}", "✓" if rows_match else "❌")
        
        if not rows_match:
            diff = total_rows - json_info['total_files']
            print_metric("  → Difference:", f"{diff:+,}", "❌")
        
        print()
        print_metric("Expected Max Prediction Steps (n):", f"{json_info['max_n']:,}")
        print_metric("Actual Max Prediction Steps (n):", f"{column_info['max_n']:,}", "✓" if max_n_match else "❌")
        
        if not max_n_match:
            diff = column_info['max_n'] - json_info['max_n']
            print_metric("  → Difference:", f"{diff:+,}", "❌")
        
        print()
        print_metric("Expected Columns (with last_values):", f"{json_info['expected_columns']:,}")
        print_metric("Actual Columns:", f"{total_cols:,}", "✓" if cols_match else "❌")
        
        if not cols_match:
            diff = total_cols - json_info['expected_columns']
            print_metric("  → Difference:", f"{diff:+,}", "❌")
        
        print()
        print_metric("Has last_value columns:", "Yes" if has_last_values else "No", "✓" if has_last_values else "⚠")
        print_metric("Last_value columns found:", f"{column_info['last_value_count']}")
        print_metric("Max Actual Data Steps (from JSON):", f"{json_info['max_n_actual']:,}")
        print_metric("Max Predicted Data Steps (from JSON):", f"{json_info['max_n_predicted']:,}")
    
    # Overall structure validation
    structure_valid = rows_match and cols_match and max_n_match and has_last_values
    
    if show_full_report:
        print_subheader("Structure Validation Result:")
        if structure_valid:
            print("  ✓ PASSED - CSV structure matches JSON source perfectly!")
        else:
            print("  ❌ FAILED - CSV structure does not match JSON source!")
            if not rows_match:
                print("     • Row count mismatch")
            if not max_n_match:
                print("     • Prediction steps (n) mismatch")
            if not cols_match:
                print("     • Column count mismatch")
            if not has_last_values:
                print("     • Missing last_value columns")
        
        print_header("BASIC STATISTICS")
        print_metric("Total Rows:", f"{total_rows:,}")
        print_metric("Total Columns:", f"{total_cols:,}")
        print_metric("Total Cells:", f"{total_cells:,}")
        print_metric("Prediction Steps (n):", f"{column_info['max_n']:,}")
        print_metric("Last Value Columns:", f"{column_info['last_value_count']}")
    
    # LAST VALUE COLUMNS ANALYSIS
    last_value_cols = column_info['last_value_cols']
    
    if show_full_report and last_value_cols:
        print_header("LAST VALUE COLUMNS ANALYSIS")
        
        last_value_missing = (df[last_value_cols].isna() | (df[last_value_cols] == '')).sum().sum()
        last_value_total = len(last_value_cols) * total_rows
        last_value_pct = (last_value_missing / last_value_total * 100) if last_value_total > 0 else 0
        
        print_metric("Last Value Columns Found:", f"{len(last_value_cols)}")
        print_metric("Total Last Value Cells:", f"{last_value_total:,}")
        print_metric("Missing Last Value Data:", f"{last_value_missing:,}", get_status(last_value_missing))
        print_metric("Missing Percentage:", f"{last_value_pct:.2f}%", get_status_percentage(last_value_pct))
        
        print_subheader("Missing Data by Last Value Field:")
        print(f"  {'Field':<30} {'Missing':>12} {'Percentage':>12}")
        print(f"  {'-'*56}")
        
        for col in last_value_cols:
            missing = (df[col].isna() | (df[col] == '')).sum()
            pct = (missing / total_rows) * 100
            print(f"  {col:<30} {missing:>12,} {pct:>11.2f}%")
    elif show_full_report and not last_value_cols:
        print_header("LAST VALUE COLUMNS ANALYSIS")
        print("\n  ⚠ No last_value columns found in CSV!")
    
    # Missing/Empty value analysis
    nan_count = df.isna().sum().sum()
    empty_string_count = (df == '').sum().sum()
    total_missing = nan_count + empty_string_count
    missing_percentage = (total_missing / total_cells) * 100
    
    if show_full_report:
        print_header("MISSING DATA ANALYSIS")
        
        print_metric("NaN Values:", f"{nan_count:,}", get_status(nan_count))
        print_metric("Empty String Values:", f"{empty_string_count:,}", get_status(empty_string_count))
        print_metric("Total Missing:", f"{total_missing:,}", get_status(total_missing))
        print_metric("Missing Percentage:", f"{missing_percentage:.2f}%", get_status_percentage(missing_percentage))
        print_metric("Valid Data Cells:", f"{total_cells - total_missing:,}", "✓")
    
    # Row-wise analysis
    rows_with_missing = (df.isna() | (df == '')).any(axis=1).sum()
    rows_completely_empty = (df.isna() | (df == '')).all(axis=1).sum()
    rows_perfect = total_rows - rows_with_missing
    
    if show_full_report:
        print_header("ROW-WISE ANALYSIS")
        
        print_metric("Rows with Perfect Data:", f"{rows_perfect:,}", get_status(total_rows - rows_perfect))
        print_metric("Rows with Some Missing:", f"{rows_with_missing:,}", get_status(rows_with_missing))
        print_metric("Rows Completely Empty:", f"{rows_completely_empty:,}", get_status(rows_completely_empty))
        
        # Calculate completeness per row
        row_completeness = ((~(df.isna() | (df == ''))).sum(axis=1) / total_cols * 100)
        
        print_subheader("Row Completeness Distribution:")
        print_metric("  Minimum Completeness:", f"{row_completeness.min():.2f}%")
        print_metric("  Average Completeness:", f"{row_completeness.mean():.2f}%")
        print_metric("  Maximum Completeness:", f"{row_completeness.max():.2f}%")
        print_metric("  Median Completeness:", f"{row_completeness.median():.2f}%")
    
    # Column-wise analysis
    if show_full_report:
        print_header("COLUMN-WISE ANALYSIS (TOP 20 PROBLEMATIC)")
        
        col_missing = (df.isna() | (df == '')).sum()
        col_missing_sorted = col_missing.sort_values(ascending=False)
        
        problematic_cols = col_missing_sorted[col_missing_sorted > 0]
        
        if len(problematic_cols) > 0:
            print(f"\n  {'Column Name':<35} {'Missing':>12} {'Percentage':>12}")
            print(f"  {'-'*60}")
            for col, missing in list(problematic_cols.items())[:20]:
                pct = (missing / total_rows) * 100
                print(f"  {col:<35} {missing:>12,} {pct:>11.2f}%")
            
            if len(problematic_cols) > 20:
                print(f"\n  ... and {len(problematic_cols) - 20} more columns with missing data")
        else:
            print("\n  ✓ No columns with missing data!")
    
    # Actual vs Predicted comparison
    actual_cols = [col for col in df.columns if col.startswith('actual_')]
    predicted_cols = [col for col in df.columns if col.startswith('predicted_')]
    
    actual_missing = (df[actual_cols].isna() | (df[actual_cols] == '')).sum().sum()
    predicted_missing = (df[predicted_cols].isna() | (df[predicted_cols] == '')).sum().sum()
    
    actual_total = len(actual_cols) * total_rows
    predicted_total = len(predicted_cols) * total_rows
    
    if show_full_report:
        print_header("ACTUAL vs PREDICTED DATA COMPARISON")
        
        print_metric("Actual Data Missing:", f"{actual_missing:,} / {actual_total:,}", 
                     f"({actual_missing/actual_total*100:.2f}%)")
        print_metric("Predicted Data Missing:", f"{predicted_missing:,} / {predicted_total:,}", 
                     f"({predicted_missing/predicted_total*100:.2f}%)")
        
        # Field type analysis
        print_header("FIELD-WISE ANALYSIS")
        
        field_types = ['open', 'high', 'low', 'close', 'volume']
        
        print(f"\n  {'Field':<12} {'Actual Missing':>18} {'Predicted Missing':>20}")
        print(f"  {'-'*52}")
        
        for field in field_types:
            actual_field_cols = [col for col in actual_cols if f'_{field}_' in col]
            predicted_field_cols = [col for col in predicted_cols if f'_{field}_' in col]
            
            actual_field_missing = (df[actual_field_cols].isna() | (df[actual_field_cols] == '')).sum().sum()
            predicted_field_missing = (df[predicted_field_cols].isna() | (df[predicted_field_cols] == '')).sum().sum()
            
            print(f"  {field.capitalize():<12} {actual_field_missing:>18,} {predicted_field_missing:>20,}")
        
        # Step-wise analysis
        print_header("PREDICTION STEP ANALYSIS (First 10 & Last 10 steps)")
        
        step_analysis = analyze_by_steps(df, column_info['max_n'])
        
        print(f"\n  {'Step':>6} {'Actual Missing':>18} {'Predicted Missing':>20} {'Total %':>10}")
        print(f"  {'-'*56}")
        
        # Show first 10 steps
        for step in range(1, min(11, column_info['max_n'] + 1)):
            if step in step_analysis:
                data = step_analysis[step]
                total_pct = (data['actual_missing'] + data['predicted_missing']) / (data['actual_total'] + data['predicted_total']) * 100
                print(f"  {step:>6} {data['actual_missing']:>18,} {data['predicted_missing']:>20,} {total_pct:>9.2f}%")
        
        # Show last 10 steps if there are more than 20 total
        if column_info['max_n'] > 20:
            print(f"  {'...':^6}")
            for step in range(column_info['max_n'] - 9, column_info['max_n'] + 1):
                if step in step_analysis:
                    data = step_analysis[step]
                    total_pct = (data['actual_missing'] + data['predicted_missing']) / (data['actual_total'] + data['predicted_total']) * 100
                    print(f"  {step:>6} {data['actual_missing']:>18,} {data['predicted_missing']:>20,} {total_pct:>9.2f}%")
    
    # Data type validation
    numeric_cols = last_value_cols + actual_cols + predicted_cols
    non_numeric_issues = 0
    invalid_columns = []
    
    for col in numeric_cols:
        non_empty = df[col][(df[col] != '') & (~df[col].isna())]
        if len(non_empty) > 0:
            try:
                pd.to_numeric(non_empty, errors='raise')
            except:
                non_numeric = pd.to_numeric(non_empty, errors='coerce').isna().sum() #type: ignore
                if non_numeric > 0:
                    non_numeric_issues += 1
                    invalid_columns.append((col, non_numeric))
    
    if show_full_report:
        print_header("DATA TYPE VALIDATION")
        
        if non_numeric_issues == 0:
            print("\n  ✓ All non-empty values are valid numeric data!")
            print("    (Including last_value, actual, and predicted columns)")
        else:
            print(f"\n  ⚠ Found {non_numeric_issues} columns with non-numeric values")
            if invalid_columns:
                print(f"\n  {'Column':<35} {'Invalid Values':>15}")
                print(f"  {'-'*52}")
                for col, count in invalid_columns[:10]:
                    print(f"  {col:<35} {count:>15,}")
                if len(invalid_columns) > 10:
                    print(f"  ... and {len(invalid_columns) - 10} more")
    
    # Calculate quality score
    quality_score = calculate_quality_score(missing_percentage, rows_perfect, total_rows, structure_valid)
    
    if show_full_report:
        # Summary and recommendations
        print_header("QUALITY SUMMARY & RECOMMENDATIONS")
        
        print(f"\n  Overall Quality Score: {quality_score:.1f}/100")
        print_quality_bar(quality_score)
        
        print("\n  Data Quality Assessment:")
        
        if missing_percentage < 1:
            print("  ✓ Excellent data quality! Very low missing data.")
        elif missing_percentage < 5:
            print("  ✓ Good data quality. Minor missing data present.")
        elif missing_percentage < 20:
            print("  ⚠ Moderate data quality. Consider investigating missing data patterns.")
        else:
            print("  ❌ Poor data quality. Significant missing data - investigation needed.")
        
        print("\n  Structure Assessment:")
        if structure_valid:
            print("  ✓ Perfect structure match with JSON source!")
        else:
            print("  ❌ Structure mismatch detected - review conversion process!")
        
        print("\n  Recommendations:")
        
        if not has_last_values:
            print("  ❌ Missing last_value columns - CSV may be from old conversion format")
        
        if last_value_cols:
            last_value_missing = (df[last_value_cols].isna() | (df[last_value_cols] == '')).sum().sum()
            if last_value_missing > 0:
                print(f"  ⚠ {last_value_missing} missing values in last_value columns - check JSON source data")
        
        if rows_completely_empty > 0:
            print(f"  ⚠ {rows_completely_empty} completely empty rows detected - consider removal")
        
        col_missing = (df.isna() | (df == '')).sum()
        problematic_cols = col_missing[col_missing > 0]
        if len(problematic_cols) > 0 and problematic_cols.max() > total_rows * 0.5:
            print(f"  ⚠ Some columns missing >50% of data - may need to be excluded from analysis")
        
        if rows_perfect == total_rows:
            print("  ✓ Perfect! All rows have complete data.")
        
        if not rows_match:
            print(f"  ❌ CSV has {abs(total_rows - json_info['total_files'])} {'more' if total_rows > json_info['total_files'] else 'fewer'} rows than JSON files")
        
        if non_numeric_issues > 0:
            print(f"  ⚠ {non_numeric_issues} columns contain non-numeric data - verify data types")
        
        print_header("ANALYSIS COMPLETE")
    
    # Return summary for batch processing
    return {
        'structure_valid': structure_valid,
        'quality_score': quality_score,
        'missing_percentage': missing_percentage,
        'rows': total_rows,
        'cols': total_cols,
        'rows_match': rows_match,
        'cols_match': cols_match,
        'has_last_values': has_last_values
    }

def parse_column_structure(columns):
    """Parse column structure to determine max prediction steps and identify column types"""
    max_n = 0
    last_value_cols = []
    
    for col in columns:
        if col.startswith('last_value_'):
            last_value_cols.append(col)
        elif '_' in col:
            parts = col.split('_')
            if parts[-1].isdigit():
                max_n = max(max_n, int(parts[-1]))
    
    return {
        'max_n': max_n,
        'last_value_cols': last_value_cols,
        'last_value_count': len(last_value_cols),
        'has_last_values': len(last_value_cols) > 0
    }

def analyze_by_steps(df, max_n):
    """Analyze missing data by prediction step"""
    step_analysis = {}
    
    for step in range(1, max_n + 1):
        actual_cols = [col for col in df.columns if col.startswith('actual_') and col.endswith(f'_{step}')]
        predicted_cols = [col for col in df.columns if col.startswith('predicted_') and col.endswith(f'_{step}')]
        
        actual_missing = (df[actual_cols].isna() | (df[actual_cols] == '')).sum().sum() if actual_cols else 0
        predicted_missing = (df[predicted_cols].isna() | (df[predicted_cols] == '')).sum().sum() if predicted_cols else 0
        
        step_analysis[step] = {
            'actual_missing': actual_missing,
            'predicted_missing': predicted_missing,
            'actual_total': len(actual_cols) * len(df),
            'predicted_total': len(predicted_cols) * len(df)
        }
    
    return step_analysis

def get_status(value):
    """Get status indicator based on value"""
    if value == 0:
        return "✓"
    elif value < 100:
        return "⚠"
    else:
        return "❌"

def get_status_percentage(percentage):
    """Get status indicator based on percentage"""
    if percentage == 0:
        return "✓"
    elif percentage < 5:
        return "⚠"
    else:
        return "❌"

def calculate_quality_score(missing_pct, perfect_rows, total_rows, structure_valid):
    """Calculate overall quality score (0-100)"""
    completeness_score = (100 - missing_pct) * 0.5  # 50% weight
    row_quality_score = (perfect_rows / total_rows * 100) * 0.3  # 30% weight
    structure_score = 100 if structure_valid else 0  # 20% weight
    structure_score *= 0.2
    
    return completeness_score + row_quality_score + structure_score

def print_quality_bar(score):
    """Print visual quality bar"""
    bar_length = 50
    filled = int(bar_length * score / 100)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    if score >= 90:
        color_indicator = "🟢"
        rating = "EXCELLENT"
    elif score >= 70:
        color_indicator = "🟡"
        rating = "GOOD"
    elif score >= 50:
        color_indicator = "🟠"
        rating = "FAIR"
    else:
        color_indicator = "🔴"
        rating = "POOR"
    
    print(f"\n  {color_indicator} [{bar}] {rating}")

def process_single_validation():
    """Process single CSV and JSON folder validation"""
    csv_path = get_csv_path()
    json_folder = get_json_folder()
    
    print("\n⏳ Analyzing JSON files...")
    json_info = analyze_json_folder(json_folder)
    print(f"  ✓ Analyzed {json_info['total_files']:,} JSON files")
    
    analyze_csv_quality(csv_path, json_info, show_full_report=True)

def process_batch_validation():
    """Process batch validation of multiple CSV files"""
    csv_folder = get_csv_folder()
    json_main_folder = get_json_main_folder()
    
    # Find CSV files
    print("\n🔍 Scanning for CSV files...")
    csv_files = sorted(csv_folder.glob('*.csv'))
    
    if not csv_files:
        print("❌ No CSV files found!")
        return
    
    print(f"  ✓ Found {len(csv_files)} CSV file(s)")
    
    # Find JSON subfolders
    print("\n🔍 Scanning for JSON subfolders...")
    json_folders = [item for item in json_main_folder.iterdir() if item.is_dir() and list(item.glob('*.json'))]
    
    if not json_folders:
        print("❌ No JSON subfolders found!")
        return
    
    print(f"  ✓ Found {len(json_folders)} JSON subfolder(s)")
    
    # Match CSV files to JSON folders
    print("\n🔗 Matching CSV files to JSON folders...")
    matches, unmatched_csv, unmatched_json = match_csv_to_json_folder(csv_files, json_folders)
    
    if matches:
        print(f"  ✓ Found {len(matches)} matching pair(s):")
        for csv_file, json_folder in matches:
            print(f"    • {csv_file.name} ← {json_folder.name}")
    
    if unmatched_csv:
        print(f"\n  ⚠ {len(unmatched_csv)} unmatched CSV file(s):")
        for csv_file in unmatched_csv:
            print(f"    • {csv_file.name}")
    
    if unmatched_json:
        print(f"\n  ⚠ {len(unmatched_json)} unmatched JSON folder(s):")
        for json_folder in unmatched_json:
            print(f"    • {json_folder.name}")
    
    if not matches:
        print("\n❌ No matching pairs found! Cannot proceed with validation.")
        return
    
    # Confirm processing
    confirm = input(f"\nValidate {len(matches)} matching pair(s)? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Batch validation cancelled.")
        return
    
    # Process each pair
    print(f"\n{'='*70}")
    print(f"BATCH VALIDATION - {len(matches)} PAIRS")
    print(f"{'='*70}\n")
    
    results = []
    
    for i, (csv_file, json_folder) in enumerate(matches, 1):
        print(f"[{i}/{len(matches)}] Validating: {csv_file.name}")
        print(f"{'─'*70}")
        
        try:
            # Analyze JSON folder
            print(f"  ⏳ Analyzing JSON folder: {json_folder.name}...")
            json_info = analyze_json_folder(json_folder)
            print(f"  ✓ Found {json_info['total_files']:,} JSON files")
            
            # Analyze CSV (brief mode)
            print(f"  ⏳ Validating CSV structure...")
            summary = analyze_csv_quality(csv_file, json_info, show_full_report=False)
            
            # Print brief summary
            status = "✓ PASS" if summary['structure_valid'] else "❌ FAIL"
            quality_emoji = "🟢" if summary['quality_score'] >= 90 else "🟡" if summary['quality_score'] >= 70 else "🟠" if summary['quality_score'] >= 50 else "🔴"
            
            print(f"\n  Status: {status}")
            print(f"  Quality Score: {quality_emoji} {summary['quality_score']:.1f}/100")
            print(f"  Missing Data: {summary['missing_percentage']:.2f}%")
            print(f"  Structure Match: {'✓' if summary['structure_valid'] else '❌'}")
            print(f"  Rows: {summary['rows']:,} {'✓' if summary['rows_match'] else '❌'}")
            print(f"  Columns: {summary['cols']:,} {'✓' if summary['cols_match'] else '❌'}")
            print(f"  Last Values: {'✓' if summary['has_last_values'] else '❌'}\n")
            
            results.append({
                'csv_name': csv_file.name,
                'json_folder': json_folder.name,
                'status': 'PASS' if summary['structure_valid'] else 'FAIL',
                'quality_score': summary['quality_score'],
                'missing_pct': summary['missing_percentage']
            })
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}\n")
            results.append({
                'csv_name': csv_file.name,
                'json_folder': json_folder.name,
                'status': 'ERROR',
                'quality_score': 0,
                'missing_pct': 100
            })
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"BATCH VALIDATION SUMMARY")
    print(f"{'='*70}\n")
    
    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed = sum(1 for r in results if r['status'] == 'FAIL')
    errors = sum(1 for r in results if r['status'] == 'ERROR')
    
    print(f"  Total Validated: {len(results)}")
    print(f"  ✓ Passed: {passed}")
    if failed > 0:
        print(f"  ❌ Failed: {failed}")
    if errors > 0:
        print(f"  ⚠ Errors: {errors}")
    
    if results:
        avg_score = sum(r['quality_score'] for r in results) / len(results)
        print(f"\n  Average Quality Score: {avg_score:.1f}/100")
    
    print(f"\n  Detailed Results:")
    print(f"  {'CSV File':<40} {'Status':>10} {'Score':>10}")
    print(f"  {'-'*62}")
    
    for r in results:
        status_symbol = "✓" if r['status'] == 'PASS' else "❌" if r['status'] == 'FAIL' else "⚠"
        print(f"  {r['csv_name']:<40} {status_symbol:>4} {r['status']:>6} {r['quality_score']:>9.1f}")
    
    print(f"\n{'='*70}\n")

def main():
    print("\n╔═══════════════════════════════════════════════════════════════════╗")
    print("║              CSV QUALITY CHECKER & VALIDATOR                      ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    
    try:
        mode = get_processing_mode()
        
        if mode == 1:
            process_single_validation()
        else:
            process_batch_validation()
        
    except KeyboardInterrupt:
        print("\n\n⚠ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()