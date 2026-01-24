import pandas as pd
import numpy as np
import os
import sys

def main():
    print("="*60)
    print("LOG RETURN CONVERTER (Strict Timestamp Preservation)")
    print("="*60)

    # 1. Get Input Path
    while True:
        input_path = input("\nEnter the full path to your CSV file:\n>> ").strip().strip('"').strip("'")
        if not input_path: continue
        if os.path.exists(input_path):
            break
        print("❌ File not found.")

    # 2. Get Save Directory
    while True:
        save_dir = input("\nEnter the folder path to save the output:\n>> ").strip().strip('"').strip("'")
        if not save_dir: continue
        if os.path.isdir(save_dir):
            break
        try:
            os.makedirs(save_dir)
            break
        except:
            print("❌ Invalid folder path.")

    print(f"\nProcessing {os.path.basename(input_path)}...")

    # 3. Load Data
    df = pd.read_csv(input_path)
    
    # Strip whitespace from column names to be safe
    df.columns = df.columns.str.strip()

    # Verify 'timestamps' exists
    if 'timestamps' not in df.columns:
        print(f"❌ ERROR: Column 'timestamps' not found. Found: {list(df.columns)}")
        print("Please ensure your CSV header is exactly: timestamps,open,high,low,close,volume")
        return

    # 4. Calculate Log Returns
    df_calc = df.copy()
    
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    
    for col in numeric_cols:
        if col in df_calc.columns:
            # Force numeric conversion
            series = pd.to_numeric(df_calc[col], errors='coerce')
            
            # Calculate Log Return
            if col == 'volume':
                 # Add epsilon to volume to avoid log(0)
                 log_ret = np.log((series + 1e-8) / (series.shift(1) + 1e-8))
            else:
                 log_ret = np.log(series / series.shift(1))
            
            # Overwrite the column with the log return
            df_calc[col] = log_ret
            print(f"✓ Converted '{col}' to log return")

    # 5. Clean Data
    # Drop the first row (it is always NaN because of shift(1))
    df_calc = df_calc.iloc[1:].copy()
    
    # Replace Infinite values with 0
    cols_to_check = [c for c in numeric_cols if c in df_calc.columns]
    
    # --- FIX IS HERE ---
    # We use .values.any() to check the entire block at once
    if np.isinf(df_calc[cols_to_check]).values.any():
        print("⚠ Replacing infinite values with 0")
        df_calc[cols_to_check] = df_calc[cols_to_check].replace([np.inf, -np.inf], 0)

    # 6. Ensure Strict Column Order
    required_order = ['timestamps', 'open', 'high', 'low', 'close', 'volume']
    final_cols = [c for c in required_order if c in df_calc.columns]
    df_final = df_calc[final_cols]

    # 7. Save Output
    filename = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(save_dir, f"{filename}_log_return.csv")
    
    # Save WITHOUT the pandas index
    df_final.to_csv(output_path, index=False)

    print("\n" + "="*60)
    print("✅ SUCCESS")
    print(f"Saved to: {output_path}")
    print(f"Columns: {list(df_final.columns)}")
    print(f"Rows: {len(df_final)}")
    print("="*60)

if __name__ == "__main__":
    main()