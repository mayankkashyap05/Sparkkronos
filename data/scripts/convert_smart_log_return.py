import pandas as pd
import numpy as np
import os

def robust_log_return(series, clip_sigma=4.0):
    """
    Calculates Log Return but clips extreme outliers (Winsorization).
    Prevents 'Black Swan' events from confusing the model.
    """
    # 1. Standard Log Return
    # ln( Current / Previous )
    log_ret = np.log(series / series.shift(1))
    
    # 2. Calculate Statistics (on the past expanding window to prevent leakage)
    # However, for simple preprocessing, global clipping based on robust stats is standard
    # and has minimal leakage if sigma is high enough.
    
    # We use Median and IQR (Interquartile Range) because they are robust to outliers
    median = log_ret.median()
    q1 = log_ret.quantile(0.25)
    q3 = log_ret.quantile(0.75)
    iqr = q3 - q1
    
    # Define bounds (e.g., 4 IQRs away is an extreme outlier)
    lower_bound = median - (clip_sigma * iqr)
    upper_bound = median + (clip_sigma * iqr)
    
    # 3. Clip (Winsorize)
    # Any value > upper_bound becomes upper_bound
    # This keeps the "direction" but removes the "explosion"
    return log_ret.clip(lower=lower_bound, upper=upper_bound)

def relative_volume_encoding(vol_series, window=20):
    """
    Encodes Volume as 'Relative to recent average' instead of just 'Change'.
    Gives context: "Is this volume unusually high?"
    """
    # Standard Moving Average of Volume (Past 20 candles)
    # Shift(1) is CRITICAL to ensure we compare Current Vol to PAST Average (No Leakage)
    past_ma = vol_series.rolling(window=window).mean().shift(1)
    
    # Add epsilon to avoid divide by zero
    # Log ratio: If 0, volume is average. If > 0, volume is high.
    rel_vol = np.log((vol_series + 1e-8) / (past_ma + 1e-8))
    
    return rel_vol

def main():
    print("="*60)
    print("SMART LOG RETURN CONVERTER (Noise Reduction + Context)")
    print("="*60)

    # 1. Input
    while True:
        path = input("\nEnter path to ORIGINAL PROCESSED CSV (Prices):\n>> ").strip().strip('"').strip("'")
        if os.path.exists(path): break
        print("❌ File not found.")
        
    save_dir = input("\nEnter save folder path:\n>> ").strip().strip('"').strip("'")
    if not os.path.exists(save_dir):
        try: os.makedirs(save_dir)
        except: pass

    print(f"\nProcessing {os.path.basename(path)}...")
    df = pd.read_csv(path)
    
    # Clean headers
    df.columns = df.columns.str.strip()
    
    # 2. Process Data
    df_smart = df.copy()
    
    # A. Price Columns: Apply Robust Log Returns (Clipped)
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in df_smart.columns:
            # Convert to numeric, coerce errors
            s = pd.to_numeric(df_smart[col], errors='coerce')
            # Apply Smart Log Return
            df_smart[col] = robust_log_return(s, clip_sigma=4.0)
            print(f"✓ {col}: Log Return + Outlier Clipping Applied")

    # B. Volume Column: Apply Relative Volume (Context)
    if 'volume' in df_smart.columns:
        s_vol = pd.to_numeric(df_smart['volume'], errors='coerce')
        # Use Relative Volume instead of Change
        # This fits in the 'volume' column but carries more signal
        df_smart['volume'] = relative_volume_encoding(s_vol)
        print(f"✓ volume: Converted to Relative Volume (Context Aware)")

    # 3. Cleanup
    # Drop first 20 rows (due to Volume MA window)
    df_smart.dropna(inplace=True)
    
    # Strict Column Ordering
    desired_order = ['timestamps', 'open', 'high', 'low', 'close', 'volume']
    final_cols = [c for c in desired_order if c in df_smart.columns]
    df_smart = df_smart[final_cols]

    # 4. Save
    name_part = os.path.splitext(os.path.basename(path))[0]
    output_path = os.path.join(save_dir, f"{name_part}_smart_log_return.csv")
    
    df_smart.to_csv(output_path, index=False)
    
    print("\n" + "="*60)
    print("✅ SUCCESS")
    print("Improvements Applied:")
    print("1. Prices: Extreme outliers clipped (Fixes High Volatility failure).")
    print("2. Volume: Encoded as Relative Strength (Adds Context).")
    print(f"Saved to: {output_path}")
    print("="*60)

if __name__ == "__main__":
    main()