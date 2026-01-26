import pandas as pd
import numpy as np
import os

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def calculate_rsi(series, period=14):
    """
    Standard RSI calculation using Wilder's Smoothing.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_sma(series, period=50):
    """
    Simple Moving Average.
    """
    return series.rolling(window=period).mean()

def calculate_atr(df, period=14):
    """
    Average True Range (Volatility).
    """
    high = df['high']
    low = df['low']
    close = df['close']
    # shift(1) is safe here because we force-sort the DF in main()
    prev_close = close.shift(1)
    
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    
    return tr.ewm(com=period-1, min_periods=period).mean()

# ==========================================
# 2. MAIN CONVERTER (ALIGNED POLARITY + SAFETY LOCK)
# ==========================================
def main():
    print("="*60)
    print("🔋 ALIGNED TROJAN HORSE GENERATOR (Leakage Proof)")
    print("   1. Sorts Data (Safety Lock)")
    print("   2. Aligns Indicators (High Value = Buy Signal)")
    print("="*60)

    # 1. Input
    while True:
        path = input("\nEnter path to ORIGINAL PROCESSED CSV (Prices):\n>> ").strip().strip('"').strip("'")
        if os.path.exists(path): break
        print("❌ File not found.")
        
    save_dir = os.path.dirname(path)

    print(f"\nProcessing {os.path.basename(path)}...")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # -------------------------------------------------------------
    # 🔒 SAFETY LOCK: FORCE CHRONOLOGICAL ORDER (CRITICAL)
    # -------------------------------------------------------------
    print("...Verifying Sort Order (Preventing Future Leakage)")
    
    # Identify timestamp column
    time_col = None
    possible_names = ['timestamps', 'timestamp', 'date', 'datetime']
    for name in possible_names:
        if name in df.columns:
            time_col = name
            break
            
    if time_col:
        # Force Sort: Oldest -> Newest
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(by=time_col, ascending=True).reset_index(drop=True)
        print(f"   ✅ Data sorted by '{time_col}' (Oldest to Newest).")
        print("   ✅ Math is now safe from leakage.")
    else:
        print("   ⚠️  WARNING: No timestamp column found. Assuming input is already sorted.")
        print("       (If your file is Newest->Oldest, this will cause leakage!)")

    # -------------------------------------------------------------
    
    # 2. CALCULATION (Aligned Polarity)
    # Logic: Transform inputs so that "High Value" always correlates with "Price Up"
    print("...Calculating & Aligning Indicators")
    
    # A. CLOSE (Target) -> Robust Log Return
    # Predicts the move from Yesterday to Today (Autoregressive Target)
    df['target_close'] = np.log(df['close'] / df['close'].shift(1))
    
    # B. OPEN -> INVERTED RSI (The "Dip Buy" Score)
    # Logic: Real RSI 30 (Oversold) -> Becomes 0.70 (High Score -> Buy)
    #        Real RSI 70 (Overbought) -> Becomes 0.30 (Low Score -> Sell)
    rsi_series = calculate_rsi(df['close'], period=14)
    df['trojan_open'] = 1.0 - (rsi_series / 100.0)
    
    # C. HIGH -> INVERTED SMA DISTANCE (The "Mean Reversion" Score)
    # Logic: Price Below SMA -> Positive Distance -> High Score -> Buy
    sma_series = calculate_sma(df['close'], period=50)
    # We multiply by -1 so that (Price < SMA) becomes Positive
    # We replace 0 SMA with NaN to avoid division error (cleaned later)
    df['trojan_high'] = -1 * (df['close'] - sma_series) / (sma_series.replace(0, np.nan))
    
    # D. LOW -> INVERTED Volatility (The "Safety" Score)
    # Logic: High Volatility is risky. We invert it so High Vol = Low Score.
    atr_series = calculate_atr(df, period=14)
    # Normalized ATR % (e.g., 0.02). Inverted to -0.02.
    df['trojan_low'] = -1 * (atr_series / df['close'])
    
    # E. VOLUME -> Relative Volume (Standard Log Ratio)
    # We keep this neutral (Log Return of Volume)
    vol_ma = df['volume'].rolling(20).mean()
    df['trojan_volume'] = np.log((df['volume'] + 1) / (vol_ma + 1))

    # 3. OVERWRITE COLUMNS
    df_final = pd.DataFrame()
    
    if time_col:
        df_final[time_col] = df[time_col]
    else:
        # Create dummy index if missing
        df_final['timestamps'] = df.index
    
    df_final['open']   = df['trojan_open']      # Inverted RSI (High = Buy)
    df_final['high']   = df['trojan_high']      # Inverted SMA (High = Buy)
    df_final['low']    = df['trojan_low']       # Inverted Vol (High = Safe)
    df_final['close']  = df['target_close']     # Target Log Return
    df_final['volume'] = df['trojan_volume']    # Relative Volume
    
    # 4. CLEANUP
    # Drop the first 50 rows (NaNs from SMA calculation)
    print(f"   Cleaning NaNs (First 50 rows)...")
    df_final.dropna(inplace=True)
    
    # Handle Infinite values (divide by zero protection)
    cols = ['open', 'high', 'low', 'close', 'volume']
    df_final[cols] = df_final[cols].replace([np.inf, -np.inf], 0)
    
    # 5. SAVE
    name_part = os.path.splitext(os.path.basename(path))[0]
    output_path = os.path.join(save_dir, f"{name_part}_trojan_aligned.csv")
    
    df_final.to_csv(output_path, index=False)
    
    print("\n" + "="*60)
    print("✅ SUCCESS! Aligned Data Ready.")
    print("="*60)
    print(f"File: {output_path}")
    print("\nDATA MAPPING (High Value = Price UP):")
    print(" 'open'   --> 1.0 - RSI (High Score means Oversold/Cheap)")
    print(" 'high'   --> Below SMA Score (High Score means Price is Low)")
    print(" 'close'  --> Log Return (Target)")
    print("="*60)
    print("👉 ACTION: Train Kronos-Base on this file.")

if __name__ == "__main__":
    main()