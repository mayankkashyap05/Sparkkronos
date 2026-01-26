import pandas as pd
import numpy as np
import os

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_sma(series, period=50):
    return series.rolling(window=period).mean()

def calculate_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period-1, min_periods=period).mean()

# ==========================================
# 2. MAIN CONVERTER (ALIGNED POLARITY)
# ==========================================
def main():
    print("="*60)
    print("🔋 ALIGNED TROJAN HORSE (Positive Correlation Mode)")
    print("   Flipping indicators so 'High Value' = 'Buy Signal'")
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
    
    # 2. CALCULATION
    print("...Calculating & Aligning Indicators")
    
    # A. CLOSE (Target) -> Robust Log Return
    df['target_close'] = np.log(df['close'] / df['close'].shift(1))
    
    # B. OPEN -> INVERTED RSI (The "Dip Buy" Score)
    # Logic: Real RSI 30 (Oversold) -> Becomes 0.70 (High Score -> Buy)
    #        Real RSI 70 (Overbought) -> Becomes 0.30 (Low Score -> Sell)
    # Formula: 1.0 - (RSI / 100)
    rsi_series = calculate_rsi(df['close'], period=14)
    df['trojan_open'] = 1.0 - (rsi_series / 100.0)
    
    # C. HIGH -> INVERTED SMA DISTANCE (The "Mean Reversion" Score)
    # Logic: Price FAR ABOVE SMA -> Negative Value -> Becomes Low Score -> Sell
    #        Price FAR BELOW SMA -> Positive Value -> Becomes High Score -> Buy
    # Formula: -(Price - SMA) / SMA
    sma_series = calculate_sma(df['close'], period=50)
    # We negate the distance so that "Below SMA" becomes a POSITIVE signal
    df['trojan_high'] = -1 * (df['close'] - sma_series) / (sma_series.replace(0, np.nan))
    
    # D. LOW -> INVERTED Volatility (The "Safety" Score)
    # Logic: High Volatility is dangerous. We want the model to be cautious.
    #        So High Volatility -> Low Score.
    #        Low Volatility -> High Score.
    atr_series = calculate_atr(df, period=14)
    # Inverse: 1 / ATR (Capped to avoid infinity)
    # A simpler approach: -ATR. Let's stick to -ATR %
    df['trojan_low'] = -1 * (atr_series / df['close'])
    
    # E. VOLUME -> Relative Volume (Kept Normal)
    # Volume is ambiguous (can be stopping volume or breakout volume)
    vol_ma = df['volume'].rolling(20).mean()
    df['trojan_volume'] = np.log((df['volume'] + 1) / (vol_ma + 1))

    # 3. OVERWRITE
    df_final = pd.DataFrame()
    df_final['timestamps'] = df['timestamps']
    
    df_final['open']   = df['trojan_open']      # Inverted RSI (High = Buy)
    df_final['high']   = df['trojan_high']      # Inverted SMA (High = Buy)
    df_final['low']    = df['trojan_low']       # Inverted Vol (High = Safe)
    df_final['close']  = df['target_close']     # Target
    df_final['volume'] = df['trojan_volume']    # Rel Vol
    
    # 4. CLEANUP
    df_final.dropna(inplace=True)
    cols = ['open', 'high', 'low', 'close', 'volume']
    df_final[cols] = df_final[cols].replace([np.inf, -np.inf], 0)
    
    # 5. SAVE
    name_part = os.path.splitext(os.path.basename(path))[0]
    output_path = os.path.join(save_dir, f"{name_part}_trojan_aligned_old.csv")
    
    df_final.to_csv(output_path, index=False)
    
    print("\n" + "="*60)
    print("✅ SUCCESS! Aligned Data Ready.")
    print("="*60)
    print(f"File: {output_path}")
    print("\nNEW MAPPING (High Value = Price UP):")
    print(" 'open'   --> 1 - RSI (High Score means Oversold/Cheap)")
    print(" 'high'   --> Below SMA Score (High Score means Price is low)")
    print("="*60)

if __name__ == "__main__":
    main()