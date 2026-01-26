import pandas as pd
import numpy as np
import os

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_sma(series, period=50):
    return series.rolling(window=period).mean()


def calculate_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(com=period - 1, min_periods=period).mean()


# ==========================================
# 2. MAIN (FULLY AUTOMATED)
# ==========================================
def main():
    print("=" * 60)
    print("🔋 ALIGNED TROJAN HORSE GENERATOR (AUTO MODE)")
    print("   • No user input")
    print("   • Leakage-safe")
    print("=" * 60)

    # -------------------------------------------------------------
    # FIXED & CORRECT PATH RESOLUTION
    # -------------------------------------------------------------
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    LIVE_DEMO_DIR = os.path.dirname(SCRIPT_DIR)

    input_path = os.path.join(
        LIVE_DEMO_DIR,
        "data",
        "raw",
        "SOLUSD_15m_processed.csv"
    )

    output_dir = os.path.join(
        LIVE_DEMO_DIR,
        "data",
        "processed"
    )

    output_file = "SOLUSD_15m_processed_trojon_aligned.csv"
    output_path = os.path.join(output_dir, output_file)

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ Input file not found:\n{input_path}")

    print(f"📥 Input : {input_path}")
    print(f"📤 Output: {output_path}")

    # -------------------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------------------
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()

    # -------------------------------------------------------------
    # 🔒 SAFETY LOCK — FORCE CHRONOLOGICAL ORDER
    # -------------------------------------------------------------
    time_col = None
    for col in ['timestamps', 'timestamp', 'date', 'datetime']:
        if col in df.columns:
            time_col = col
            break

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col, ascending=True).reset_index(drop=True)
        print(f"✅ Sorted by '{time_col}' (Oldest → Newest)")
    else:
        print("⚠️  No timestamp column found — assuming already sorted")

    # -------------------------------------------------------------
    # FEATURE ENGINEERING (ALIGNED POLARITY)
    # -------------------------------------------------------------
    print("⚙️  Calculating aligned features...")

    # Target (Log Return)
    df['target_close'] = np.log(df['close'] / df['close'].shift(1))

    # Inverted RSI (Dip-buy score)
    rsi = calculate_rsi(df['close'], 14)
    df['trojan_open'] = 1.0 - (rsi / 100.0)

    # Inverted SMA distance (Mean reversion)
    sma = calculate_sma(df['close'], 50)
    df['trojan_high'] = -1 * (df['close'] - sma) / sma.replace(0, np.nan)

    # Inverted ATR (Risk penalty)
    atr = calculate_atr(df, 14)
    df['trojan_low'] = -1 * (atr / df['close'])

    # Relative Volume
    vol_ma = df['volume'].rolling(20).mean()
    df['trojan_volume'] = np.log((df['volume'] + 1) / (vol_ma + 1))

    # -------------------------------------------------------------
    # FINAL DATASET
    # -------------------------------------------------------------
    df_final = pd.DataFrame()

    if time_col:
        df_final[time_col] = df[time_col]
    else:
        df_final['timestamps'] = df.index

    df_final['open'] = df['trojan_open']
    df_final['high'] = df['trojan_high']
    df_final['low'] = df['trojan_low']
    df_final['close'] = df['target_close']
    df_final['volume'] = df['trojan_volume']

    # -------------------------------------------------------------
    # CLEANUP
    # -------------------------------------------------------------
    df_final.dropna(inplace=True)
    df_final.replace([np.inf, -np.inf], 0, inplace=True)

    # -------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------
    df_final.to_csv(output_path, index=False)

    print("=" * 60)
    print("✅ DONE — Trojan-Aligned dataset ready")
    print("=" * 60)
    print("High value = bullish bias")
    print("Next step: feed directly into Kronos / TFT / agent models")


if __name__ == "__main__":
    main()
