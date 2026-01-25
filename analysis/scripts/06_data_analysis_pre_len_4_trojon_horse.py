import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def analyze_trojan_file(file_path):
    filename = os.path.basename(file_path)
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return

    print("\n" + "="*60)
    print(f"🐴 TROJAN HORSE DECODER & PREDICTIVE CHECKS")
    print("="*60)
    print(f"File: {filename}")
    print(f"Rows: {len(df):,}")

    # ----------------------------------------------------
    # A. DECODE THE COLUMNS (Mental Mapping)
    # ----------------------------------------------------
    # Open   = RSI (0.0 - 1.0)
    # High   = SMA Distance
    # Low    = Volatility (ATR %)
    # Close  = Log Return (Target)
    # Volume = Relative Volume (Log Ratio)

    print("\n1. DATA INTEGRITY CHECK")
    print("-" * 30)
    
    # Check RSI (Open)
    rsi_min = df['open'].min()
    rsi_max = df['open'].max()
    print(f"RSI (Open): Min={rsi_min:.4f}, Max={rsi_max:.4f}")
    if rsi_min < 0 or rsi_max > 1.0:
        print("   ⚠️  WARNING: RSI values out of bounds (0-1). Scaling issue?")
    else:
        print("   ✅ RSI looks valid.")

    # Check Log Returns (Close)
    ret_mean = df['close'].mean()
    print(f"Target (Close): Mean={ret_mean:.6f} (Should be near 0)")
    
    # ----------------------------------------------------
    # B. SIGNAL QUALITY CHECK (Does this data predict anything?)
    # ----------------------------------------------------
    print("\n2. SIGNAL PREDICTIVE POWER (The 'Why' it works)")
    print("-" * 30)

    # TEST 1: The RSI Reversion Edge
    # Does High RSI (Open > 0.7) lead to Negative Returns (Close < 0)?
    overbought = df[df['open'] > 0.7]
    oversold = df[df['open'] < 0.3]
    
    avg_ret_ob = overbought['close'].mean()
    avg_ret_os = oversold['close'].mean()
    
    print(f"RSI > 70 (Overbought) -> Next Avg Return: {avg_ret_ob:.6f}")
    print(f"RSI < 30 (Oversold)   -> Next Avg Return: {avg_ret_os:.6f}")
    
    if avg_ret_ob < avg_ret_os:
        print("   ✅ SUCCESS: High RSI leads to lower returns (Mean Reversion detected).")
    else:
        print("   ⚠️  WARNING: No clear Mean Reversion signal found.")

    # TEST 2: The Volatility Expansion Edge
    # Does High Volatility (Low) lead to larger absolute moves?
    high_vol = df[df['low'] > df['low'].quantile(0.8)]
    low_vol = df[df['low'] < df['low'].quantile(0.2)]
    
    move_hv = high_vol['close'].abs().mean()
    move_lv = low_vol['close'].abs().mean()
    
    print(f"High Volatility Context -> Avg Move Size: {move_hv:.6f}")
    print(f"Low Volatility Context  -> Avg Move Size: {move_lv:.6f}")
    
    if move_hv > move_lv * 1.2:
        print("   ✅ SUCCESS: Volatility indicator correctly predicts regime.")
    else:
        print("   ⚠️  WARNING: Volatility signal is weak.")

    # ----------------------------------------------------
    # C. VISUALIZATION DASHBOARD
    # ----------------------------------------------------
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3)

    # Plot 1: RSI Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(df['open'], bins=50, ax=ax1, color='purple', kde=True)
    ax1.set_title("RSI Distribution (Open)")
    ax1.axvline(0.7, color='red', linestyle='--')
    ax1.axvline(0.3, color='green', linestyle='--')

    # Plot 2: Volatility Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(df['low'], bins=50, ax=ax2, color='orange', kde=True)
    ax2.set_title("Volatility/ATR % (Low)")

    # Plot 3: Target Log Returns
    ax3 = fig.add_subplot(gs[0, 2])
    sns.histplot(df['close'], bins=100, ax=ax3, color='blue', kde=False)
    ax3.set_title("Target Log Returns (Close)")
    ax3.set_xlim(-0.03, 0.03)

    # Plot 4: RSI vs Future Return (Scatter) - The "Alpha" Plot
    # We want to see a negative slope here (High RSI = Negative Return)
    ax4 = fig.add_subplot(gs[1, 0])
    # Downsample for speed
    sample = df.sample(min(2000, len(df)))
    sns.regplot(x=sample['open'], y=sample['close'], ax=ax4, 
                scatter_kws={'alpha':0.2, 'color':'gray'}, line_kws={'color':'red'})
    ax4.set_title("Correlation: RSI (x) vs Next Return (y)")
    ax4.set_xlabel("RSI (Open)")
    ax4.set_ylabel("Next Log Return")

    # Plot 5: Volatility vs Absolute Return - The "Risk" Plot
    # We want to see a positive slope (High Vol Input = High Abs Return)
    ax5 = fig.add_subplot(gs[1, 1])
    sns.regplot(x=sample['low'], y=sample['close'].abs(), ax=ax5,
                scatter_kws={'alpha':0.2, 'color':'gray'}, line_kws={'color':'blue'})
    ax5.set_title("Correlation: Volatility (x) vs Move Size (y)")
    ax5.set_xlabel("Volatility (Low)")
    ax5.set_ylabel("Absolute Return Size")

    # Plot 6: Correlation Heatmap
    ax6 = fig.add_subplot(gs[1, 2])
    corr = df[['open', 'high', 'low', 'close', 'volume']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax6, fmt='.2f')
    ax6.set_title("Feature Correlation Matrix")

    # Plot 7: Time Series Reconstruction
    ax7 = fig.add_subplot(gs[2, :])
    snippet = df.iloc[-300:].reset_index(drop=True)
    price_proxy = 100 * np.exp(snippet['close'].cumsum())
    
    ln1 = ax7.plot(price_proxy, color='black', label='Price (Reconstructed)', alpha=0.6)
    ax7.set_ylabel("Price Proxy")
    
    ax7b = ax7.twinx()
    ln2 = ax7b.plot(snippet['open'], color='purple', label='RSI (Open)', linewidth=1.5, alpha=0.8)
    ax7b.set_ylim(0, 1)
    ax7b.axhline(0.7, color='red', alpha=0.3, linestyle='--')
    ax7b.axhline(0.3, color='green', alpha=0.3, linestyle='--')
    
    # Legend combining both axes
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax7.legend(lns, labs, loc='upper left')
    ax7.set_title("Visual Confirmation: Price vs RSI")

    plt.tight_layout()
    plt.show()

    print("\n✅ Analysis Complete.")
    print("   Look at 'Plot 4': If the red line slopes DOWN, your model will learn to short tops.")
    print("   Look at 'Plot 6': Ensure features are not 100% correlated (dark red/blue everywhere).")

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def main():
    while True:
        user_input = input("\nEnter path to TROJAN CSV file:\n>> ").strip().strip('"').strip("'")
        
        if not user_input: continue
        if user_input.lower() in ['exit', 'q']: break
        
        if not os.path.exists(user_input):
            print("❌ Path not found.")
            continue

        analyze_trojan_file(user_input)

if __name__ == "__main__":
    main()