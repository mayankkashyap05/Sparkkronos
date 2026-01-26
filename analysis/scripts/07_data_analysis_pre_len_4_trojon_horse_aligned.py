import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def verify_dip_hunter_strategy(file_path):
    filename = os.path.basename(file_path)
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return

    print("\n" + "="*60)
    print(f"🎯 DIP HUNTER STRATEGY AUDIT")
    print("="*60)
    print(f"File: {filename}")
    
    # ----------------------------------------------------
    # 1. SETUP & CLEANING
    # ----------------------------------------------------
    # Trojan Columns
    col_score = 'last_value_open' # Inverted RSI (0.8 = Oversold)
    
    if col_score not in df.columns:
        print("❌ Error: Column 'last_value_open' not found.")
        return

    # Calculate Actual & Predicted Returns (Cumulative 1-Hour)
    actual_cols = [c for c in df.columns if 'actual_close_' in c]
    pred_cols = [c for c in df.columns if 'predicted_close_' in c]
    
    df['actual_return'] = df[actual_cols].sum(axis=1)
    df['pred_return'] = df[pred_cols].sum(axis=1)
    
    # ----------------------------------------------------
    # 2. APPLY THE STRATEGY RULES
    # ----------------------------------------------------
    # Rule 1: Market is Oversold (Score > 0.70)
    # Rule 2: AI Agrees (Prediction > 0)
    
    # Filter for Potential Setups (Market Condition)
    setups = df[df[col_score] > 0.70].copy()
    
    # Filter for Executed Trades (AI Confirmation)
    trades = setups[setups['pred_return'] > 0].copy()
    
    n_setups = len(setups)
    n_trades = len(trades)
    
    if n_trades == 0:
        print("❌ No trades found matching the strategy criteria.")
        return

    # ----------------------------------------------------
    # 3. PERFORMANCE METRICS
    # ----------------------------------------------------
    # Did the trade make money? (Actual Return > 0)
    trades['win'] = trades['actual_return'] > 0
    
    win_rate = trades['win'].mean() * 100
    avg_win = trades[trades['win']]['actual_return'].mean()
    avg_loss = trades[~trades['win']]['actual_return'].mean()
    
    # Profit Factor (Gross Gains / Gross Losses)
    gross_gain = trades[trades['win']]['actual_return'].sum()
    gross_loss = abs(trades[~trades['win']]['actual_return'].sum())
    
    profit_factor = gross_gain / gross_loss if gross_loss != 0 else 0
    
    print(f"\n📊 STRATEGY STATISTICS (Long Only)")
    print("-" * 40)
    print(f"Total Candles Scanned:   {len(df):,}")
    print(f"Valid Setups (RSI < 30): {n_setups:,} ({n_setups/len(df)*100:.1f}%)")
    print(f"Trades Taken (AI Says Up): {n_trades:,} ({n_trades/n_setups*100:.1f}% agreement)")
    print("-" * 40)
    print(f"🏆 WIN RATE:      {win_rate:.2f}%")
    print(f"💰 PROFIT FACTOR: {profit_factor:.2f}")
    print("-" * 40)
    
    # ----------------------------------------------------
    # 4. VERDICT
    # ----------------------------------------------------
    if win_rate > 55 and profit_factor > 1.2:
        print("✅ PASS: Strategy is PROFITABLE. Safe to trade.")
    elif profit_factor > 1.1:
        print("⚠️ MARGINAL: Strategy makes money but risk is high.")
    else:
        print("❌ FAIL: Strategy loses money. Do not trade.")

    # ----------------------------------------------------
    # 5. VISUALIZATION: EQUITY CURVE
    # ----------------------------------------------------
    # Simulate Account Growth
    trades = trades.sort_index() # Ensure time order
    trades['pnl'] = trades['actual_return']
    trades['equity'] = trades['pnl'].cumsum()
    
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(12, 6))
    
    # Plot Equity Curve
    plt.plot(range(len(trades)), trades['equity'], color='green', linewidth=2, label='Strategy Equity')
    plt.axhline(0, color='black', linestyle='--')
    
    plt.title(f"Dip Hunter Equity Curve (Win Rate: {win_rate:.1f}%)")
    plt.xlabel("Number of Trades")
    plt.ylabel("Cumulative Log Return")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    while True:
        path = input("\nEnter prediction CSV path:\n>> ").strip().strip('"').strip("'")
        if not path: continue
        if path.lower() in ['q', 'exit']: break
        
        if os.path.exists(path):
            verify_dip_hunter_strategy(path)
        else:
            print("❌ File not found.")