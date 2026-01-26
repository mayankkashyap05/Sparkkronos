import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# ==========================================
# CONFIGURATION
# ==========================================
TAKE_PROFIT_PCT = 0.020  # +2.0%
STOP_LOSS_PCT   = -0.015 # -1.5%
TROJAN_ENTRY_THRESHOLD = 0.70 # Inverted RSI > 0.70

def simulate_sniper_logic(row):
    """
    Simulates the trade path candle-by-candle to check for TP/SL hits.
    Returns: (Final PnL, Exit Reason, Duration Candles)
    """
    cumulative_pnl = 0.0
    
    # We step through the 4 future candles (1 Hour max)
    # Note: We use 'actual_close_X' which represents the Log Return of that candle
    steps = ['actual_close_1', 'actual_close_2', 'actual_close_3', 'actual_close_4']
    
    for i, step_col in enumerate(steps):
        candle_return = row[step_col]
        cumulative_pnl += candle_return
        
        # 1. Check Take Profit
        if cumulative_pnl >= TAKE_PROFIT_PCT:
            return TAKE_PROFIT_PCT, 'TP Hit (+2.0%)', i + 1
            
        # 2. Check Stop Loss
        if cumulative_pnl <= STOP_LOSS_PCT:
            return STOP_LOSS_PCT, 'SL Hit (-1.5%)', i + 1
            
    # 3. Time Exit (Held for full 4 candles)
    return cumulative_pnl, 'Time Exit (1h)', 4

def verify_sniper_strategy(file_path):
    filename = os.path.basename(file_path)
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return

    print("\n" + "="*60)
    print(f"🔫 SNIPER STRATEGY AUDIT (+{TAKE_PROFIT_PCT*100}% TP / {STOP_LOSS_PCT*100}% SL)")
    print("="*60)
    print(f"File: {filename}")
    
    # ----------------------------------------------------
    # 1. FILTER TRADES (ENTRY LOGIC)
    # ----------------------------------------------------
    col_score = 'last_value_open' # Inverted RSI
    
    if col_score not in df.columns:
        print("❌ Error: Trojan columns not found.")
        return

    # Calculate Predicted Return (Sum of 4 steps) to verify AI Signal
    pred_cols = [c for c in df.columns if 'predicted_close_' in c]
    df['pred_return'] = df[pred_cols].sum(axis=1)
    
    # Filter: Oversold (Trojan > 0.70) AND AI Says Buy (Pred > 0)
    trades = df[(df[col_score] > TROJAN_ENTRY_THRESHOLD) & (df['pred_return'] > 0)].copy()
    
    if len(trades) == 0:
        print("❌ No trades found.")
        return

    # ----------------------------------------------------
    # 2. SIMULATE EXITS
    # ----------------------------------------------------
    print(f"Processing {len(trades)} trades...", end="")
    
    # Apply the simulation row by row
    simulation_results = trades.apply(simulate_sniper_logic, axis=1)
    
    # Unpack results into new columns
    trades['sniper_pnl'] = [x[0] for x in simulation_results]
    trades['exit_reason'] = [x[1] for x in simulation_results]
    trades['duration'] = [x[2] for x in simulation_results]
    
    # Calculate Blind Hold PnL for comparison (Sum of all 4 closes)
    actual_cols = [c for c in df.columns if 'actual_close_' in c]
    trades['blind_pnl'] = trades[actual_cols].sum(axis=1)
    
    print(" Done.")

    # ----------------------------------------------------
    # 3. DEEP ANALYSIS & COMPARISON
    # ----------------------------------------------------
    
    # --- Statistics ---
    def get_stats(pnl_series):
        wins = pnl_series > 0
        gross_gain = pnl_series[wins].sum()
        gross_loss = abs(pnl_series[~wins].sum())
        pf = gross_gain / gross_loss if gross_loss != 0 else 0
        return wins.mean() * 100, pf, pnl_series.sum()

    blind_win, blind_pf, blind_total = get_stats(trades['blind_pnl'])
    sniper_win, sniper_pf, sniper_total = get_stats(trades['sniper_pnl'])
    
    print(f"\n📊 PERFORMANCE COMPARISON")
    print("-" * 65)
    print(f"{'METRIC':<20} | {'🐢 BLIND HOLD':<15} | {'🔫 SNIPER EXIT':<15} |")
    print("-" * 65)
    print(f"{'Win Rate':<20} | {blind_win:6.2f}%         | {sniper_win:6.2f}%         |")
    print(f"{'Profit Factor':<20} | {blind_pf:6.2f}          | {sniper_pf:6.2f}          |")
    print(f"{'Total Return (Log)':<20} | {blind_total:6.4f}          | {sniper_total:6.4f}          |")
    print("-" * 65)
    
    # Improvement Check
    improvement = ((sniper_total - blind_total) / abs(blind_total)) * 100
    if sniper_total > blind_total:
        print(f"✅ IMPROVEMENT: Sniper strategy earned {improvement:.1f}% more profit.")
    else:
        print(f"⚠️ REGRESSION: Blind hold was better by {abs(improvement):.1f}%.")

    # --- Exit Analysis ---
    exit_counts = trades['exit_reason'].value_counts()
    print(f"\n🔍 EXIT DISTRIBUTION")
    for reason, count in exit_counts.items():
        print(f"  {reason}: {count} trades ({count/len(trades)*100:.1f}%)")

    # ----------------------------------------------------
    # 4. VISUALIZATION
    # ----------------------------------------------------
    trades = trades.sort_index()
    trades['equity_blind'] = trades['blind_pnl'].cumsum()
    trades['equity_sniper'] = trades['sniper_pnl'].cumsum()
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Chart 1: Equity Curves
    # Using  logic
    ax1.plot(range(len(trades)), trades['equity_blind'], label='Blind Hold (1hr)', color='gray', linestyle='--')
    ax1.plot(range(len(trades)), trades['equity_sniper'], label='Sniper Exit (Dynamic)', color='green', linewidth=2)
    ax1.axhline(0, color='black')
    ax1.set_title("Equity Curve Comparison")
    ax1.set_xlabel("Trade Number")
    ax1.set_ylabel("Cumulative Log Return")
    ax1.legend()
    
    # Chart 2: Exit Distribution Pie
    # Using  logic
    # Clean labels without emojis for matplotlib
    colors = {'TP Hit (+2.0%)': '#4CAF50', 'SL Hit (-1.5%)': '#F44336', 'Time Exit (1h)': '#FFC107'}
    present_colors = [colors.get(x, 'gray') for x in exit_counts.index]
    
    ax2.pie(exit_counts, labels=exit_counts.index, autopct='%1.1f%%', startangle=140, colors=present_colors)
    ax2.set_title("How Trades Ended")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    while True:
        try:
            # Clean input to prevent errors if user drags & drops file with quotes
            path = input("\nEnter prediction CSV path:\n>> ").strip().strip('"').strip("'")
            if not path: continue
            if path.lower() in ['q', 'exit']: break
            
            if os.path.exists(path):
                verify_sniper_strategy(path)
            else:
                print("❌ File not found.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break