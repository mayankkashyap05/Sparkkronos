import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# ==========================================
# ⚖️ TIMEKEEPER STRATEGY SETTINGS
# ==========================================
# We found that "Blind Hold" (Time Exit) beats "Sniper" (Targets).
# So we use Time as the primary exit, with a "Catastrophe Stop" only.

STOP_LOSS_PCT   = -0.030  # -3.0% (Wide Safety Net for Crashes)
TAKE_PROFIT_PCT = 9.999   # DISABLED (Let winners run fully)
TROJAN_ENTRY_THRESHOLD = 0.70 # Inverted RSI > 0.70 (Oversold)

def simulate_timekeeper_logic(row):
    """
    Simulates the 'Timekeeper' trade path.
    - Holds for the full duration (4 candles) to capture the trend.
    - Only exits early if a DISASTER happens (-3% Loss).
    Returns: (Final PnL, Exit Reason, Duration Candles)
    """
    cumulative_pnl = 0.0
    
    # We step through the 4 future candles (1 Hour max)
    steps = ['actual_close_1', 'actual_close_2', 'actual_close_3', 'actual_close_4']
    
    for i, step_col in enumerate(steps):
        # Add the log return of this 15m candle
        candle_return = row[step_col]
        cumulative_pnl += candle_return
        
        # 1. Check Catastrophic Stop Loss
        if cumulative_pnl <= STOP_LOSS_PCT:
            return STOP_LOSS_PCT, '🛑 Hard Stop (-3%)', i + 1
            
    # 2. Time Exit (The Standard Exit)
    # If we survive 1 hour, we take whatever the market gave us (Profit or Loss)
    return cumulative_pnl, '⏳ Time Exit (1h)', 4

def verify_strategy(file_path):
    filename = os.path.basename(file_path)
    try:
        # Check if path is a directory (Common User Error Fix)
        if os.path.isdir(file_path):
            print(f"❌ Error: '{filename}' is a directory, not a CSV file.")
            return

        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return

    print("\n" + "="*60)
    print(f"🕰️ TIMEKEEPER STRATEGY AUDIT (1 Hr Hold / -3% Safety Stop)")
    print("="*60)
    print(f"File: {filename}")
    
    # ----------------------------------------------------
    # 1. FILTER TRADES (ENTRY LOGIC)
    # ----------------------------------------------------
    col_score = 'last_value_open' # Inverted RSI
    
    if col_score not in df.columns:
        print("❌ Error: Trojan columns not found. Is this the right file?")
        return

    # Calculate Predicted Return (Sum of 4 steps)
    pred_cols = [c for c in df.columns if 'predicted_close_' in c]
    df['pred_return'] = df[pred_cols].sum(axis=1)
    
    # Filter: Oversold (Trojan > 0.70) AND AI Says Buy (Pred > 0)
    trades = df[(df[col_score] > TROJAN_ENTRY_THRESHOLD) & (df['pred_return'] > 0)].copy()
    
    if len(trades) == 0:
        print("❌ No trades found matching criteria.")
        return

    # ----------------------------------------------------
    # 2. SIMULATE EXITS
    # ----------------------------------------------------
    print(f"Simulating {len(trades)} trades...", end="")
    
    # Apply the simulation
    simulation_results = trades.apply(simulate_timekeeper_logic, axis=1)
    
    # Unpack results
    trades['realized_pnl'] = [x[0] for x in simulation_results]
    trades['exit_reason'] = [x[1] for x in simulation_results]
    trades['duration'] = [x[2] for x in simulation_results]
    
    print(" Done.")

    # ----------------------------------------------------
    # 3. ANALYSIS
    # ----------------------------------------------------
    
    # Calculate Stats
    wins = trades['realized_pnl'] > 0
    n_wins = wins.sum()
    n_loss = (~wins).sum()
    win_rate = n_wins / len(trades) * 100
    
    gross_gain = trades[wins]['realized_pnl'].sum()
    gross_loss = abs(trades[~wins]['realized_pnl'].sum())
    profit_factor = gross_gain / gross_loss if gross_loss != 0 else 0
    
    total_return = trades['realized_pnl'].sum()
    
    print(f"\n📊 REALISTIC PERFORMANCE")
    print("-" * 40)
    print(f"Trades Taken:    {len(trades)}")
    print(f"Win Rate:        {win_rate:.2f}%")
    print(f"Profit Factor:   {profit_factor:.2f} (Target > 1.2)")
    print(f"Total Net PnL:   {total_return:.4f} (Log Return)")
    print("-" * 40)

    # --- Exit Analysis ---
    exit_counts = trades['exit_reason'].value_counts()
    print(f"\n🔍 EXIT DISTRIBUTION")
    for reason, count in exit_counts.items():
        print(f"  {reason}: {count} trades ({count/len(trades)*100:.1f}%)")

    # ----------------------------------------------------
    # 4. VISUALIZATION
    # ----------------------------------------------------
    trades = trades.sort_index()
    trades['equity'] = trades['realized_pnl'].cumsum()
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Equity Curve
    
    ax1.plot(range(len(trades)), trades['equity'], color='blue', linewidth=2)
    ax1.axhline(0, color='black', linestyle='--')
    ax1.set_title(f"Timekeeper Equity Curve (PF: {profit_factor:.2f})")
    ax1.set_xlabel("Trade #")
    ax1.set_ylabel("Cumulative PnL")
    
    # Trade Outcome Distribution
    
    outcomes = ['Win' if x > 0 else 'Loss' for x in trades['realized_pnl']]
    outcome_counts = pd.Series(outcomes).value_counts()
    colors = ['#4CAF50', '#F44336'] # Green, Red
    ax2.pie(outcome_counts, labels=outcome_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title("Win/Loss Ratio")
    
    plt.tight_layout()
    plt.show()

# ==========================================
# MAIN LOOP
# ==========================================
if __name__ == "__main__":
    while True:
        try:
            user_input = input("\nEnter prediction CSV path:\n>> ").strip()
            # Remove quotes if user dragged/dropped file
            path = user_input.strip('"').strip("'")
            
            if not path: continue
            if path.lower() in ['q', 'exit']: break
            
            if os.path.exists(path):
                verify_strategy(path)
            else:
                print(f"❌ File not found: {path}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"❌ Critical Error: {e}")