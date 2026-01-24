import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import sys
import os

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def load_data(path):
    # Strip quotes if user pasted them
    clean_path = path.strip().strip('"').strip("'")
    
    if not os.path.exists(clean_path):
        print(f"\n❌ CRITICAL: File not found at: {clean_path}")
        return None
        
    try:
        df = pd.read_csv(clean_path)
        print(f"✅ Successfully loaded {len(df):,} rows.")
        return df
    except Exception as e:
        print(f"\n❌ Error reading CSV: {e}")
        return None

def get_market_regime(df):
    """
    Classifies volatility regimes based on the MAGNITUDE of returns.
    """
    # For Log Returns, volatility is just the absolute value of the return (magnitude)
    # We use 'last_value_close' (previous return) to define the regime for the prediction
    df['volatility'] = df['last_value_close'].abs()
    
    # Rolling 50-period volatility percentile
    df['vol_percentile'] = df['volatility'].rolling(50).rank(pct=True)
    
    conditions = [
        (df['vol_percentile'] > 0.8),
        (df['vol_percentile'] < 0.2)
    ]
    choices = ['High Volatility', 'Low Volatility']
    df['regime'] = np.select(conditions, choices, default='Normal')
    return df

# ==========================================
# 2. MAIN ANALYSIS ENGINE
# ==========================================
def analyze_predictions(file_path):
    df = load_data(file_path)
    if df is None:
        return
    
    # -------------------------------------
    # A. PRE-PROCESSING
    # -------------------------------------
    # Map columns for clarity
    # NOTE: Assuming the file contains LOG RETURNS as calculated previously
    actual = df['actual_close_1']  # Actual Log Return
    pred = df['predicted_close_1'] # Predicted Log Return
    prev = df['last_value_close']  # Previous Log Return
    
    # Directions (1 for UP, -1 for DOWN)
    # Since data is returns, positive = UP, negative = DOWN
    df['actual_dir'] = np.sign(actual)
    df['pred_dir'] = np.sign(pred)
    
    # Residuals
    df['error'] = actual - pred
    
    # Add Market Regimes
    df = get_market_regime(df)

    print("\n" + "="*60)
    print(f"📊 LOG-RETURN ANALYSIS REPORT | Rows: {len(df):,}")
    print("="*60)

    # -------------------------------------
    # B. GLOBAL METRICS (Adapted for Returns)
    # -------------------------------------
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = np.mean(np.abs(actual - pred)) # MAE is better for small numbers than MAPE
    r2 = r2_score(actual, pred)
    dir_acc = accuracy_score(df['actual_dir'], df['pred_dir']) * 100
    
    # --- MOMENTUM CHECK (The "Cheat" Test Redefined) ---
    # Correlation with Target (Ideally High)
    true_corr = pred.corr(actual)
    # Correlation with Previous Return (Momentum)
    momentum_corr = pred.corr(prev)
    
    print(f"\n1. GLOBAL PERFORMANCE")
    print(f"   RMSE (Return Error):     {rmse:.6f}")
    print(f"   MAE  (Abs Error):        {mae:.6f}")
    print(f"   R² Score (Fit):          {r2:.4f}")
    print(f"   Directional Accuracy:    {dir_acc:.2f}%  <-- CRITICAL METRIC")
    
    print(f"\n2. STRATEGY ANALYSIS (Momentum vs Predictive)")
    print(f"   Correlation to Target:   {true_corr:.4f}")
    print(f"   Correlation to Past:     {momentum_corr:.4f}")
    
    if momentum_corr > 0.5:
        print("   ℹ️ NOTE: High correlation to past returns. Model is effectively a Momentum Strategy.")
        print("      (It bets that recent moves will continue). This is valid, not cheating.")
    elif momentum_corr < -0.5:
        print("   ℹ️ NOTE: High negative correlation. Model is a Mean Reversion Strategy.")
        print("      (It bets the price will snap back).")
    else:
        print("   ✅ NOTE: Low correlation to past. Model is finding novel signals.")

    # --- SIMULATED TRADING PnL ---
    # Simple Strategy: If Pred > 0, Long. If Pred < 0, Short.
    # Return = Sign(Pred) * Actual_Return
    df['strategy_pnl'] = df['pred_dir'] * actual
    cumulative_return = df['strategy_pnl'].cumsum()
    
    total_pnl = cumulative_return.iloc[-1]
    
    # Avoid division by zero for Sharpe
    std_dev = df['strategy_pnl'].std()
    if std_dev == 0:
        sharpe = 0
    else:
        sharpe = df['strategy_pnl'].mean() / std_dev * np.sqrt(252*96) # Annualized (approx 15m candles)

    print(f"\n3. SIMULATED TRADING RESULTS (No Fees)")
    print(f"   Total Cumulative Log Return: {total_pnl:.4f}")
    print(f"   Sharpe Ratio (Approx):       {sharpe:.2f}")

    # -------------------------------------
    # C. SEGMENTED ANALYSIS
    # -------------------------------------
    print(f"\n4. PERFORMANCE BY REGIME")
    try:
        # Using the new non-deprecated method
        regime_stats = df.groupby('regime')[['actual_close_1', 'predicted_close_1', 'actual_dir', 'pred_dir']].apply(
            lambda x: pd.Series({
                'RMSE': np.sqrt(mean_squared_error(x['actual_close_1'], x['predicted_close_1'])),
                'Dir Accuracy': accuracy_score(x['actual_dir'], x['pred_dir']) * 100,
                'Count': len(x)
            })
        )
        print(regime_stats)
    except Exception as e:
        print(f"Could not calculate regime stats (likely too few rows): {e}")

    # -------------------------------------
    # D. VISUALIZATION DASHBOARD
    # -------------------------------------
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3)

    # Plot 1: Actual vs Predicted Returns (Scatter)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_data = df.sample(n=min(10000, len(df))) 
    sns.scatterplot(x=plot_data['actual_close_1'], y=plot_data['predicted_close_1'], alpha=0.3, ax=ax1, color='blue')
    # Perfect fit line
    lims = [
        np.min([ax1.get_xlim(), ax1.get_ylim()]),
        np.max([ax1.get_xlim(), ax1.get_ylim()]),
    ]
    ax1.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
    ax1.set_title(f'Predicted vs Actual Returns')
    ax1.set_xlabel('Actual Log Return')
    ax1.set_ylabel('Predicted Log Return')

    # Plot 2: Cumulative PnL Curve
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(np.asarray(cumulative_return), color='green')
    ax2.set_title('Cumulative Strategy Return (No Fees)')
    ax2.set_ylabel('Log Return')
    ax2.axhline(0, color='black', linewidth=1)

    # Plot 3: Residual Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    sns.histplot(data=df, x='error', kde=True, ax=ax3, color='purple', bins=50)
    ax3.set_title('Error Distribution')
    ax3.set_xlabel('Error (Actual - Pred)')

    # Plot 4: Rolling Accuracy
    ax4 = fig.add_subplot(gs[1, :]) 
    window = max(20, int(len(df)/20)) # Adaptive window, min 20
    rolling_acc = (df['actual_dir'] == df['pred_dir']).rolling(window).mean() * 100
    ax4.plot(rolling_acc.to_numpy(), color='blue', linewidth=1)
    ax4.axhline(50, color='red', linestyle='--', alpha=0.5)
    ax4.set_title(f'Rolling Directional Accuracy (Window={window})')
    ax4.set_ylabel('Accuracy %')

    # Plot 5: Zoomed In Time Series (Returns)
    ax5 = fig.add_subplot(gs[2, :]) 
    zoom_len = min(100, len(df))
    recent_df = df.iloc[-zoom_len:]
    ax5.plot(range(len(recent_df)), recent_df['actual_close_1'], label='Actual Return', color='black', alpha=0.6)
    ax5.plot(range(len(recent_df)), recent_df['predicted_close_1'], label='Predicted Return', color='orange', linestyle='--')
    ax5.set_title(f'Zoomed View: Last {zoom_len} Returns')
    ax5.legend()

    plt.tight_layout()
    plt.show()

    print("\n✅ Log-Return Analysis Complete.")

# ==========================================
# 3. INTERACTIVE ENTRY POINT
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("LOG RETURN ANALYSIS TOOL")
    print("="*60)
    
    while True:
        user_input = input("\nEnter the path to your PREDICTION CSV file:\n>> ")
        
        # Allow exit
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Exiting...")
            break
            
        if not user_input.strip():
            continue
            
        analyze_predictions(user_input)
        
        # Ask to run again or exit
        again = input("\nAnalyze another file? (y/n): ").lower()
        if again != 'y':
            break