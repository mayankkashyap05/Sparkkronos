import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import sys
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Update this path to your exact file location
FILE_PATH = r"analysis\data\prediction_SOL_15min-model=kronos-mini-lookback=504-pred_len=1-temperature=0.9-top_p=0.98-sample_count=2.csv"

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def load_data(path):
    if not os.path.exists(path):
        print(f"CRITICAL: File not found at {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"Successfully loaded {len(df):,} rows.")
    return df

def get_market_regime(df):
    """
    Classifies each row into a market regime based on volatility.
    """
    # Calculate True Range or simple volatility
    df['volatility'] = df['last_value_high'] - df['last_value_low']
    # Rolling 50-period volatility percentile
    df['vol_percentile'] = df['volatility'].rolling(50).rank(pct=True)
    
    conditions = [
        (df['vol_percentile'] > 0.8),
        (df['vol_percentile'] < 0.2)
    ]
    choices = ['High Volatility', 'Low Volatility']
    df['regime'] = np.select(conditions, choices, default='Normal')
    return df

def safe_mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# ==========================================
# 3. MAIN ANALYSIS ENGINE
# ==========================================
def analyze_predictions():
    df = load_data(FILE_PATH)
    
    # -------------------------------------
    # A. PRE-PROCESSING
    # -------------------------------------
    # Focus on CLOSE price for deep financial analysis
    actual = df['actual_close_1']
    pred = df['predicted_close_1']
    prev = df['last_value_close']
    
    # Calculate Returns (Moves)
    df['actual_ret'] = (actual - prev) / prev
    df['pred_ret'] = (pred - prev) / prev
    
    # Directions (1 for UP, -1 for DOWN)
    df['actual_dir'] = np.sign(df['actual_ret'])
    df['pred_dir'] = np.sign(df['pred_ret'])
    
    # Residuals
    df['error'] = actual - pred
    df['abs_error'] = np.abs(df['error'])
    
    # Add Market Regimes
    df = get_market_regime(df)

    print("\n" + "="*60)
    print(f"📊 DEEP ANALYSIS REPORT | Rows: {len(df):,}")
    print("="*60)

    # -------------------------------------
    # B. GLOBAL METRICS
    # -------------------------------------
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = safe_mape(actual.values, pred.values)
    r2 = r2_score(actual, pred)
    dir_acc = accuracy_score(df['actual_dir'], df['pred_dir']) * 100
    
    # Lag Check
    true_corr = pred.corr(actual)
    lag_corr = pred.corr(prev)
    
    print(f"\n1. GLOBAL PERFORMANCE")
    print(f"   RMSE (Price Error):      {rmse:.4f}")
    print(f"   MAPE (Relative Error):   {mape:.2f}%")
    print(f"   R² Score (Fit):          {r2:.4f}")
    print(f"   Directional Accuracy:    {dir_acc:.2f}%")
    
    print(f"\n2. CHEAT DETECTION (Lag Test)")
    print(f"   Correlation to Target:   {true_corr:.4f}")
    print(f"   Correlation to Previous: {lag_corr:.4f}")
    if lag_corr > true_corr:
        print("   ⚠️ FAIL: Model is likely just copying the last known value.")
    else:
        print("   ✅ PASS: Model shows predictive signal beyond lag.")

    # -------------------------------------
    # C. SEGMENTED ANALYSIS (The "Deep" Part)
    # -------------------------------------
    print(f"\n3. PERFORMANCE BY REGIME (Where does it fail?)")
    regime_stats = df.groupby('regime').apply(
        lambda x: pd.Series({
            'RMSE': np.sqrt(mean_squared_error(x['actual_close_1'], x['predicted_close_1'])),
            'Dir Accuracy': accuracy_score(x['actual_dir'], x['pred_dir']) * 100,
            'Count': len(x)
        })
    )
    print(regime_stats)

    # -------------------------------------
    # D. VISUALIZATION DASHBOARD
    # -------------------------------------
    # We use a GridSpec layout for a professional dashboard look
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3)

    # Plot 1: Actual vs Predicted (Scatter)
    ax1 = fig.add_subplot(gs[0, 0])
    # Downsample for scatter plot if > 10k rows to keep rendering fast
    plot_data = df.sample(n=min(10000, len(df))) 
    sns.scatterplot(x=plot_data['actual_close_1'], y=plot_data['predicted_close_1'], alpha=0.1, ax=ax1, color='blue')
    min_val, max_val = plot_data['actual_close_1'].min(), plot_data['actual_close_1'].max()
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax1.set_title(f'Actual vs Predicted (Sample of {len(plot_data)})')

    # Plot 2: Directional Confusion Matrix
    ax2 = fig.add_subplot(gs[0, 1])
    cm = confusion_matrix(df['actual_dir'], df['pred_dir'], labels=[1, -1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, xticklabels=['Up', 'Down'], yticklabels=['Up', 'Down'])
    ax2.set_title('Directional Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')

    # Plot 3: Residual Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    sns.histplot(data=df, x='error', kde=True, ax=ax3, color='purple', bins=50)
    ax3.set_title('Error Distribution (Residuals)')
    ax3.set_xlabel('Error (Actual - Pred)')

    # Plot 4: Rolling Accuracy (Time Series)
    ax4 = fig.add_subplot(gs[1, :]) # Spans entire middle row
    window = 1000 # Rolling window of 1000 candles
    rolling_acc = (df['actual_dir'] == df['pred_dir']).rolling(window).mean() * 100
    ax4.plot(rolling_acc.to_numpy(), color='green', linewidth=1)
    ax4.axhline(50, color='red', linestyle='--', alpha=0.5)
    ax4.set_title(f'Rolling Directional Accuracy (Window={window})')
    ax4.set_ylabel('Accuracy %')

    # Plot 5: Time Series Snippet (Zoomed In)
    ax5 = fig.add_subplot(gs[2, :]) # Spans entire bottom row
    zoom_len = 200
    # Take the most recent 200 points
    recent_df = df.iloc[-zoom_len:]
    ax5.plot(recent_df.index, recent_df['actual_close_1'], label='Actual', color='black', alpha=0.7)
    ax5.plot(recent_df.index, recent_df['predicted_close_1'], label='Predicted', color='orange', linestyle='--')
    ax5.fill_between(recent_df.index, recent_df['actual_close_1'], recent_df['predicted_close_1'], color='gray', alpha=0.1)
    ax5.set_title(f'Zoomed View: Last {zoom_len} Periods')
    ax5.legend()

    plt.tight_layout()
    plt.show()

    print("\n✅ Analysis Complete. Dashboard generated.")

if __name__ == "__main__":
    analyze_predictions()