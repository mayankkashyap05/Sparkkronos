import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def analyze_aligned_prediction(file_path):
    filename = os.path.basename(file_path)
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return

    print("\n" + "="*60)
    print(f"🔋 ALIGNED TROJAN ANALYSIS (High Score = Buy Check)")
    print("="*60)
    print(f"File: {filename}")
    print(f"Rows: {len(df):,}")

    # ----------------------------------------------------
    # 1. IDENTIFY COLUMNS
    # ----------------------------------------------------
    # In Aligned Mode:
    # last_value_open = Inverted RSI Score (0.8 means Oversold/Buy)
    # last_value_high = Inverted SMA Score (High means Price is Low/Buy)
    
    col_score = 'last_value_open'
    
    if col_score not in df.columns:
        print("❌ Error: Column 'last_value_open' not found.")
        return

    # ----------------------------------------------------
    # 2. CALCULATE 1-HOUR TREND ACCURACY
    # ----------------------------------------------------
    actual_cols = [c for c in df.columns if 'actual_close_' in c]
    pred_cols = [c for c in df.columns if 'predicted_close_' in c]
    
    df['cumulative_actual'] = df[actual_cols].sum(axis=1)
    df['cumulative_pred'] = df[pred_cols].sum(axis=1)
    
    matches = (np.sign(df['cumulative_actual']) == np.sign(df['cumulative_pred']))
    trend_acc = matches.mean() * 100
    
    print(f"\n🎯 1-HOUR TREND ACCURACY: {trend_acc:.2f}%")
    
    # ----------------------------------------------------
    # 3. DID THE MODEL LEARN THE ALIGNED STRATEGY?
    # ----------------------------------------------------
    print("\n🔍 STRATEGY VALIDATION (Positive Correlation Check)")
    
    # Check 1: High Score (Oversold/Dip) -> Expect BUY (Positive Prediction)
    # Score > 0.7 means Real RSI was < 30.
    high_score_context = df[df[col_score] > 0.7]
    bullish_bets = (high_score_context['cumulative_pred'] > 0).mean() * 100
    
    print(f"1. High Score Handling (Dip Buying):")
    print(f"   - Input Score > 0.7 (Real RSI < 30)")
    print(f"   - Model bet UP:   {bullish_bets:.1f}% of the time.")
    
    if bullish_bets > 50:
        print("   ✅ GOOD. Model is buying the dip.")
    else:
        print("   ⚠️  FAIL. Model is still panic selling.")

    # Check 2: Low Score (Overbought/Top) -> Expect SELL (Negative Prediction)
    # Score < 0.3 means Real RSI was > 70.
    low_score_context = df[df[col_score] < 0.3]
    bearish_bets = (low_score_context['cumulative_pred'] < 0).mean() * 100
    
    print(f"2. Low Score Handling (Top Shorting):")
    print(f"   - Input Score < 0.3 (Real RSI > 70)")
    print(f"   - Model bet DOWN: {bearish_bets:.1f}% of the time.")
    
    if bearish_bets > 50:
        print("   ✅ GOOD. Model is shorting the top.")
    else:
        print("   ⚠️  FAIL. Model is chasing the pump.")

    # ----------------------------------------------------
    # 4. VISUALIZATION
    # ----------------------------------------------------
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2)

    # Plot A: Score vs Prediction (We want POSITIVE Correlation now)
    ax1 = fig.add_subplot(gs[0, 0])
    sns.regplot(x=df[col_score], y=df['cumulative_pred'], ax=ax1, 
                scatter_kws={'alpha':0.1, 'color':'green'}, line_kws={'color':'blue'})
    ax1.set_title("Strategy Check: Input Score vs Prediction")
    ax1.set_xlabel("Input Score (High = Buy)")
    ax1.set_ylabel("Predicted Return")
    # We want the blue line to go UP (Positive Slope)
    
    # Plot B: Accuracy Scatter
    ax2 = fig.add_subplot(gs[0, 1])
    sns.scatterplot(x=df['cumulative_actual'], y=df['cumulative_pred'], 
                    alpha=0.3, ax=ax2, color='blue')
    ax2.axhline(0, color='black')
    ax2.axvline(0, color='black')
    ax2.set_title(f"Accuracy Check (Acc: {trend_acc:.1f}%)")

    # Plot C: Error Distribution
    df['abs_error'] = (df['cumulative_actual'] - df['cumulative_pred']).abs()
    ax3 = fig.add_subplot(gs[1, :])
    sns.scatterplot(x=df[col_score], y=df['abs_error'], ax=ax3, alpha=0.3, color='gray')
    ax3.set_title("Error vs Input Score")
    ax3.set_xlabel("Input Score")

    plt.tight_layout()
    plt.show()
    print("\n✅ Analysis Complete. Look for a BLUE LINE going UP in the top-left chart.")

def main():
    while True:
        user_input = input("\nEnter path to ALIGNED PREDICTION CSV:\n>> ").strip().strip('"').strip("'")
        if not user_input: continue
        if user_input.lower() in ['exit', 'q']: break
        if not os.path.exists(user_input):
            print("❌ Path not found.")
            continue
        analyze_aligned_prediction(user_input)

if __name__ == "__main__":
    main()