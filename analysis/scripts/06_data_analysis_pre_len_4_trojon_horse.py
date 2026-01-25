import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def analyze_prediction_file(file_path):
    filename = os.path.basename(file_path)
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return

    print("\n" + "="*60)
    print(f"🐴 TROJAN PREDICTION ANALYSIS")
    print("="*60)
    print(f"File: {filename}")
    print(f"Rows: {len(df):,}")

    # ----------------------------------------------------
    # 1. IDENTIFY COLUMNS
    # ----------------------------------------------------
    # In prediction files, the input context is in 'last_value_...'
    # Trojan Mapping:
    # last_value_open  = RSI Context
    # last_value_high  = Trend Dist Context
    # last_value_low   = Volatility Context
    # actual_close_1..4 = The Future Log Returns (The Truth)
    # predicted_close_1..4 = What the model guessed
    
    col_rsi = 'last_value_open'
    col_vol = 'last_value_low'
    
    if col_rsi not in df.columns:
        print("❌ Error: Column 'last_value_open' not found. Is this a prediction file?")
        return

    # ----------------------------------------------------
    # 2. CALCULATE 1-HOUR TREND ACCURACY
    # ----------------------------------------------------
    # Sum of 4 steps = Total 1-Hour Log Return
    actual_cols = [c for c in df.columns if 'actual_close_' in c]
    pred_cols = [c for c in df.columns if 'predicted_close_' in c]
    
    df['cumulative_actual'] = df[actual_cols].sum(axis=1)
    df['cumulative_pred'] = df[pred_cols].sum(axis=1)
    
    # Directional Accuracy (Sign Match)
    matches = (np.sign(df['cumulative_actual']) == np.sign(df['cumulative_pred']))
    trend_acc = matches.mean() * 100
    
    print(f"\n🎯 1-HOUR TREND ACCURACY: {trend_acc:.2f}%")
    
    # ----------------------------------------------------
    # 3. DID THE MODEL LEARN THE 'RSI' STRATEGY?
    # ----------------------------------------------------
    print("\n🔍 STRATEGY VALIDATION (Did it learn?)")
    
    # Check 1: RSI Mean Reversion
    # If RSI (last_value_open) was High (>0.7), did the model PREDICT Down?
    high_rsi_context = df[df[col_rsi] > 0.7]
    low_rsi_context = df[df[col_rsi] < 0.3]
    
    bearish_bets = (high_rsi_context['cumulative_pred'] < 0).mean() * 100
    bullish_bets = (low_rsi_context['cumulative_pred'] > 0).mean() * 100
    
    print(f"1. Overbought Handling (RSI > 0.7):")
    print(f"   - Model bet DOWN: {bearish_bets:.1f}% of the time.")
    if bearish_bets > 50:
        print("   ✅ GOOD. Model understands Overbought = Sell.")
    else:
        print("   ⚠️  WARNING. Model is chasing momentum at tops.")

    print(f"2. Oversold Handling (RSI < 0.3):")
    print(f"   - Model bet UP:   {bullish_bets:.1f}% of the time.")
    if bullish_bets > 50:
        print("   ✅ GOOD. Model understands Oversold = Buy.")
    else:
        print("   ⚠️  WARNING. Model is panic selling at bottoms.")

    # ----------------------------------------------------
    # 4. VISUALIZATION
    # ----------------------------------------------------
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2)

    # Plot A: RSI Context vs Model Prediction
    # We want a NEGATIVE correlation (High RSI -> Negative Prediction)
    ax1 = fig.add_subplot(gs[0, 0])
    sns.regplot(x=df[col_rsi], y=df['cumulative_pred'], ax=ax1, 
                scatter_kws={'alpha':0.1, 'color':'purple'}, line_kws={'color':'red'})
    ax1.set_title("Strategy Check: RSI Input vs Model Prediction")
    ax1.set_xlabel("Input RSI (last_value_open)")
    ax1.set_ylabel("Predicted 1H Return")
    ax1.axvline(0.7, color='red', linestyle='--')
    ax1.axvline(0.3, color='green', linestyle='--')

    # Plot B: Actual vs Predicted Trend
    ax2 = fig.add_subplot(gs[0, 1])
    sns.scatterplot(x=df['cumulative_actual'], y=df['cumulative_pred'], 
                    alpha=0.3, ax=ax2, color='blue')
    ax2.axhline(0, color='black')
    ax2.axvline(0, color='black')
    ax2.set_title(f"Accuracy Check: Actual vs Predicted (Acc: {trend_acc:.1f}%)")
    
    # Plot C: Error Distribution by RSI
    # Does it fail more when RSI is extreme?
    df['abs_error'] = (df['cumulative_actual'] - df['cumulative_pred']).abs()
    ax3 = fig.add_subplot(gs[1, :])
    sns.scatterplot(x=df[col_rsi], y=df['abs_error'], ax=ax3, alpha=0.3, color='gray')
    ax3.set_title("Risk Analysis: Prediction Error vs RSI Level")
    ax3.set_xlabel("RSI Level")
    ax3.set_ylabel("Absolute Error")

    plt.tight_layout()
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    while True:
        user_input = input("\nEnter path to PREDICTION CSV file:\n>> ").strip().strip('"').strip("'")
        
        if not user_input: continue
        if user_input.lower() in ['exit', 'q']: break
        
        if not os.path.exists(user_input):
            print("❌ Path not found.")
            continue

        analyze_prediction_file(user_input)

if __name__ == "__main__":
    main()