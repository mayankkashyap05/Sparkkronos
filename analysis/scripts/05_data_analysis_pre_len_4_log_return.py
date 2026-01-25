import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import sys

# ==========================================
# 1. METRICS ENGINE
# ==========================================
def calculate_metrics(actual, predicted, label="Step"):
    """
    Calculates metrics for Log Return data (Sign-based Accuracy).
    """
    # Remove NaNs to prevent crashes
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    actual = actual[mask]
    predicted = predicted[mask]

    if len(actual) == 0:
        return {
            "Step": label,
            "Accuracy (%)": 0.0,
            "RMSE": 0.0,
            "MAE": 0.0,
            "Correlation": 0.0
        }

    # 1. Directional Accuracy (Sign Match)
    act_sign = np.sign(actual)
    pred_sign = np.sign(predicted)
    
    matches = (act_sign == pred_sign)
    dir_acc = matches.mean() * 100

    # 2. Magnitude Error
    error = actual - predicted
    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))
    
    # 3. Correlation
    if len(actual) > 1:
        corr = np.corrcoef(actual, predicted)[0, 1]
        if np.isnan(corr): corr = 0
    else:
        corr = 0

    return {
        "Step": label,
        "Accuracy (%)": dir_acc,
        "RMSE": rmse,
        "MAE": mae,
        "Correlation": corr
    }

# ==========================================
# 2. SINGLE FILE ANALYSIS
# ==========================================
def analyze_single_file(file_path, show_plots=True):
    filename = os.path.basename(file_path)
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return None

    # Check required columns
    if 'actual_close_1' not in df.columns:
        # print(f"⚠️ Skipping {filename}: Missing 'actual_close_1'") # Optional noise reduction
        return None

    print("\n" + "="*60)
    print(f"🔬 ANALYZING: {filename}")
    print("="*60)
    print(f"✅ Loaded {len(df):,} rows.")

    # ----------------------------------------------------
    # A. STEP-BY-STEP ANALYSIS
    # ----------------------------------------------------
    results = []
    steps = [1, 2, 3, 4]
    
    for i in steps:
        act_col = f'actual_close_{i}'
        pred_col = f'predicted_close_{i}'
        
        if act_col in df.columns and pred_col in df.columns:
            metrics = calculate_metrics(df[act_col], df[pred_col], label=f"T+{i}")
            results.append(metrics)

    if not results:
        print("❌ No step data found.")
        return None

    # ----------------------------------------------------
    # B. CUMULATIVE (TREND) ANALYSIS
    # ----------------------------------------------------
    # Sum Actuals and Predictions to get the Trend
    actual_cols = [f'actual_close_{i}' for i in steps if f'actual_close_{i}' in df.columns]
    pred_cols = [f'predicted_close_{i}' for i in steps if f'predicted_close_{i}' in df.columns]
    
    df['cumulative_actual'] = df[actual_cols].sum(axis=1)
    df['cumulative_pred'] = df[pred_cols].sum(axis=1)
    
    trend_metrics = calculate_metrics(df['cumulative_actual'], df['cumulative_pred'], label="FULL HOUR (Trend)")
    results.append(trend_metrics)

    # ----------------------------------------------------
    # C. DISPLAY TEXT SUMMARY
    # ----------------------------------------------------
    results_df = pd.DataFrame(results)
    print("\n--- Performance Summary ---")
    print(results_df.round(4).to_string(index=False))

    trend_acc = trend_metrics['Accuracy (%)']
    print("\n" + "-"*60)
    print(f"🎯 1-HOUR TREND ACCURACY: {trend_acc:.2f}%")
    
    if trend_acc > 52.0:
        print("✅ SUCCESS: High predictive power!")
    elif trend_acc < 48.0:
        print("⚠️  INVERT: Consistent failure detected (Fade Opportunity).")
    else:
        print("❌ NOISE: Near 50% (Random).")
    print("-" * 60)

    # ----------------------------------------------------
    # D. VISUALIZATION (Only if requested)
    # ----------------------------------------------------
    if show_plots:
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3)

        # 1. Accuracy per Step
        ax1 = fig.add_subplot(gs[0, 0])
        # Fix: Added hue=x and legend=False to silence warning
        sns.barplot(x='Step', y='Accuracy (%)', hue='Step', data=results_df, ax=ax1, palette='viridis', legend=False)
        ax1.axhline(50, color='red', linestyle='--', label='Random (50%)')
        ax1.set_ylim(40, 60)
        ax1.set_title(f"Accuracy by Step\n{filename}", fontsize=10)

        # 2. Cumulative Scatter
        ax2 = fig.add_subplot(gs[0, 1])
        plot_data = df.sample(n=min(5000, len(df)))
        sns.scatterplot(x=plot_data['cumulative_actual'], y=plot_data['cumulative_pred'], 
                        alpha=0.3, ax=ax2, color='blue')
        lims = [min(ax2.get_xlim()), max(ax2.get_xlim())]
        ax2.plot(lims, lims, 'r--')
        ax2.set_title("Actual vs Predicted Trend")
        ax2.set_xlabel("Actual 1H Return")
        ax2.set_ylabel("Predicted 1H Return")

        # 3. Simulated PnL
        ax3 = fig.add_subplot(gs[0, 2])
        df['trend_signal'] = np.sign(df['cumulative_pred'])
        df['trend_pnl'] = df['trend_signal'] * df['cumulative_actual']
        df['cumulative_equity'] = df['trend_pnl'].cumsum()
        ax3.plot(df['cumulative_equity'].to_numpy(), color='green')
        ax3.set_title("Simulated PnL (No Fees)")
        ax3.axhline(0, color='black')

        # 4. Volatility Regime
        ax4 = fig.add_subplot(gs[1, :])
        df['abs_move'] = df['cumulative_actual'].abs()
        # Handle case with too few rows for quantiles
        if len(df) > 10:
            quantiles = df['abs_move'].quantile([0.33, 0.66])
            def get_regime(x):
                if x < quantiles[0.33]: return 'Low Vol'
                if x < quantiles[0.66]: return 'Med Vol'
                return 'High Vol'
            df['regime'] = df['abs_move'].apply(get_regime)
            
            # Fix: Explicit column selection to silence groupby warning
            regime_acc = df.groupby('regime')[['cumulative_actual', 'cumulative_pred']].apply(
                lambda x: (np.sign(x['cumulative_actual']) == np.sign(x['cumulative_pred'])).mean() * 100
            ).reset_index(name='Accuracy')
            
            sns.barplot(x='regime', y='Accuracy', hue='regime', data=regime_acc, ax=ax4, 
                        order=['Low Vol', 'Med Vol', 'High Vol'], palette='rocket', legend=False)
            ax4.axhline(50, color='red', linestyle='--')
            ax4.set_ylim(40, 60)
            ax4.set_title("Accuracy by Volatility Regime")

        plt.tight_layout()
        plt.show()

    return {
        "Filename": filename,
        "Trend_Accuracy": trend_acc,
        "Rows": len(df)
    }

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def main():
    while True:
        user_input = input("\nEnter path to CSV file OR Folder:\n>> ").strip().strip('"').strip("'")
        
        if not user_input: continue
        if user_input.lower() in ['exit', 'q']: break
        
        if not os.path.exists(user_input):
            print("❌ Path not found.")
            continue

        # Check if directory or file
        if os.path.isdir(user_input):
            # --- BATCH MODE ---
            print(f"\n📂 Scanning folder: {user_input}")
            all_files = glob.glob(os.path.join(user_input, "*.csv"))
            if not all_files:
                print("❌ No CSV files found in folder.")
                continue
                
            print(f"Found {len(all_files)} files. analyzing...")
            
            summary_list = []
            for f in all_files:
                # Don't show plots for batch mode, just gather data
                res = analyze_single_file(f, show_plots=False)
                if res:
                    summary_list.append(res)
            
            if summary_list:
                summary_df = pd.DataFrame(summary_list)
                summary_df = summary_df.sort_values(by="Trend_Accuracy", ascending=False)
                
                print("\n" + "="*60)
                print("🏆 FOLDER LEADERBOARD (Sorted by Trend Accuracy)")
                print("="*60)
                print(summary_df.to_string(index=False))
                print("\n✅ Batch analysis complete.")
                
        else:
            # --- SINGLE FILE MODE ---
            analyze_single_file(user_input, show_plots=True)

if __name__ == "__main__":
    main()