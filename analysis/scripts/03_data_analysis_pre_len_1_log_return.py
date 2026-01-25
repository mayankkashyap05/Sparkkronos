import pandas as pd
import numpy as np
import glob
import os

def calculate_metrics(df):
    """
    Calculates aggregate metrics for LOG RETURN data.
    """
    results = {}
    
    # --- 1. Data Integrity Check ---
    results['Total_Trades'] = len(df)
    
    # --- 2. Directional Accuracy (The Correct Logic for Returns) ---
    # Since the data is ALREADY a return (change from previous price):
    # Positive Value = Price Went UP
    # Negative Value = Price Went DOWN
    
    # We simply compare the sign of the Actual Return vs Predicted Return
    # +1 means they match (e.g., both UP or both DOWN)
    # 0 means they mismatch
    
    # Use sign() to get -1, 0, or 1
    actual_sign = np.sign(df['actual_close_1'])
    pred_sign = np.sign(df['predicted_close_1'])
    
    # Check where signs are equal (excluding flat 0 moves if necessary, but usually fine)
    correct_direction = (actual_sign == pred_sign)
    
    # Calculate percentage
    results['Directional_Accuracy_Pct'] = correct_direction.mean() * 100

    # --- 3. Error Metrics (Precision) ---
    # Simple difference between the Actual Return and Predicted Return
    df['error_close'] = df['actual_close_1'] - df['predicted_close_1']
    
    # RMSE: Root Mean Squared Error
    results['RMSE_Close'] = np.sqrt(np.mean(df['error_close'] ** 2))
    
    # MAE: Mean Absolute Error
    results['MAE_Close'] = np.mean(np.abs(df['error_close']))

    return results

def main():
    # 1. Setup Path
    # Using relative path as requested: analysis/data
    folder_path = os.path.join('analysis', 'data')
    
    # Get all csv files
    all_files = glob.glob(os.path.join(folder_path, '*.csv'))
    
    if not all_files:
        print(f"❌ No CSV files found in: {folder_path}")
        print("Please check that the folder exists and contains .csv files.")
        return

    print(f"📂 Found {len(all_files)} files. Analyzing Log Returns...")
    
    analysis_data = []

    # 2. Process Each File
    for i, filename in enumerate(all_files):
        try:
            df = pd.read_csv(filename)
            
            # Skip empty files
            if df.empty:
                continue

            # Check for required columns
            required_cols = ['actual_close_1', 'predicted_close_1']
            if not all(col in df.columns for col in required_cols):
                # Only warn if it's not a summary file or something else
                if "prediction_" in filename:
                    print(f"⚠️  Skipping {os.path.basename(filename)}: Missing columns.")
                continue
                
            metrics = calculate_metrics(df)
            metrics['filename'] = os.path.basename(filename)
            analysis_data.append(metrics)
            
            # Optional: Print progress every 100 files
            if (i + 1) % 100 == 0:
                print(f"   Processed {i + 1} files...")
            
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")

    if not analysis_data:
        print("No valid data could be extracted.")
        return

    # 3. Create Ranking DataFrame
    results_df = pd.DataFrame(analysis_data)
    
    # --- RANKING LOGIC ---
    # Sort by Directional Accuracy (Highest first)
    ranked_df = results_df.sort_values(
        by=['Directional_Accuracy_Pct', 'RMSE_Close'], 
        ascending=[False, True]
    )

    # 4. Display Results
    print("\n" + "="*80)
    print("🏆 TOP 5 BEST CONFIGURATIONS (Real Log-Return Accuracy)")
    print("="*80)
    
    # Display clean columns
    display_cols = ['filename', 'Directional_Accuracy_Pct', 'RMSE_Close', 'MAE_Close', 'Total_Trades']
    print(ranked_df[display_cols].head(5).to_string(index=False))

    # 5. Save Report
    output_filename = 'analysis/best_log_return_results.csv'
    ranked_df.to_csv(output_filename, index=False)
    print(f"\n✅ Full analysis saved to: {output_filename}")

    # 6. Winner Details
    if not ranked_df.empty:
        winner = ranked_df.iloc[0]
        print("\n" + "-"*40)
        print(f"WINNER: {winner['filename']}")
        print("-" * 40)
        print(f"It analyzed {int(winner['Total_Trades'])} trades.")
        print(f"True Accuracy: {winner['Directional_Accuracy_Pct']:.2f}%")
        print(f"Average Error: {winner['MAE_Close']:.6f}")

if __name__ == "__main__":
    main()