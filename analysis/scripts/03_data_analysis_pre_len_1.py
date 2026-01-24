import pandas as pd
import numpy as np
import glob
import os

def calculate_metrics(df):
    """
    Calculates aggregate metrics across ALL rows in the CSV.
    """
    results = {}
    
    # --- 1. Data Integrity Check ---
    # We count how many rows (trades/predictions) are in this specific file
    results['Total_Trades'] = len(df)
    
    # --- 2. Directional Accuracy (The "Profit" Metric) ---
    # Logic: Did the model predict the correct move (Up or Down) relative to the PREVIOUS close?
    
    # Calculate the Actual Move: (Actual Future Close) - (The Last Known Close before prediction)
    actual_move = df['actual_close_1'] - df['last_value_close']
    
    # Calculate the Predicted Move: (Predicted Future Close) - (The Last Known Close)
    pred_move = df['predicted_close_1'] - df['last_value_close']
    
    # Check if the sign matches (e.g., both are positive or both are negative)
    # +1 means Correct Direction, 0 means Incorrect
    # We use a small epsilon for float comparison safety, but sign() is generally robust here.
    correct_direction = (np.sign(actual_move) == np.sign(pred_move))
    
    # Calculate percentage of correct distinct moves
    results['Directional_Accuracy_Pct'] = correct_direction.mean() * 100

    # --- 3. Error Metrics (Precision) ---
    df['error_close'] = df['actual_close_1'] - df['predicted_close_1']
    
    # RMSE: Penalizes large errors heavily (good for safety)
    results['RMSE_Close'] = np.sqrt(np.mean(df['error_close'] ** 2))
    
    # MAE: The average 'miss' distance
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

    print(f"📂 Found {len(all_files)} files. Analyzing hundreds of rows per file...")
    
    analysis_data = []

    # 2. Process Each File
    for i, filename in enumerate(all_files):
        try:
            df = pd.read_csv(filename)
            
            # Skip empty files
            if df.empty:
                continue

            # Check for required columns to avoid crashing on bad files
            required_cols = ['last_value_close', 'actual_close_1', 'predicted_close_1']
            if not all(col in df.columns for col in required_cols):
                print(f"⚠️  Skipping {os.path.basename(filename)}: Missing required columns.")
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
    # Primary Sort: Directional Accuracy (Highest first) -> We want to be right more often.
    # Secondary Sort: RMSE (Lowest first) -> When we are wrong, we want to be wrong by a small amount.
    ranked_df = results_df.sort_values(
        by=['Directional_Accuracy_Pct', 'RMSE_Close'], 
        ascending=[False, True]
    )

    # 4. Display Results
    print("\n" + "="*80)
    print("🏆 TOP 5 BEST CONFIGURATIONS (Sorted by Directional Accuracy)")
    print("="*80)
    
    # Display clean columns
    display_cols = ['filename', 'Directional_Accuracy_Pct', 'RMSE_Close', 'MAE_Close', 'Total_Trades']
    print(ranked_df[display_cols].head(5).to_string(index=False))

    # 5. Save Report
    output_filename = 'analysis/best_model_results.csv'
    ranked_df.to_csv(output_filename, index=False)
    print(f"\n✅ Full analysis saved to: {output_filename}")

    # 6. Winner Details
    winner = ranked_df.iloc[0]
    print("\n" + "-"*40)
    print(f"WINNER: {winner['filename']}")
    print("-" * 40)
    print(f"It analyzed {int(winner['Total_Trades'])} prediction lines.")
    print(f"It correctly predicted price direction {winner['Directional_Accuracy_Pct']:.2f}% of the time.")
    print(f"Its average error (MAE) was {winner['MAE_Close']:.4f}.")

if __name__ == "__main__":
    main()