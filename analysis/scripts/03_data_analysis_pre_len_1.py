import pandas as pd
import numpy as np
import glob
import os

def calculate_metrics(df):
    """
    Calculates error metrics and directional accuracy for a single CSV.
    Focuses on 'Close' price for the primary ranking, but you can adapt for others.
    """
    results = {}
    
    # --- 1. Error Metrics (Lower is Better) ---
    # Calculate errors for Close price
    df['error_close'] = df['actual_close_1'] - df['predicted_close_1']
    
    # RMSE (Root Mean Squared Error)
    results['RMSE_Close'] = np.sqrt(np.mean(df['error_close'] ** 2))
    
    # MAE (Mean Absolute Error)
    results['MAE_Close'] = np.mean(np.abs(df['error_close']))
    
    # MAPE (Mean Absolute Percentage Error) - careful with zeros
    results['MAPE_Close'] = np.mean(np.abs(df['error_close'] / df['actual_close_1'])) * 100

    # --- 2. Directional Accuracy (Higher is Better) ---
    # Did the model correctly predict if price would go UP or DOWN from the last close?
    
    # Actual move: Actual Close - Last Known Close
    actual_move = df['actual_close_1'] - df['last_value_close']
    
    # Predicted move: Predicted Close - Last Known Close
    pred_move = df['predicted_close_1'] - df['last_value_close']
    
    # Check if signs match (both positive or both negative)
    # We use np.sign() to get -1, 0, or 1. 
    # (actual_move * pred_move) > 0 implies same direction.
    correct_direction = (np.sign(actual_move) == np.sign(pred_move))
    
    results['Directional_Accuracy_Pct'] = correct_direction.mean() * 100
    
    return results

def main():
    # Define your relative path
    folder_path = os.path.join('analysis', 'data')
    all_files = glob.glob(os.path.join(folder_path, '*.csv'))
    
    if not all_files:
        print(f"No CSV files found in {folder_path}")
        return

    print(f"Found {len(all_files)} files. Analyzing...")
    
    analysis_data = []

    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            
            # precise matching of your column names
            required_cols = ['last_value_close', 'actual_close_1', 'predicted_close_1']
            if not all(col in df.columns for col in required_cols):
                print(f"Skipping {os.path.basename(filename)}: Missing columns")
                continue
                
            metrics = calculate_metrics(df)
            metrics['filename'] = os.path.basename(filename)
            analysis_data.append(metrics)
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # Create a DataFrame of all results
    results_df = pd.DataFrame(analysis_data)
    
    if results_df.empty:
        print("No valid data extracted.")
        return

    # --- Ranking Strategy ---
    
    # Sort by Directional Accuracy (Descending) first, then RMSE (Ascending)
    # We want High Accuracy and Low Error.
    ranked_df = results_df.sort_values(
        by=['Directional_Accuracy_Pct', 'RMSE_Close'], 
        ascending=[False, True]
    )

    print("\n" + "="*50)
    print("TOP 3 MODELS (Best Directional Accuracy & Lowest Error)")
    print("="*50)
    print(ranked_df[['filename', 'Directional_Accuracy_Pct', 'RMSE_Close', 'MAE_Close']].head(3).to_string(index=False))

    # Save full report
    output_file = 'model_comparison_report.csv'
    ranked_df.to_csv(output_file, index=False)
    print(f"\nFull report saved to: {output_file}")

    # Identify the absolute winner
    winner = ranked_df.iloc[0]
    print(f"\n🏆 BEST CONFIGURATION: {winner['filename']}")
    print(f"   Accuracy: {winner['Directional_Accuracy_Pct']:.2f}%")
    print(f"   RMSE:     {winner['RMSE_Close']:.4f}")

if __name__ == "__main__":
    main()