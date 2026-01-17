"""
sequential_app.py - Orchestrates batch-by-batch execution of app.py
Uses Prefect for production-grade workflow management
"""

import pandas as pd
import numpy as np
from pathlib import Path
from prefect import flow, task
import subprocess
import json


@task(name="Load OHLCV Data")
def load_data(csv_path: str):
    """Load and prepare OHLCV data"""
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"✓ Loaded {len(df)} candles")
    return df


@task(name="Generate Batch")
def get_batch(df: pd.DataFrame, start_idx: int, window_size: int):
    """Extract single batch"""
    end_idx = start_idx + window_size
    
    if end_idx > len(df):
        return None
    
    batch = df.iloc[start_idx:end_idx].copy()
    
    batch_info = {
        'batch_id': start_idx + 1,
        'start_idx': start_idx,
        'end_idx': end_idx - 1,
        'timestamp_start': str(batch.iloc[0]['timestamp']),
        'timestamp_end': str(batch.iloc[-1]['timestamp']),
        'size': len(batch)
    }
    
    return batch, batch_info


@task(name="Run App.py")
def execute_app(batch: pd.DataFrame, batch_info: dict):
    """
    Execute app.py with current batch and wait for completion
    """
    # Save batch to temporary file
    batch_file = 'current_batch.csv'
    batch.to_csv(batch_file, index=False)
    
    # Save batch info
    with open('batch_info.json', 'w') as f:
        json.dump(batch_info, f)
    
    print(f"\n{'='*60}")
    print(f"▶️  Running app.py for Batch {batch_info['batch_id']}")
    print(f"   Time Range: {batch_info['timestamp_start']} → {batch_info['timestamp_end']}")
    print(f"{'='*60}\n")
    
    # Execute app.py and WAIT for completion
    result = subprocess.run(
        ['python', 'app.py'],
        capture_output=True,
        text=True,
        timeout=300  # 5 minute timeout
    )
    
    # Check if app.py completed successfully
    if result.returncode != 0:
        print(f"❌ app.py failed for batch {batch_info['batch_id']}")
        print(f"Error: {result.stderr}")
        return None
    
    print(f"✓ app.py completed for Batch {batch_info['batch_id']}")
    
    # Read result from app.py output
    try:
        # Assuming app.py saves result to result.json
        with open('result.json', 'r') as f:
            app_result = json.load(f)
        return app_result
    except:
        # Or parse from stdout
        return {'stdout': result.stdout, 'batch_id': batch_info['batch_id']}


@flow(name="OHLCV Batch Processing Pipeline")
def run_pipeline(
    csv_path: str = 'ohlcv_data.csv',
    window_size: int = 50,
    step_size: int = 1
):
    """
    Main pipeline: Process CSV batch-by-batch through app.py
    """
    # Load data
    df = load_data(csv_path)
    
    # Calculate total batches
    total_batches = (len(df) - window_size) // step_size + 1
    print(f"\n📊 Total batches to process: {total_batches}\n")
    
    # Store results
    all_results = []
    
    # Process each batch
    for i in range(0, len(df) - window_size + 1, step_size):
        batch, batch_info = get_batch(df, i, window_size)
        
        if batch is None:
            break
        
        # Execute app.py and WAIT for it to complete
        result = execute_app(batch, batch_info)
        
        if result:
            all_results.append(result)
        
        # Progress
        progress = (len(all_results) / total_batches) * 100
        print(f"📈 Progress: {progress:.1f}% ({len(all_results)}/{total_batches})")
    
    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('pipeline_results.csv', index=False)
    
    print(f"\n✅ Pipeline completed! Processed {len(all_results)} batches")
    print(f"📄 Results saved to: pipeline_results.csv")
    
    return all_results


if __name__ == "__main__":
    # Run the pipeline
    results = run_pipeline(
        csv_path='ohlcv_data.csv',
        window_size=50,
        step_size=1
    )