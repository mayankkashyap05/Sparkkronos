"""
Rolling Window Sequential Processor
====================================

Simple and robust batch-by-batch processing for real-world streaming.

Pipeline:
    DataLoader → Sequential Batcher → App.py (one batch at a time)

Features:
- Rolling window generation
- One batch processed completely before next
- Progress tracking
- Result aggregation
- Error handling per batch
"""

import pandas as pd
from typing import Iterator, Dict, Any, Optional, Callable, List
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class WindowConfig:
    """Rolling window configuration"""
    window_size: int = 50
    step_size: int = 1
    
    def __post_init__(self):
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if self.step_size < 1:
            raise ValueError("step_size must be >= 1")


class SequentialProcessor:
    """
    Sequential rolling window processor.
    
    Generates batches one-by-one and processes each completely
    before moving to the next.
    """
    
    def __init__(self, config: Optional[WindowConfig] = None):
        """
        Initialize processor.
        
        Args:
            config: Window configuration (default: window_size=50, step_size=1)
        """
        self.config = config or WindowConfig()
        self.total_batches = 0
        self.current_batch = 0
        self.results = []
    
    def calculate_total_batches(self, data_length: int) -> int:
        """Calculate total number of batches"""
        if data_length < self.config.window_size:
            return 0
        return (data_length - self.config.window_size) // self.config.step_size + 1
    
    def generate_batches(self, df: pd.DataFrame) -> Iterator[Dict[str, Any]]:
        """
        Generate rolling window batches from DataFrame.
        
        Args:
            df: Input DataFrame with datetime index
            
        Yields:
            Dictionary with batch information:
                - batch_number: Current batch (1-indexed)
                - total_batches: Total batches
                - data: Batch DataFrame
                - start_idx: Start index in original data
                - end_idx: End index in original data
                - start_time: First timestamp in batch
                - end_time: Last timestamp in batch
        """
        # Validate
        if len(df) < self.config.window_size:
            raise ValueError(
                f"Data has {len(df)} rows but window_size is {self.config.window_size}"
            )
        
        # Calculate total batches
        self.total_batches = self.calculate_total_batches(len(df))
        
        # Generate batches
        start_idx = 0
        batch_num = 1
        
        while start_idx + self.config.window_size <= len(df):
            end_idx = start_idx + self.config.window_size
            
            # Extract batch (copy to avoid reference issues)
            batch_data = df.iloc[start_idx:end_idx].copy()
            
            # Get timestamps
            start_time = batch_data.index[0] if isinstance(batch_data.index, pd.DatetimeIndex) else None
            end_time = batch_data.index[-1] if isinstance(batch_data.index, pd.DatetimeIndex) else None
            
            yield {
                'batch_number': batch_num,
                'total_batches': self.total_batches,
                'data': batch_data,
                'start_idx': start_idx,
                'end_idx': end_idx - 1,
                'start_time': start_time,
                'end_time': end_time,
                'window_size': len(batch_data)
            }
            
            # Move to next window
            start_idx += self.config.step_size
            batch_num += 1
    
    def process_batches(
        self,
        df: pd.DataFrame,
        predict_func: Callable,
        **predict_kwargs
    ) -> Iterator[Dict[str, Any]]:
        """
        Process batches through prediction pipeline.
        
        Each batch is FULLY processed before the next begins.
        
        Args:
            df: Input DataFrame (from DataLoader)
            predict_func: Prediction function from app.py
            **predict_kwargs: Arguments for predict function (lookback, pred_len, etc.)
            
        Yields:
            Dictionary with batch results:
                - batch_info: Batch metadata
                - predictions: Prediction results
                - status: 'success' or 'failed'
                - error: Error message if failed
        """
        for batch in self.generate_batches(df):
            batch_num = batch['batch_number']
            total = batch['total_batches']
            
            print(f"\n{'='*70}")
            print(f"  BATCH {batch_num}/{total}")
            print(f"{'='*70}")
            print(f"  Rows: {batch['start_idx']} - {batch['end_idx']}")
            if batch['start_time']:
                print(f"  Time: {batch['start_time']} → {batch['end_time']}")
            print(f"{'='*70}")
            
            try:
                # Process this batch through app.py
                predictions, actuals, pred_df = predict_func(
                    data=batch['data'],
                    **predict_kwargs
                )
                
                result = {
                    'batch_info': {
                        'batch_number': batch_num,
                        'total_batches': total,
                        'start_idx': batch['start_idx'],
                        'end_idx': batch['end_idx'],
                        'start_time': str(batch['start_time']) if batch['start_time'] else None,
                        'end_time': str(batch['end_time']) if batch['end_time'] else None,
                    },
                    'predictions': predictions,
                    'actuals': actuals,
                    'pred_df': pred_df,
                    'status': 'success',
                    'error': None
                }
                
                print(f"\n  ✓ Batch {batch_num} completed successfully")
                print(f"  Generated {len(predictions)} predictions\n")
                
            except Exception as e:
                result = {
                    'batch_info': {
                        'batch_number': batch_num,
                        'total_batches': total,
                    },
                    'predictions': [],
                    'actuals': [],
                    'pred_df': None,
                    'status': 'failed',
                    'error': str(e)
                }
                
                print(f"\n  ✗ Batch {batch_num} failed: {e}\n")
            
            # Store result
            self.results.append(result)
            
            # Yield result (pipeline waits here for next iteration)
            yield result
    
    def process_in_chunks(
        self,
        df: pd.DataFrame,
        predict_batch_func: Callable,
        batch_size: int = 32,
        **predict_kwargs
    ) -> Iterator[List[Dict[str, Any]]]:
        """
        Process windows in parallel chunks (batches) for GPU acceleration.
        
        Args:
            df: Input DataFrame
            predict_batch_func: The 'predict_batch_windows' function from app.py
            batch_size: Number of windows to send to GPU at once
            **predict_kwargs: Arguments for prediction (lookback, pred_len, etc.)
            
        Yields:
            List of results for each chunk
        """
        batch_accumulator = []
        
        # Reuse existing generator
        for window_info in self.generate_batches(df):
            batch_accumulator.append(window_info)
            
            # When bucket is full, process
            if len(batch_accumulator) >= batch_size:
                print(f"\n[GPU] Processing chunk of {len(batch_accumulator)} windows...")
                
                try:
                    # Call the batch function
                    results = predict_batch_func(
                        windows_list=batch_accumulator,
                        **predict_kwargs
                    )
                    
                    # Store results internally
                    for res in results:
                        self.results.append(res)
                    
                    # Yield results to caller
                    yield results
                    
                except Exception as e:
                    print(f"Chunk failed: {e}")
                
                # Reset accumulator
                batch_accumulator = []
        
        # Process remaining windows (final partial chunk)
        if batch_accumulator:
            print(f"\n[GPU] Processing final chunk of {len(batch_accumulator)} windows...")
            try:
                results = predict_batch_func(
                    windows_list=batch_accumulator,
                    **predict_kwargs
                )
                for res in results:
                    self.results.append(res)
                yield results
            except Exception as e:
                print(f"Final chunk failed: {e}")

    def get_all_results(self) -> list:
        """Get all processed results"""
        return self.results
    
    def get_successful_results(self) -> list:
        """Get only successful results"""
        return [r for r in self.results if r['status'] == 'success']
    
    def get_failed_results(self) -> list:
        """Get only failed results"""
        return [r for r in self.results if r['status'] == 'failed']


# Convenience Function
# ============================================================================

def process_sequential(
    df: pd.DataFrame,
    predict_func: Callable,
    window_size: int = 50,
    step_size: int = 1,
    **predict_kwargs
) -> list:
    """
    Convenience function for sequential processing.
    
    Args:
        df: Input DataFrame
        predict_func: Prediction function
        window_size: Size of each window
        step_size: Step between windows
        **predict_kwargs: Arguments for prediction function
        
    Returns:
        List of all results
        
    Example:
        from src.data.data_loader import load_csv_data
        from src.app import predict_from_dataframe
        from src.sequential import process_sequential
        
        df = load_csv_data()
        results = process_sequential(
            df, 
            predict_from_dataframe,
            window_size=40,
            step_size=1,
            lookback=8,
            pred_len=2
        )
    """
    config = WindowConfig(window_size=window_size, step_size=step_size)
    processor = SequentialProcessor(config)
    
    # Process all batches
    for result in processor.process_batches(df, predict_func, **predict_kwargs):
        pass  # Results are stored internally
    
    return processor.get_all_results()

def process_sequential_batch(
    df: pd.DataFrame,
    predict_batch_func: Callable,
    batch_size: int = 32,
    window_size: int = 50,
    step_size: int = 1,
    **predict_kwargs
) -> list:
    """
    Convenience function for batched sequential processing.
    
    Args:
        df: Input DataFrame
        predict_batch_func: The 'predict_batch_windows' function
        batch_size: GPU Batch size (windows per inference)
        window_size: Size of sliding window
        step_size: Step between windows
        **predict_kwargs: Model args (lookback, pred_len, etc.)
    """
    config = WindowConfig(window_size=window_size, step_size=step_size)
    processor = SequentialProcessor(config)
    
    # Iterate to drive the generator
    for _ in processor.process_in_chunks(df, predict_batch_func, batch_size, **predict_kwargs):
        pass
        
    return processor.get_all_results()

if __name__ == "__main__":
    # Simple test
    print("Sequential Processor Test")
    print("="*70)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    sample_df = pd.DataFrame({
        'open': range(100),
        'high': range(100),
        'low': range(100),
        'close': range(100),
        'volume': range(100)
    }, index=dates)
    
    print(f"Sample data: {len(sample_df)} rows")
    
    # Mock prediction function
    def mock_predict(data, lookback=5, pred_len=2, **kwargs):
        print(f"    Processing {len(data)} rows...")
        return [], [], None
    
    # Process
    config = WindowConfig(window_size=10, step_size=2)
    processor = SequentialProcessor(config)
    
    batch_count = 0
    for result in processor.process_batches(sample_df, mock_predict, lookback=5, pred_len=2):
        batch_count += 1
        if batch_count >= 3:
            print("\n... (stopping after 3 batches for demo)")
            break
    
    print("\n" + "="*70)
    print("Test complete!")