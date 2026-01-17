import os
import pandas as pd
import numpy as np
import json
import datetime
import warnings
import sys
from typing import Optional, Tuple, List, Dict, Any

warnings.filterwarnings('ignore')

# Add project root to path
# Assuming app.py is inside a 'scripts' or 'src' folder
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Resolve models directory: prefer `src/models`, fallback to project-root `models`
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
if not os.path.isdir(MODELS_DIR):
    MODELS_DIR = os.path.normpath(os.path.join(PROJECT_ROOT, '..', 'models'))
# If app.py is in the root, use this instead:
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

try:
    # Assuming model code is in 'src'
    from src.model import Kronos, KronosTokenizer, KronosPredictor
    MODEL_AVAILABLE = True
except ImportError:
    try:
        # Fallback if app.py is in 'src'
        from model import Kronos, KronosTokenizer, KronosPredictor
        MODEL_AVAILABLE = True
    except ImportError:
        MODEL_AVAILABLE = False
        print("[WARN] Kronos model libraries not found. Model-related functions will fail.")


# Global variables
tokenizer: Optional[Any] = None
model: Optional[Any] = None
predictor: Optional[Any] = None

# Local model configurations
AVAILABLE_MODELS = {
    'kronos-mini': {
        'name': 'Kronos-mini',
        'model_path': os.path.join(MODELS_DIR, 'kronos_mini'),
        'tokenizer_path': os.path.join(MODELS_DIR, 'kronos_tokenizer_2k'),
        'context_length': 2048,
        'params': '4.1M',
        'description': 'Lightweight model, suitable for fast prediction'
    },
    'kronos-small': {
        'name': 'Kronos-small',
        'model_path': os.path.join(MODELS_DIR, 'kronos_small'),
        'tokenizer_path': os.path.join(MODELS_DIR, 'kronos_tokenizer_base'),
        'context_length': 512,
        'params': '24.7M',
        'description': 'Small model, balanced performance and speed'
    },
    'kronos-base': {
        'name': 'Kronos-base',
        'model_path': os.path.join(MODELS_DIR, 'kronos_base'),
        'tokenizer_path': os.path.join(MODELS_DIR, 'kronos_tokenizer_base'),
        'context_length': 512,
        'params': '102.3M',
        'description': 'Base model, provides better prediction quality'
    }
}

# ============================================================================
# Utility Functions for Clean Output
# ============================================================================

def log_info(message: str):
    """Print info message"""
    print(f"[INFO] {message}")

def log_success(message: str):
    """Print success message"""
    print(f"[ OK ] {message}")

def log_error(message: str):
    """Print error message"""
    print(f"[FAIL] {message}")

def log_section(title: str):
    """Print section header"""
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")

# ============================================================================
# Data Loading and Saving Functions
# ============================================================================

def load_data_from_file(file_path: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load data file (CSV or Feather)"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.feather'):
            df = pd.read_feather(file_path)
        else:
            return None, "Unsupported file format"
        
        # Process and validate the DataFrame
        df, error = load_data_from_dataframe(df)
        if error:
            return None, error
            
        return df, None
        
    except Exception as e:
        return None, f"Failed to load file: {str(e)}"

def load_data_from_dataframe(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load and process data from an existing DataFrame"""
    try:
        # Make a copy to avoid modifying the original DataFrame
        df = df.copy()

        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return None, f"Missing required columns: {required_cols}"
        
        # Handle timestamp column - check for ambiguity first
        timestamp_source = None
        
        # Check if 'timestamps' exists as both column and index level (ambiguous case)
        if 'timestamps' in df.columns and hasattr(df.index, 'names') and 'timestamps' in df.index.names:
            # Use the column version and drop the index level to avoid ambiguity
            df = df.reset_index(level='timestamps', drop=True)
            timestamp_source = 'column'
        elif 'timestamps' in df.columns:
            timestamp_source = 'column'
        elif 'timestamp' in df.columns:
            timestamp_source = 'timestamp'
        elif 'date' in df.columns:
            timestamp_source = 'date'
        elif pd.api.types.is_datetime64_any_dtype(df.index):
            timestamp_source = 'index'
        
        # Process timestamp based on source
        if timestamp_source == 'column':
            df['timestamps'] = pd.to_datetime(df['timestamps'])
        elif timestamp_source == 'timestamp':
            df['timestamps'] = pd.to_datetime(df['timestamp'])
        elif timestamp_source == 'date':
            df['timestamps'] = pd.to_datetime(df['date'])
        elif timestamp_source == 'index':
            df['timestamps'] = df.index
            # Reset index name to avoid ambiguity when timestamps becomes a column
            df.index.name = None
        else:
            # Fallback: create a dummy index
            log_info("No timestamp column found, creating dummy time index.")
            df['timestamps'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1H')
        
        # Ensure numeric columns
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Process volume (optional)
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Process amount (optional)
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Remove NaN rows that might have been introduced
        df = df.dropna(subset=required_cols + ['timestamps'])
        
        # Sort by timestamp to ensure correct order
        df = df.sort_values(by='timestamps').reset_index(drop=True)

        return df, None
        
    except Exception as e:
        return None, f"Failed to process DataFrame: {str(e)}"

def save_prediction_results(group: str, prediction_type: str, prediction_results: List[Dict[str, Any]], 
                            actual_data: List[Dict[str, Any]], input_data: pd.DataFrame, 
                            prediction_params: Dict[str, Any], output_config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Save prediction results to JSON"""
    try:
        # Extract config or use defaults
        if output_config:
            custom_path = output_config.get('custom_output_path')
            base_dir = output_config.get('prediction_files_dir', 'results')
            filename_template = output_config.get('prediction_filename_template', 'prediction_{group}_{timestamp}.json')
            create_subdirs = output_config.get('create_timestamped_subdirs', False)
        else:
            custom_path = None
            base_dir = 'results'
            filename_template = 'prediction_{group}_{timestamp}.json'
            create_subdirs = False

        # Determine base directory
        if custom_path:
            # Custom path overrides everything
            results_dir = custom_path
        else:
            # Use configured base directory
            if not os.path.isabs(base_dir):
                main_project_root = os.path.dirname(PROJECT_ROOT)
                base_dir = os.path.join(main_project_root, base_dir)
            results_dir = base_dir
            
            # Add timestamped subdirectory if configured
            if create_subdirs:
                subdir = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
                results_dir = os.path.join(results_dir, subdir)

        os.makedirs(results_dir, exist_ok=True)

        # Generate filename from template
        base_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = filename_template.format(group=group, timestamp=base_timestamp)
        filepath = os.path.join(results_dir, filename)

        # Fallback: if a file with the same name somehow exists, add an incrementing suffix
        if os.path.exists(filepath):
            counter = 1
            while True:
                alt_filename = filename_template.format(group=group, timestamp=f"{base_timestamp}_{counter}")
                alt_path = os.path.join(results_dir, alt_filename)
                if not os.path.exists(alt_path):
                    filename = alt_filename
                    filepath = alt_path
                    break
                counter += 1
        
        save_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'group': group,
            'prediction_type': prediction_type,
            'prediction_params': prediction_params,
            'input_data_summary': {
                'rows': len(input_data),
                'columns': list(input_data.columns),
                'price_range': {
                    'open': {'min': float(input_data['open'].min()), 'max': float(input_data['open'].max())},
                    'high': {'min': float(input_data['high'].min()), 'max': float(input_data['high'].max())},
                    'low': {'min': float(input_data['low'].min()), 'max': float(input_data['low'].max())},
                    'close': {'min': float(input_data['close'].min()), 'max': float(input_data['close'].max())}
                },
                'last_values': {
                    'open': float(input_data['open'].iloc[-1]),
                    'high': float(input_data['high'].iloc[-1]),
                    'low': float(input_data['low'].iloc[-1]),
                    'close': float(input_data['close'].iloc[-1]),
                    'volume': float(input_data['volume'].iloc[-1]) if 'volume' in input_data.columns else 0
                }
            },
            'prediction_results': prediction_results,
            'actual_data': actual_data
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        log_success(f"Results saved to {os.path.basename(filepath)}")
        return filepath
        
    except Exception as e:
        log_error(f"Failed to save results: {e}")
        return None

# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_key: str = 'kronos-small', device: str = 'cpu') -> Any:
    """Load Kronos model from local directory"""
    global tokenizer, model, predictor
    
    if not MODEL_AVAILABLE:
        raise Exception('Kronos model library not available. Please check installation.')
    
    if model_key not in AVAILABLE_MODELS:
        raise Exception(f'Unsupported model: {model_key}. Available: {list(AVAILABLE_MODELS.keys())}')
    
    model_config = AVAILABLE_MODELS[model_key]
    
    # Verify local paths exist
    if not os.path.exists(model_config['model_path']):
        raise Exception(f"Model path not found: {model_config['model_path']}")
    if not os.path.exists(model_config['tokenizer_path']):
        raise Exception(f"Tokenizer path not found: {model_config['tokenizer_path']}")
    
    log_info(f"Loading model: {model_config['name']} ({model_config['params']})")
    
    # Load from local paths
    tokenizer = KronosTokenizer.from_pretrained(model_config['tokenizer_path'])
    model = Kronos.from_pretrained(model_config['model_path'])
    
    # Create predictor
    predictor = KronosPredictor(
        model, 
        tokenizer, 
        device=device, 
        max_context=model_config['context_length']
    )
    
    log_success(f"Model ready on {device}")
    return predictor

# ============================================================================
# Core Prediction Logic (Internal)
# ============================================================================

def _predict_core(df: pd.DataFrame, lookback: int = 400, pred_len: int = 120, 
                  temperature: float = 1.0, top_p: float = 0.9, 
                  sample_count: int = 1, start_date: Optional[str] = None, 
                  group: str = "default", output_config: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], pd.DataFrame]:
    """
    Internal core prediction logic. 
    Handles both DataFrames with 'timestamps' column and with timestamps as index.
    """
    global predictor
    
    if predictor is None:
        raise Exception('Model not loaded. Call load_model() first')
        
    if len(df) < lookback:
        raise Exception(f'Insufficient data: need {lookback} rows, got {len(df)}')
    
    # Required columns
    required_cols = ['open', 'high', 'low', 'close']
    if 'volume' in df.columns:
        required_cols.append('volume')
    
    # Handle timestamp column/index - ensure we have a 'timestamps' column
    if 'timestamps' not in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df.index):
            # Timestamps are in the index, create a column
            df = df.copy()
            df['timestamps'] = df.index
        else:
            raise Exception("No timestamp column or datetime index found")
    
    # Time period selection
    if start_date:
        start_dt = pd.to_datetime(start_date)
        mask = df['timestamps'] >= start_dt
        time_range_df = df[mask].reset_index(drop=True) # Reset index for safe iloc
        
        # We need at least lookback data points *from the start_date*
        if len(time_range_df) < lookback:
            raise Exception(
                f'Insufficient data from {start_dt.strftime("%Y-%m-%d %H:%M")}: '
                f'need {lookback} context rows, got {len(time_range_df)}'
            )
        
        x_df = time_range_df.iloc[:lookback][required_cols]
        x_timestamp = time_range_df.iloc[:lookback]['timestamps']
        
        # Check if we have *actual* data for comparison
        if len(time_range_df) >= lookback + pred_len:
            y_timestamp = time_range_df.iloc[lookback:lookback+pred_len]['timestamps']
        else:
            y_timestamp = None # We are predicting into the unknown future

        prediction_type = f"Window prediction"
    else:
        # Default behavior: use the first 'lookback' rows
        x_df = df.iloc[:lookback][required_cols]
        x_timestamp = df.iloc[:lookback]['timestamps']
        
        # Check if we have *actual* data for comparison
        if len(df) >= lookback + pred_len:
            y_timestamp = df.iloc[lookback:lookback+pred_len]['timestamps']
        else:
            y_timestamp = None

        prediction_type = "Sequential data prediction"
    
    # --- Start of prediction logic ---
    
    # Ensure Series format
    if isinstance(x_timestamp, pd.DatetimeIndex):
        x_timestamp = pd.Series(x_timestamp, name='timestamps')
    if y_timestamp is not None and isinstance(y_timestamp, pd.DatetimeIndex):
        y_timestamp = pd.Series(y_timestamp, name='timestamps')
    
    if not isinstance(x_timestamp, pd.Series):
        x_timestamp = pd.Series(x_timestamp, name='timestamps')
    if y_timestamp is not None and not isinstance(y_timestamp, pd.Series):
        y_timestamp = pd.Series(y_timestamp, name='timestamps')
    
    # Normalize start_date
    derived_start_date = (
        x_timestamp.iloc[0].isoformat() if len(x_timestamp) > 0 else (start_date if start_date else None)
    )
    derived_end_date = (
        x_timestamp.iloc[-1].isoformat() if len(x_timestamp) > 0 else None
    )
    
    # Display prediction parameters
    print(f"\n  Context window: {lookback} periods")
    print(f"  Prediction length: {pred_len} periods")
    print(f"  Temperature: {temperature}")
    print(f"  Top-p: {top_p}")
    
    log_info("Running prediction...")
    
    # Perform prediction
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp, # Pass y_timestamp if available for teacher-forcing (if model supports it)
        pred_len=pred_len,
        T=temperature,
        top_p=top_p,
        sample_count=sample_count
    )
    
    # Prepare actual data (if available)
    actual_data = []
    comparison_df = None
    if start_date:
        start_dt = pd.to_datetime(start_date)
        mask = df['timestamps'] >= start_dt
        time_range_df = df[mask].reset_index(drop=True)
        
        if len(time_range_df) >= lookback + pred_len:
            comparison_df = time_range_df.iloc[lookback:lookback+pred_len]
    else:
        if len(df) >= lookback + pred_len:
            comparison_df = df.iloc[lookback:lookback+pred_len]

    if comparison_df is not None:
        for _, row in comparison_df.iterrows():
            actual_data.append({
                'timestamp': row['timestamps'].isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0,
            })
    
    # Calculate future timestamps
    last_timestamp = x_timestamp.iloc[-1]
    
    # Try to infer frequency
    if len(x_timestamp) > 1:
        time_diff = x_timestamp.iloc[1] - x_timestamp.iloc[0]
        freq = pd.infer_freq(x_timestamp)
        if freq is None:
             log_info(f"Could not infer frequency. Using diff: {time_diff}")
             freq = time_diff
    else:
        log_info("Only 1 context point, cannot infer frequency. Assuming 1 Hour.")
        freq = '1H' # Default fallback
        
    future_timestamps = pd.date_range(
        start=last_timestamp,
        periods=pred_len + 1, # +1 to exclude the start date itself
        freq=freq
    )[1:] # Exclude the first one, which is last_timestamp

    
    # Prepare prediction results
    prediction_results = []
    for i, (_, row) in enumerate(pred_df.iterrows()):
        if i < len(future_timestamps):
            ts_val = future_timestamps[i]
            timestamp_str = getattr(ts_val, 'isoformat', lambda: str(ts_val))()
        else:
            timestamp_str = f"T+{i+1}" # Fallback
            
        prediction_results.append({
            'timestamp': timestamp_str,
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume']) if 'volume' in row else 0,
        })
    
    # Save results
    save_prediction_results(
        group=group,
        prediction_type=prediction_type,
        prediction_results=prediction_results,
        actual_data=actual_data,
        input_data=x_df,
        prediction_params={
            'lookback': lookback,
            'pred_len': pred_len,
            'temperature': temperature,
            'top_p': top_p,
            'sample_count': sample_count,
            'context_start_date': derived_start_date,
            'context_end_date': derived_end_date
        },
        output_config=output_config
    )
    
    log_success(f"Generated {pred_len} prediction points")
    if actual_data:
        log_info(f"Comparison data: {len(actual_data)} actual points available")
    
    return prediction_results, actual_data, pred_df

# ============================================================================
# Public Prediction Functions
# ============================================================================

def predict_from_file(file_path: str, lookback: int = 400, pred_len: int = 120, 
                      temperature: float = 1.0, top_p: float = 0.9, 
                      sample_count: int = 1, start_date: Optional[str] = None, 
                      group: str = "default", output_config: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], pd.DataFrame]:
    """
    Load data from a file and perform prediction.
    
    Args:
        file_path (str): Path to the .csv or .feather file.
        lookback (int): Number of historical data points to use as context.
        pred_len (int): Number of future data points to predict.
        temperature (float): Sampling temperature.
        top_p (float): Top-p sampling.
        sample_count (int): Number of samples to generate.
        start_date (Optional[str]): ISO format string (e.g., "2024-01-31T10:00:00").
                                    If provided, prediction context starts from this date.
                                    If None, context starts from the beginning of the file.
        group (str): A name for this prediction group (used for saving results).
        output_config (Optional[Dict[str, Any]]): Output configuration dict with keys:
            - prediction_files_dir: Base directory for output files
            - prediction_filename_template: Template for filenames (use {group} and {timestamp})
            - create_timestamped_subdirs: Whether to create timestamped subdirectories
            - custom_output_path: Override path that ignores all other settings

    Returns:
        Tuple of (predictions, actuals, pred_df)
    """
    
    # Step 1: Load data
    log_info(f"Loading data from {os.path.basename(file_path)}")
    df, error = load_data_from_file(file_path)
    if error:
        raise Exception(error)
    
    if df is None:
        raise Exception('Failed to load data file')
    
    log_success(f"Loaded {len(df)} data points")

    # Step 2: Call the core function
    return _predict_core(
        df=df,
        lookback=lookback,
        pred_len=pred_len,
        temperature=temperature,
        top_p=top_p,
        sample_count=sample_count,
        start_date=start_date,
        group=group,
        output_config=output_config
    )

def predict_from_dataframe(data: pd.DataFrame, lookback: int = 400, pred_len: int = 120, 
                           temperature: float = 1.0, top_p: float = 0.9, 
                           sample_count: int = 1, start_date: Optional[str] = None, 
                           group: str = "default", output_config: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], pd.DataFrame]:
    """
    Perform prediction from an in-memory DataFrame.
    
    Args:
        data (pd.DataFrame): DataFrame with OHLCV data. Must have 'open', 'high',
                             'low', 'close', and a timestamp column ('timestamps',
                             'timestamp', 'date', or be in the index).
        lookback (int): Number of historical data points to use as context.
        pred_len (int): Number of future data points to predict.
        temperature (float): Sampling temperature.
        top_p (float): Top-p sampling.
        sample_count (int): Number of samples to generate.
        start_date (Optional[str]): ISO format string (e.g., "2024-01-31T10:00:00").
                                    If provided, prediction context starts from this date.
                                    If None, context starts from the beginning of the DataFrame.
        group (str): A name for this prediction group (used for saving results).
        output_config (Optional[Dict[str, Any]]): Output configuration dict with keys:
            - prediction_files_dir: Base directory for output files
            - prediction_filename_template: Template for filenames (use {group} and {timestamp})
            - create_timestamped_subdirs: Whether to create timestamped subdirectories
            - custom_output_path: Override path that ignores all other settings

    Returns:
        Tuple of (predictions, actuals, pred_df)
    """
    
    # Step 1: Process data
    log_info(f"Processing data from DataFrame")
    df, error = load_data_from_dataframe(data) # This function ensures 'timestamps' col exists
    if error:
        raise Exception(error)
    
    if df is None:
        raise Exception('Failed to process DataFrame')
    
    log_success(f"Processed {len(df)} data points")
    
    # Step 2: Call the core function
    return _predict_core(
        df=df,
        lookback=lookback,
        pred_len=pred_len,
        temperature=temperature,
        top_p=top_p,
        sample_count=sample_count,
        start_date=start_date,
        group=group,
        output_config=output_config
    )


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    log_section("KRONOS TIME SERIES PREDICTION PIPELINE")
    
    print(f"\n  Project Root: {PROJECT_ROOT}")
    print(f"  Model Status: {'Available' if MODEL_AVAILABLE else 'Not Available'}")
    
    if not MODEL_AVAILABLE:
        log_error("Model libraries not found. Exiting.")
        sys.exit(1)
        
    try:
        # Step 1: Load model
        log_section("STEP 1: MODEL INITIALIZATION")
        # Use 'cuda' if available, otherwise 'cpu'
        device = 'cuda' if 'torch' in sys.modules and sys.modules['torch'].cuda.is_available() else 'cpu'
        load_model(model_key='kronos-mini', device=device)
        
        # Step 2: Run prediction from file
        log_section("STEP 2: PREDICTION FROM FILE")
        data_path = os.path.join(PROJECT_ROOT, 'data', 'test.csv')
        
        if not os.path.exists(data_path):
            log_error(f"Test data file not found: {data_path}")
            # Create a dummy test.csv for demonstration
            log_info("Creating dummy 'data/test.csv' for demonstration...")
            os.makedirs(os.path.join(PROJECT_ROOT, 'data'), exist_ok=True)
            sample_data_df = pd.DataFrame({
                'timestamp': pd.date_range(start='2025-01-01', periods=20, freq='1H'),
                'open': np.random.uniform(100, 102, 20).round(2),
                'high': np.random.uniform(102, 104, 20).round(2),
                'low': np.random.uniform(98, 100, 20).round(2),
                'close': np.random.uniform(100, 102, 20).round(2),
                'volume': np.random.randint(1000, 5000, 20)
            })
            sample_data_df.to_csv(data_path, index=False)
            log_success(f"Dummy data saved to {data_path}")


        predictions, actuals, pred_df = predict_from_file(
            file_path=data_path,
            lookback=10,
            pred_len=5,
            temperature=1.0,
            top_p=0.9,
            group="BTCUSDT_1H_FILE"
        )
        
        # --- Example of using the predict_from_dataframe function ---
        log_section("STEP 3: PREDICTION FROM DATAFRAME (EXAMPLE)")
        
        # Create a sample dataframe in memory
        sample_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2025-01-01', periods=20, freq='1H'),
            'open': np.random.uniform(100, 102, 20).round(2),
            'high': np.random.uniform(102, 104, 20).round(2),
            'low': np.random.uniform(98, 100, 20).round(2),
            'close': np.random.uniform(100, 102, 20).round(2),
            'volume': np.random.randint(1000, 5000, 20)
        })

        log_info(f"Created sample DataFrame with {len(sample_df)} rows")

        df_predictions, df_actuals, df_pred_df = predict_from_dataframe(
            data=sample_df,
            lookback=10,
            pred_len=5,
            group="BTCUSDT_1H_DF",
            start_date="2025-01-01T05:00:00" # Example of starting from a specific date
        )
        # --- End of example ---

        # Summary
        log_section("EXECUTION SUMMARY")
        print(f"\n  File Predictions Generated: {len(predictions)}")
        print(f"  File Actual Data Points: {len(actuals)}")
        print(f"\n  DF Predictions Generated: {len(df_predictions)}")
        print(f"  DF Actual Data Points: {len(df_actuals)}")
        print(f"  Status: Complete\n")
        
        print(f"{'─' * 70}\n")
        
    except Exception as e:
        log_error(f"{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)