# app.py

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
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

try:
    from src.model import Kronos, KronosTokenizer, KronosPredictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

# Global variables
tokenizer: Optional[Any] = None
model: Optional[Any] = None
predictor: Optional[Any] = None

# Local model configurations
AVAILABLE_MODELS = {
    'kronos-mini': {
        'name': 'Kronos-mini',
        'model_path': os.path.join(PROJECT_ROOT, 'models', 'kronos_mini'),
        'tokenizer_path': os.path.join(PROJECT_ROOT, 'models', 'kronos_tokenizer_2k'),
        'context_length': 2048,
        'params': '4.1M',
        'description': 'Lightweight model, suitable for fast prediction'
    },
    'kronos-small': {
        'name': 'Kronos-small',
        'model_path': os.path.join(PROJECT_ROOT, 'models', 'kronos_small'),
        'tokenizer_path': os.path.join(PROJECT_ROOT, 'models', 'kronos_tokenizer_base'),
        'context_length': 512,
        'params': '24.7M',
        'description': 'Small model, balanced performance and speed'
    },
    'kronos-base': {
        'name': 'Kronos-base',
        'model_path': os.path.join(PROJECT_ROOT, 'models', 'kronos_base'),
        'tokenizer_path': os.path.join(PROJECT_ROOT, 'models', 'kronos_tokenizer_base'),
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
# Core Functions
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
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return None, f"Missing required columns: {required_cols}"
        
        # Process timestamp column
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'])
        elif 'timestamp' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            df['timestamps'] = pd.to_datetime(df['date'])
        else:
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
        
        # Remove NaN rows
        df = df.dropna()
        
        return df, None
        
    except Exception as e:
        return None, f"Failed to load file: {str(e)}"

def load_data_from_dataframe(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load and process data from an existing DataFrame"""
    try:
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return None, f"Missing required columns: {required_cols}"
        
        # Process timestamp column
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'])
        elif 'timestamp' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            df['timestamps'] = pd.to_datetime(df['date'])
        else:
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
        
        # Remove NaN rows
        df = df.dropna()
        
        return df, None
        
    except Exception as e:
        return None, f"Failed to process DataFrame: {str(e)}"

def save_prediction_results(group: str, prediction_type: str, prediction_results: List[Dict[str, Any]], 
                           actual_data: List[Dict[str, Any]], input_data: pd.DataFrame, 
                           prediction_params: Dict[str, Any]) -> Optional[str]:
    """Save prediction results to JSON"""
    try:
        results_dir = os.path.join(PROJECT_ROOT, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'prediction_{timestamp}.json'
        filepath = os.path.join(results_dir, filename)
        
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

def load_model(model_key: str = 'kronos-small', device: str = 'cpu') -> Any:
    """Load Kronos model from local directory"""
    global tokenizer, model, predictor
    
    if not MODEL_AVAILABLE:
        raise Exception('Kronos model library not available')
    
    if model_key not in AVAILABLE_MODELS:
        raise Exception(f'Unsupported model: {model_key}')
    
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

def predict(file_path: str, lookback: int = 400, pred_len: int = 120, temperature: float = 1.0, 
            top_p: float = 0.9, sample_count: int = 1, start_date: Optional[str] = None, 
            group: str = "default") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], pd.DataFrame]:
    """Perform prediction"""
    global predictor
    
    if predictor is None:
        raise Exception('Model not loaded. Call load_model() first')
    
    # Load data
    log_info(f"Loading data from {os.path.basename(file_path)}")
    df, error = load_data_from_file(file_path)
    if error:
        raise Exception(error)
    
    if df is None:
        raise Exception('Failed to load data file')
    
    if len(df) < lookback:
        raise Exception(f'Insufficient data: need {lookback} rows, got {len(df)}')
    
    log_success(f"Loaded {len(df)} data points")
    
    # Required columns
    required_cols = ['open', 'high', 'low', 'close']
    if 'volume' in df.columns:
        required_cols.append('volume')
    
    # Time period selection
    if start_date:
        start_dt = pd.to_datetime(start_date)
        mask = df['timestamps'] >= start_dt
        time_range_df = df[mask]
        
        if len(time_range_df) < lookback + pred_len:
            raise Exception(
                f'Insufficient data from {start_dt.strftime("%Y-%m-%d %H:%M")}: '
                f'need {lookback + pred_len}, got {len(time_range_df)}'
            )
        
        x_df = time_range_df.iloc[:lookback][required_cols]
        x_timestamp = time_range_df.iloc[:lookback]['timestamps']
        y_timestamp = time_range_df.iloc[lookback:lookback+pred_len]['timestamps']
        
        time_range_timestamps: pd.Series = pd.Series(time_range_df['timestamps'])
        time_span = time_range_timestamps.iloc[lookback+pred_len-1] - time_range_timestamps.iloc[0]
        prediction_type = f"Window prediction"
    else:
        x_df = df.iloc[:lookback][required_cols]
        x_timestamp = df.iloc[:lookback]['timestamps']
        y_timestamp = df.iloc[lookback:lookback+pred_len]['timestamps'] if len(df) >= lookback + pred_len else None
        prediction_type = "Latest data prediction"
    
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
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=temperature,
        top_p=top_p,
        sample_count=sample_count
    )
    
    # Prepare actual data (if available)
    actual_data = []
    if start_date:
        start_dt = pd.to_datetime(start_date)
        mask = df['timestamps'] >= start_dt
        time_range_df = df[mask]
        
        if len(time_range_df) >= lookback + pred_len:
            actual_df = time_range_df.iloc[lookback:lookback+pred_len]
            for _, row in actual_df.iterrows():
                actual_data.append({
                    'timestamp': row['timestamps'].isoformat(),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']) if 'volume' in row else 0,
                })
    else:
        if len(df) >= lookback + pred_len:
            actual_df = df.iloc[lookback:lookback+pred_len]
            for _, row in actual_df.iterrows():
                actual_data.append({
                    'timestamp': row['timestamps'].isoformat(),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']) if 'volume' in row else 0,
                })
    
    # Calculate future timestamps
    if start_date:
        start_dt = pd.to_datetime(start_date)
        mask = df['timestamps'] >= start_dt
        time_range_df = df[mask]
        
        if len(time_range_df) >= lookback:
            timestamps_series: pd.Series = pd.Series(time_range_df['timestamps'])
            df_timestamps_series: pd.Series = pd.Series(df['timestamps'])
            last_timestamp = timestamps_series.iloc[lookback-1]
            time_diff = df_timestamps_series.iloc[1] - df_timestamps_series.iloc[0]
            future_timestamps = pd.date_range(
                start=last_timestamp + time_diff,
                periods=pred_len,
                freq=time_diff
            )
        else:
            future_timestamps = range(pred_len)
    else:
        last_timestamp = df['timestamps'].iloc[lookback-1] if len(df) > lookback else df['timestamps'].iloc[-1]
        time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]
        future_timestamps = pd.date_range(
            start=last_timestamp + time_diff,
            periods=pred_len,
            freq=time_diff
        )
    
    # Prepare prediction results
    prediction_results = []
    for i, (_, row) in enumerate(pred_df.iterrows()):
        if i < len(future_timestamps):
            timestamp_item = future_timestamps[i]
            try:
                if isinstance(timestamp_item, pd.Timestamp):
                    timestamp_str = timestamp_item.isoformat()
                elif hasattr(timestamp_item, 'isoformat') and not isinstance(timestamp_item, (pd.DatetimeIndex, int)):
                    timestamp_str = timestamp_item.isoformat()
                else:
                    timestamp_str = str(timestamp_item)
            except (AttributeError, TypeError):
                timestamp_str = str(timestamp_item)
        else:
            timestamp_str = f"T{i}"
            
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
            'start_date': derived_start_date,
            'end_date': derived_end_date
        }
    )
    
    log_success(f"Generated {pred_len} prediction points")
    if actual_data:
        log_info(f"Comparison data: {len(actual_data)} actual points available")
    
    return prediction_results, actual_data, pred_df


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    log_section("KRONOS TIME SERIES PREDICTION PIPELINE")
    
    print(f"\n  Project Root: {PROJECT_ROOT}")
    print(f"  Model Status: {'Available' if MODEL_AVAILABLE else 'Not Available'}")
    
    try:
        # Step 1: Load model
        log_section("STEP 1: MODEL INITIALIZATION")
        load_model(model_key='kronos-mini', device='cuda')
        
        # Step 2: Run prediction
        log_section("STEP 2: PREDICTION")
        data_path = os.path.join(PROJECT_ROOT, 'data', 'test.csv')
        
        predictions, actuals, pred_df = predict(
            file_path=data_path,
            lookback=5,
            pred_len=2,
            temperature=1.0,
            top_p=0.9,
            group="BTCUSDT_1H"
        )
        
        # Summary
        log_section("EXECUTION SUMMARY")
        print(f"\n  Predictions Generated: {len(predictions)}")
        print(f"  Actual Data Points: {len(actuals)}")
        print(f"  Status: Complete\n")
        
        print(f"{'─' * 70}\n")
        
    except Exception as e:
        log_error(f"{str(e)}")
        sys.exit(1)