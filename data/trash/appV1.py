#app.py

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
    print("Warning: Kronos model cannot be imported")

# Global variables
tokenizer: Optional[Any] = None
model: Optional[Any] = None
predictor: Optional[Any] = None

# Local model configurations (using downloaded models)
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

def load_data_file(file_path: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
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

def calculate_error_metrics(predictions: List[Dict[str, Any]], actuals: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Calculate error metrics (MAE, RMSE, MAPE) for each column"""
    if not predictions or not actuals or len(predictions) != len(actuals):
        return {}
    
    metrics = {}
    columns = ['open', 'high', 'low', 'close', 'volume']
    
    for col in columns:
        if col in predictions[0] and col in actuals[0]:
            pred_values = [float(p[col]) for p in predictions]
            actual_values = [float(a[col]) for a in actuals]
            
            # Calculate MAE (Mean Absolute Error)
            mae = np.mean(np.abs(np.array(pred_values) - np.array(actual_values)))
            
            # Calculate RMSE (Root Mean Squared Error)
            rmse = np.sqrt(np.mean((np.array(pred_values) - np.array(actual_values)) ** 2))
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            # Avoid division by zero
            actual_array = np.array(actual_values)
            mape = np.mean(np.abs((np.array(pred_values) - actual_array) / np.where(actual_array != 0, actual_array, 1))) * 100
            
            metrics[col] = {
                "MAE (Mean Absolute Error)": float(mae),
                "RMSE (Root Mean Squared Error)": float(rmse),
                "MAPE (Mean Absolute Percentage Error)": float(mape)
            }
            
            # Add MSE for volume (as shown in the example)
            if col == 'volume':
                mse = np.mean((np.array(pred_values) - np.array(actual_values)) ** 2)
                metrics[col]["MSE (Mean Squared Error)"] = float(mse)
    
    return metrics

def save_prediction_results(file_path: str, prediction_type: str, prediction_results: List[Dict[str, Any]], 
                           actual_data: List[Dict[str, Any]], input_data: pd.DataFrame, 
                           prediction_params: Dict[str, Any]) -> Optional[str]:
    """Save prediction results to JSON"""
    try:
        # Create results directory
        results_dir = os.path.join(PROJECT_ROOT, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'prediction_{timestamp}.json'
        filepath = os.path.join(results_dir, filename)
        
        # Prepare save data
        save_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'file_path': file_path,
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
                    'close': float(input_data['close'].iloc[-1])
                }
            },
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'analysis': {}
        }
        
        # Comparison analysis if actual data exists
        if actual_data and len(actual_data) > 0 and len(prediction_results) > 0:
            last_pred = prediction_results[0]
            first_actual = actual_data[0]
            
            save_data['analysis']['continuity'] = {
                'last_prediction': {
                    'open': last_pred['open'],
                    'high': last_pred['high'],
                    'low': last_pred['low'],
                    'close': last_pred['close']
                },
                'first_actual': {
                    'open': first_actual['open'],
                    'high': first_actual['high'],
                    'low': first_actual['low'],
                    'close': first_actual['close']
                },
                'gaps': {
                    'open_gap': abs(last_pred['open'] - first_actual['open']),
                    'high_gap': abs(last_pred['high'] - first_actual['high']),
                    'low_gap': abs(last_pred['low'] - first_actual['low']),
                    'close_gap': abs(last_pred['close'] - first_actual['close'])
                },
                'gap_percentages': {
                    'open_gap_pct': (abs(last_pred['open'] - first_actual['open']) / first_actual['open']) * 100,
                    'high_gap_pct': (abs(last_pred['high'] - first_actual['high']) / first_actual['high']) * 100,
                    'low_gap_pct': (abs(last_pred['low'] - first_actual['low']) / first_actual['low']) * 100,
                    'close_gap_pct': (abs(last_pred['close'] - first_actual['close']) / first_actual['close']) * 100
                }
            }
            
            # Calculate error metrics
            error_metrics = calculate_error_metrics(prediction_results, actual_data)
            if error_metrics:
                save_data['analysis']['error_metrics'] = error_metrics
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Results saved: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
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
    
    print(f"📦 Loading model from: {model_config['model_path']}")
    print(f"📦 Loading tokenizer from: {model_config['tokenizer_path']}")
    
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
    
    print(f"✅ Model loaded: {model_config['name']} ({model_config['params']}) on {device}")
    return predictor

def predict(file_path: str, lookback: int = 400, pred_len: int = 120, temperature: float = 1.0, 
            top_p: float = 0.9, sample_count: int = 1, start_date: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], pd.DataFrame]:
    """Perform prediction"""
    global predictor
    
    if predictor is None:
        raise Exception('Model not loaded. Call load_model() first')
    
    # Load data
    df, error = load_data_file(file_path)
    if error:
        raise Exception(error)
    
    # Type check to ensure df is not None
    if df is None:
        raise Exception('Failed to load data file')
    
    if len(df) < lookback:
        raise Exception(f'Insufficient data: need {lookback} rows, got {len(df)}')
    
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
        
        # Ensure we're working with pandas Series
        time_range_timestamps: pd.Series = pd.Series(time_range_df['timestamps'])
        time_span = time_range_timestamps.iloc[lookback+pred_len-1] - time_range_timestamps.iloc[0]
        prediction_type = f"Window prediction (lookback={lookback}, pred={pred_len}, span={time_span})"
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
    
    # Ensure x_timestamp and y_timestamp are Series
    if not isinstance(x_timestamp, pd.Series):
        x_timestamp = pd.Series(x_timestamp, name='timestamps')
    if y_timestamp is not None and not isinstance(y_timestamp, pd.Series):
        y_timestamp = pd.Series(y_timestamp, name='timestamps')
    
    # Normalize start_date to the actual first timestamp of the lookback window
    # This records the true beginning of the context used for prediction
    derived_start_date = (
        x_timestamp.iloc[0].isoformat() if len(x_timestamp) > 0 else (start_date if start_date else None)
    )
    # Also capture the end_date as the last timestamp of the lookback window
    derived_end_date = (
        x_timestamp.iloc[-1].isoformat() if len(x_timestamp) > 0 else None
    )
    
    print(f"🔮 Predicting with {model.__class__.__name__}...")
    
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
            # Ensure we're working with pandas Series
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
        # Handle timestamp formatting
        if i < len(future_timestamps):
            timestamp_item = future_timestamps[i]
            # Handle different timestamp types
            try:
                if isinstance(timestamp_item, pd.Timestamp):
                    timestamp_str = timestamp_item.isoformat()
                elif hasattr(timestamp_item, 'isoformat') and not isinstance(timestamp_item, (pd.DatetimeIndex, int)):
                    timestamp_str = timestamp_item.isoformat()
                else:
                    # Convert to string as fallback
                    timestamp_str = str(timestamp_item)
            except (AttributeError, TypeError):
                # Final fallback
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
        file_path=file_path,
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
    
    print(f"✅ Prediction complete: {pred_len} points generated")
    if actual_data:
        print(f"📊 Comparison: {len(actual_data)} actual points available")
    
    return prediction_results, actual_data, pred_df


if __name__ == '__main__':
    print("=" * 60)
    print("KRONOS PIPELINE - Standalone Prediction")
    print("=" * 60)
    print(f"Model availability: {MODEL_AVAILABLE}")
    print(f"Project root: {PROJECT_ROOT}")
    print()
    
    # Example usage
    try:
        # 1. Load model
        print("Step 1: Loading model...")
        load_model(model_key='kronos-small', device='cpu')
        print()
        
        # 2. Run prediction
        print("Step 2: Running prediction...")
        data_path = os.path.join(PROJECT_ROOT, 'data', 'test.csv')
        
        predictions, actuals, pred_df = predict(
            file_path=data_path,
            lookback=400,
            pred_len=120,
            temperature=1.0,
            top_p=0.9
        )
        
        print()
        print(f"📈 Predictions: {len(predictions)} points")
        print(f"📊 Actuals: {len(actuals)} points")
        print(f"✅ Pipeline complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")