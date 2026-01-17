# app.py - PERFECTED PRODUCTION VERSION

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
    """
    Load data file (CSV or Feather) with comprehensive validation.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        Tuple of (DataFrame, error_message)
    """
    try:
        # Validate file existence
        if not os.path.exists(file_path):
            return None, f"File not found: {file_path}"
        
        # Load based on file extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.feather'):
            df = pd.read_feather(file_path)
        else:
            return None, "Unsupported file format. Use CSV or Feather."
        
        # Validate minimum data
        if len(df) == 0:
            return None, "File is empty"
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return None, f"Missing required columns: {missing_cols}"
        
        # Process timestamp column (try multiple common names)
        timestamp_col = None
        for col_name in ['timestamps', 'timestamp', 'date', 'datetime', 'time']:
            if col_name in df.columns:
                timestamp_col = col_name
                break
        
        if timestamp_col:
            try:
                df['timestamps'] = pd.to_datetime(df[timestamp_col])
            except Exception as e:
                return None, f"Failed to parse timestamps: {str(e)}"
        else:
            # Generate synthetic timestamps
            print("⚠️  No timestamp column found, generating synthetic timestamps")
            df['timestamps'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1H')
        
        # Ensure numeric columns with validation
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().sum() == len(df):
                return None, f"Column '{col}' contains no valid numeric data"
        
        # Process volume (optional)
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            # Replace NaN volumes with 0
            df['volume'] = df['volume'].fillna(0)
        else:
            # Add volume column with zeros if not present
            df['volume'] = 0.0
        
        # Process amount (optional)
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            df['amount'] = df['amount'].fillna(0)
        
        # Remove rows with NaN in required columns
        initial_len = len(df)
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'timestamps'])
        
        if len(df) == 0:
            return None, "No valid rows after removing NaN values"
        
        if len(df) < initial_len:
            print(f"⚠️  Removed {initial_len - len(df)} rows with missing values")
        
        # Validate price data integrity
        invalid_rows = (df['high'] < df['low']) | (df['high'] < df['open']) | \
                      (df['high'] < df['close']) | (df['low'] > df['open']) | \
                      (df['low'] > df['close'])
        
        if invalid_rows.sum() > 0:
            print(f"⚠️  Found {invalid_rows.sum()} rows with invalid OHLC relationships")
            # Optionally fix or remove invalid rows
            df = df[~invalid_rows]
        
        # Sort by timestamp
        df = df.sort_values(by=['timestamps']).reset_index(drop=True)  # type: ignore
        
        # Detect time frequency
        if len(df) > 1:
            time_diffs = df['timestamps'].diff().dropna()
            most_common_freq = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
            print(f"📊 Detected time frequency: {most_common_freq}")
        
        return df, None
        
    except Exception as e:
        return None, f"Failed to load file: {str(e)}"


def calculate_error_metrics(
    predictions_df: pd.DataFrame, 
    actuals_df: pd.DataFrame,
    align_on: str = 'timestamp'
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate comprehensive error metrics (MAE, RMSE, MAPE, sMAPE, MSE, R², Directional Accuracy).
    
    PERFECTED VERSION with fixes:
    ✅ Uses pandas DataFrames for efficiency
    ✅ Proper timestamp alignment with fail-fast validation
    ✅ Fixed directional accuracy using np.sign()
    ✅ Added sMAPE (Symmetric MAPE)
    ✅ Volume-Weighted MAE for price columns
    ✅ No silent data truncation
    
    Args:
        predictions_df: DataFrame with predictions (must have timestamp index or column)
        actuals_df: DataFrame with actual values (must have timestamp index or column)
        align_on: Column name to align on (default: 'timestamp')
        
    Returns:
        Dictionary of metrics for each column
        
    Raises:
        ValueError: If data cannot be aligned or is invalid
    """
    if predictions_df.empty or actuals_df.empty:
        raise ValueError("Empty DataFrames provided")
    
    # ============================================================
    # STEP 1: PERFECT ALIGNMENT (Fail-Fast Validation)
    # ============================================================
    
    # Ensure timestamp column exists
    if align_on in predictions_df.columns and align_on in actuals_df.columns:
        # Set timestamp as index if not already
        if predictions_df.index.name != align_on:
            predictions_df = predictions_df.set_index(align_on).sort_index()
        if actuals_df.index.name != align_on:
            actuals_df = actuals_df.set_index(align_on).sort_index()
    
    # Find common timestamps (intersection)
    common_index = predictions_df.index.intersection(actuals_df.index)
    
    if len(common_index) == 0:
        raise ValueError(
            f"❌ FATAL: No overlapping timestamps found!\n"
            f"   Predictions: {len(predictions_df)} rows from {predictions_df.index[0]} to {predictions_df.index[-1]}\n"
            f"   Actuals: {len(actuals_df)} rows from {actuals_df.index[0]} to {actuals_df.index[-1]}"
        )
    
    # Align both DataFrames to common timestamps
    pred_aligned = predictions_df.loc[common_index]
    actual_aligned = actuals_df.loc[common_index]
    
    # Validate alignment quality
    alignment_ratio = len(common_index) / max(len(predictions_df), len(actuals_df))
    if alignment_ratio < 0.5:
        print(f"⚠️  WARNING: Only {alignment_ratio*100:.1f}% of data aligned. Possible data quality issue.")
    
    print(f"\n✅ Timestamp Alignment: {len(common_index)} common points")
    print(f"   Predictions: {len(predictions_df)} → {len(pred_aligned)}")
    print(f"   Actuals: {len(actuals_df)} → {len(actual_aligned)}")
    
    # ============================================================
    # STEP 2: CALCULATE METRICS FOR EACH COLUMN
    # ============================================================
    
    metrics = {}
    columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Check if volume exists (needed for VW-MAE)
    has_volume = 'volume' in actual_aligned.columns
    
    for col in columns:
        # Skip if column doesn't exist in both
        if col not in pred_aligned.columns or col not in actual_aligned.columns:
            continue
        
        try:
            # Extract values as numpy arrays (FAST!)
            pred_values = pred_aligned[col].values
            actual_values = actual_aligned[col].values
            
            # Validate no NaN or Inf
            if np.any(np.isnan(pred_values)) or np.any(np.isinf(pred_values)):
                print(f"⚠️  Skipping {col}: NaN/Inf in predictions")
                continue
            
            if np.any(np.isnan(actual_values)) or np.any(np.isinf(actual_values)):
                print(f"⚠️  Skipping {col}: NaN/Inf in actuals")
                continue
            
            # --------------------------------------------------------
            # Calculate Errors
            # --------------------------------------------------------
            errors = pred_values - actual_values
            abs_errors = np.abs(errors)
            squared_errors = errors ** 2
            
            # MAE (Mean Absolute Error)
            mae = np.mean(abs_errors)
            
            # MSE (Mean Squared Error)
            mse = np.mean(squared_errors)
            
            # RMSE (Root Mean Squared Error)
            rmse = np.sqrt(mse)
            
            # --------------------------------------------------------
            # MAPE (Mean Absolute Percentage Error) - Zero Handling
            # --------------------------------------------------------
            non_zero_mask = actual_values != 0
            if non_zero_mask.sum() > 0:
                percentage_errors = abs_errors[non_zero_mask] / np.abs(actual_values[non_zero_mask])
                mape = np.mean(percentage_errors) * 100
                mape_valid = True
            else:
                mape = np.nan
                mape_valid = False
            
            # --------------------------------------------------------
            # sMAPE (Symmetric MAPE) - PERFECT FIX #3
            # --------------------------------------------------------
            denominator = np.abs(actual_values) + np.abs(pred_values)
            non_zero_mask_smape = denominator != 0
            smape = np.nan
            
            if np.sum(non_zero_mask_smape) > 0:
                smape_values = (2 * abs_errors[non_zero_mask_smape]) / denominator[non_zero_mask_smape]
                smape = np.mean(smape_values) * 100
            
            # --------------------------------------------------------
            # R² Score (Coefficient of Determination)
            # --------------------------------------------------------
            ss_res = np.sum(squared_errors)
            ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
            r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
            
            # --------------------------------------------------------
            # DIRECTIONAL ACCURACY - PERFECT FIX #1
            # --------------------------------------------------------
            if col in ['open', 'high', 'low', 'close'] and len(pred_values) > 1:
                # Use np.sign() to correctly handle zero changes
                pred_direction = np.sign(np.diff(pred_values))
                actual_direction = np.sign(np.diff(actual_values))
                directional_accuracy = np.mean(pred_direction == actual_direction) * 100
            else:
                directional_accuracy = np.nan
            
            # --------------------------------------------------------
            # Relative Error (normalized by actual mean)
            # --------------------------------------------------------
            actual_mean = np.mean(np.abs(actual_values))
            relative_error = (mae / actual_mean * 100) if actual_mean != 0 else np.nan
            
            # --------------------------------------------------------
            # Build Metrics Dictionary
            # --------------------------------------------------------
            metrics[col] = {
                "MAE (Mean Absolute Error)": float(mae),
                "MSE (Mean Squared Error)": float(mse),
                "RMSE (Root Mean Squared Error)": float(rmse),
                "Relative Error (%)": float(relative_error) if not np.isnan(relative_error) else "N/A",
                "R² Score": float(r2_score) if not np.isnan(r2_score) else "N/A",
            }
            
            # Add MAPE only if valid
            if mape_valid:
                metrics[col]["MAPE (%)"] = float(mape)
            else:
                metrics[col]["MAPE (%)"] = "N/A (zero actuals)"
            
            # Add sMAPE (always calculated)
            if not np.isnan(smape):
                metrics[col]["sMAPE (%)"] = float(smape)
            else:
                metrics[col]["sMAPE (%)"] = "N/A"
            
            # Add directional accuracy for price columns
            if not np.isnan(directional_accuracy):
                metrics[col]["Directional Accuracy (%)"] = float(directional_accuracy)
            
            # --------------------------------------------------------
            # VOLUME-WEIGHTED MAE FOR PRICE COLUMNS - PERFECT FIX #5
            # --------------------------------------------------------
            if col in ['open', 'high', 'low', 'close'] and has_volume:
                try:
                    actual_volume = actual_aligned['volume'].values
                    total_volume = np.sum(actual_volume)
                    
                    if total_volume > 0 and not np.any(np.isnan(actual_volume)):
                        # Weight the PRICE errors by volume
                        vw_mae = np.sum(abs_errors * actual_volume) / total_volume
                        metrics[col]["VW-MAE (Volume-Weighted Price Error)"] = float(vw_mae)
                except Exception as e:
                    print(f"⚠️  Could not calculate VW-MAE for {col}: {e}")
            
            # --------------------------------------------------------
            # Volume-Specific Metrics
            # --------------------------------------------------------
            if col == 'volume':
                # Volume-weighted volume error (self-weighted)
                total_actual_volume = np.sum(actual_values)
                if total_actual_volume > 0:
                    volume_weighted_error = np.sum(abs_errors * actual_values) / total_actual_volume
                    metrics[col]["Self-Weighted Error"] = float(volume_weighted_error)
        
        except Exception as e:
            print(f"❌ Error calculating metrics for {col}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    return metrics


def calculate_prediction_statistics_df(predictions_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate statistical summary of predictions from DataFrame.
    
    Args:
        predictions_df: DataFrame with prediction data
        
    Returns:
        Dictionary with statistical summaries
    """
    if predictions_df.empty:
        return {}
    
    stats = {}
    columns = ['open', 'high', 'low', 'close', 'volume']
    
    for col in columns:
        if col in predictions_df.columns:
            values = predictions_df[col].values
            
            # Convert to numpy array to ensure proper typing
            values_array = np.asarray(values, dtype=float)
            
            stats[col] = {
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'mean': float(np.mean(values_array)),
                'median': float(np.median(values_array)),
                'std': float(np.std(values_array)),
                'variance': float(np.var(values_array)),
                'range': float(np.max(values_array) - np.min(values_array)),
                'coefficient_of_variation': float(np.std(values_array) / np.mean(values_array)) if np.mean(values_array) != 0 else np.nan
            }
            
            # Calculate trend (linear regression slope)
            if len(values_array) > 1:
                x = np.arange(len(values_array))
                slope, _ = np.polyfit(x, values_array, 1)
                stats[col]['trend_slope'] = float(slope)
                stats[col]['trend_direction'] = 'upward' if slope > 0 else 'downward' if slope < 0 else 'flat'
    
    return stats


def save_prediction_results(
    file_path: str, 
    prediction_type: str, 
    predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    input_data: pd.DataFrame, 
    prediction_params: Dict[str, Any],
    metrics: Dict[str, Dict[str, Any]]
) -> Optional[str]:
    """
    Save prediction results to JSON with comprehensive analysis.
    
    PERFECTED VERSION:
    ✅ Uses DataFrames instead of List[Dict]
    ✅ Includes perfect metrics
    ✅ Better error handling
    ✅ Comprehensive analysis
    
    Args:
        file_path: Path to input data file
        prediction_type: Type of prediction performed
        predictions_df: DataFrame with predictions
        actuals_df: DataFrame with actual values
        input_data: Input DataFrame used for prediction
        prediction_params: Parameters used for prediction
        metrics: Error metrics dictionary
        
    Returns:
        Path to saved file or None if failed
    """
    try:
        # Create results directory
        results_dir = os.path.join(PROJECT_ROOT, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'prediction_{timestamp}.json'
        filepath = os.path.join(results_dir, filename)
        
        # Convert DataFrames to records for JSON serialization
        predictions_records = predictions_df.reset_index().to_dict('records')
        actuals_records = actuals_df.reset_index().to_dict('records') if not actuals_df.empty else []
        
        # Calculate input data statistics
        input_stats = {}
        for col in ['open', 'high', 'low', 'close']:
            if col in input_data.columns:
                input_stats[col] = {
                    'min': float(input_data[col].min()),
                    'max': float(input_data[col].max()),
                    'mean': float(input_data[col].mean()),
                    'std': float(input_data[col].std()),
                    'first_value': float(input_data[col].iloc[0]),
                    'last_value': float(input_data[col].iloc[-1])
                }
        
        # Prepare save data
        save_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'file_path': file_path,
            'prediction_type': prediction_type,
            'prediction_params': prediction_params,
            'input_data_summary': {
                'rows': len(input_data),
                'columns': list(input_data.columns),
                'statistics': input_stats,
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
            'predictions': predictions_records,
            'actuals': actuals_records,
            'analysis': {
                'prediction_statistics': calculate_prediction_statistics_df(predictions_df),
                'error_metrics': metrics
            }
        }
        
        # Comparison analysis if actual data exists
        if not actuals_df.empty and not predictions_df.empty:
            # Get first prediction and first actual for continuity check
            first_pred_idx = predictions_df.index[0]
            first_actual_idx = actuals_df.index[0]
            
            if first_pred_idx == first_actual_idx:
                first_pred = predictions_df.iloc[0]
                first_actual = actuals_df.iloc[0]
                
                save_data['analysis']['continuity'] = {
                    'timestamp': str(first_pred_idx),
                    'prediction': {
                        'open': float(first_pred['open']),
                        'high': float(first_pred['high']),
                        'low': float(first_pred['low']),
                        'close': float(first_pred['close'])
                    },
                    'actual': {
                        'open': float(first_actual['open']),
                        'high': float(first_actual['high']),
                        'low': float(first_actual['low']),
                        'close': float(first_actual['close'])
                    },
                    'gaps': {
                        'open_gap': float(abs(first_pred['open'] - first_actual['open'])),
                        'high_gap': float(abs(first_pred['high'] - first_actual['high'])),
                        'low_gap': float(abs(first_pred['low'] - first_actual['low'])),
                        'close_gap': float(abs(first_pred['close'] - first_actual['close']))
                    },
                    'gap_percentages': {
                        'open_gap_pct': float((abs(first_pred['open'] - first_actual['open']) / first_actual['open']) * 100),
                        'high_gap_pct': float((abs(first_pred['high'] - first_actual['high']) / first_actual['high']) * 100),
                        'low_gap_pct': float((abs(first_pred['low'] - first_actual['low']) / first_actual['low']) * 100),
                        'close_gap_pct': float((abs(first_pred['close'] - first_actual['close']) / first_actual['close']) * 100)
                    }
                }
        
        # Save to file with proper formatting
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ Results saved: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_model(model_key: str = 'kronos-small', device: str = 'cpu') -> Any:
    """
    Load Kronos model from local directory.
    
    Args:
        model_key: Model identifier from AVAILABLE_MODELS
        device: Device to load model on ('cpu' or 'cuda')
        
    Returns:
        KronosPredictor instance
    """
    global tokenizer, model, predictor
    
    if not MODEL_AVAILABLE:
        raise Exception('Kronos model library not available')
    
    if model_key not in AVAILABLE_MODELS:
        raise Exception(f'Unsupported model: {model_key}. Available: {list(AVAILABLE_MODELS.keys())}')
    
    model_config = AVAILABLE_MODELS[model_key]
    
    # Verify local paths exist
    if not os.path.exists(model_config['model_path']):
        raise Exception(f"Model path not found: {model_config['model_path']}")
    if not os.path.exists(model_config['tokenizer_path']):
        raise Exception(f"Tokenizer path not found: {model_config['tokenizer_path']}")
    
    print(f"📦 Loading model: {model_config['name']}")
    print(f"   Model path: {model_config['model_path']}")
    print(f"   Tokenizer path: {model_config['tokenizer_path']}")
    print(f"   Context length: {model_config['context_length']}")
    print(f"   Parameters: {model_config['params']}")
    
    try:
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
        
        print(f"✅ Model loaded successfully on {device}")
        return predictor
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise


def infer_time_frequency(timestamps: pd.Series) -> pd.Timedelta:
    """
    Infer the most common time frequency from a series of timestamps.
    
    Args:
        timestamps: Series of datetime values
        
    Returns:
        Most common time difference as Timedelta
    """
    if len(timestamps) < 2:
        return pd.Timedelta(hours=1)  # type: ignore  # type: ignore  # Default to 1 hour
    
    # Calculate all time differences
    time_diffs = timestamps.diff().dropna()
    
    # Find most common (mode)
    if len(time_diffs) > 0:
        mode_diff = time_diffs.mode()
        if len(mode_diff) > 0:
            result = mode_diff.iloc[0]
            try:
                # Convert to Timedelta and handle NaT properly
                if pd.isna(result):  # type: ignore
                    return pd.Timedelta(hours=1)  # type: ignore  # type: ignore
                # Use type: ignore to suppress the type checker warning
                # since we've already checked for NaT above
                timedelta_result: pd.Timedelta = pd.Timedelta(result)  # type: ignore
                return timedelta_result
            except (TypeError, ValueError):
                return pd.Timedelta(hours=1)  # type: ignore  # type: ignore
        else:
            result = time_diffs.median()
            try:
                # Convert to Timedelta and handle NaT properly
                if pd.isna(result):  # type: ignore
                    return pd.Timedelta(hours=1)  # type: ignore  # type: ignore
                # Use type: ignore to suppress the type checker warning
                # since we've already checked for NaT above
                timedelta_result: pd.Timedelta = pd.Timedelta(result)  # type: ignore
                return timedelta_result
            except (TypeError, ValueError):
                return pd.Timedelta(hours=1)  # type: ignore  # type: ignore
    
    return pd.Timedelta(hours=1)  # type: ignore


def predict(file_path: str, 
           lookback: int = 400, 
           pred_len: int = 120, 
           temperature: float = 1.0, 
           top_p: float = 0.9, 
           sample_count: int = 1, 
           start_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Perform time series prediction with comprehensive validation and analysis.
    
    PERFECTED VERSION:
    ✅ Returns DataFrames instead of List[Dict]
    ✅ Calculates perfect metrics
    ✅ Fixed timestamp generation
    ✅ Better error messages
    
    Args:
        file_path: Path to data file
        lookback: Number of historical points to use
        pred_len: Number of points to predict
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        sample_count: Number of samples to generate
        start_date: Optional start date for prediction window
        
    Returns:
        Tuple of (predictions_df, actuals_df, metrics)
    """
    global predictor, model
    
    if predictor is None:
        raise Exception('Model not loaded. Call load_model() first')
    
    # Load data
    print(f"📂 Loading data from: {file_path}")
    df, error = load_data_file(file_path)
    if error:
        raise Exception(error)
    
    # Type check to ensure df is not None
    if df is None:
        raise Exception('Failed to load data file')
    
    print(f"📊 Loaded {len(df)} rows")
    
    # Validate sufficient data
    min_required = lookback + (pred_len if start_date else 0)
    if len(df) < min_required:
        raise Exception(
            f'Insufficient data: need at least {min_required} rows '
            f'(lookback={lookback} + pred_len={pred_len}), got {len(df)}'
        )
    
    # Required columns
    required_cols = ['open', 'high', 'low', 'close']
    if 'volume' in df.columns:
        required_cols.append('volume')
    
    # Time period selection
    if start_date:
        print(f"📅 Using start date: {start_date}")
        start_dt = pd.to_datetime(start_date)
        mask = df['timestamps'] >= start_dt
        time_range_df = df[mask].copy()
        
        if len(time_range_df) < lookback + pred_len:
            raise Exception(
                f'Insufficient data from {start_dt.strftime("%Y-%m-%d %H:%M")}: '
                f'need {lookback + pred_len} rows, got {len(time_range_df)}'
            )
        
        x_df = time_range_df.iloc[:lookback][required_cols]
        x_timestamp = time_range_df.iloc[:lookback]['timestamps']
        y_timestamp = time_range_df.iloc[lookback:lookback+pred_len]['timestamps']
        
        # Ensure we're working with pandas Series
        time_range_timestamps: pd.Series = pd.Series(time_range_df['timestamps'])
        time_span = time_range_timestamps.iloc[lookback+pred_len-1] - time_range_timestamps.iloc[0]
        prediction_type = f"Window prediction (lookback={lookback}, pred={pred_len}, span={time_span})"
        
        # Store actual context range
        context_start = time_range_timestamps.iloc[0]
        context_end = time_range_timestamps.iloc[lookback-1]
        
    else:
        print(f"📅 Using latest data (last {lookback} points)")
        # Use the most recent data
        if len(df) < lookback:
            raise Exception(f'Insufficient data: need {lookback} rows, got {len(df)}')
        
        # If we have enough data for validation, anchor the lookback window
        # to end immediately before the last pred_len rows. Otherwise, use the
        # very last lookback window with no future actuals available.
        if len(df) >= lookback + pred_len:
            # Context window: the block right before the last pred_len rows
            x_df = df.iloc[-(lookback + pred_len):-pred_len][required_cols]
            x_timestamp = df.iloc[-(lookback + pred_len):-pred_len]['timestamps']
            # Future timestamps for alignment/actuals
            y_timestamp = df.iloc[-pred_len:]['timestamps']
            prediction_type = "Latest data prediction (validated on tail)"
            context_start = x_timestamp.iloc[0]
            context_end = x_timestamp.iloc[-1]
        else:
            # Not enough tail for validation: predict into the future
            x_df = df.iloc[-lookback:][required_cols]
            x_timestamp = df.iloc[-lookback:]['timestamps']
            y_timestamp = None
            prediction_type = "Latest data prediction"
            context_start = df['timestamps'].iloc[-lookback]
            context_end = df['timestamps'].iloc[-1]
    
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
    
    # Reset index for clean series
    x_timestamp = x_timestamp.reset_index(drop=True)
    if y_timestamp is not None:
        y_timestamp = y_timestamp.reset_index(drop=True)
    
    # Infer time frequency from the actual lookback window
    time_freq = infer_time_frequency(x_timestamp)
    print(f"⏱️  Detected time frequency: {time_freq}")
    
    # Normalize start_date to the actual first timestamp of the lookback window
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
    
    # ============================================================
    # PERFECT FIX: Use actual timestamps from data for alignment
    # ============================================================
    
    # Get the actual timestamps that should be used for predictions
    # These should be the timestamps that come after the lookback window
    if y_timestamp is not None:
        # Use the real future timestamps that immediately follow the context window
        actual_prediction_timestamps = pd.Series(y_timestamp).copy()
    else:
        # Fallback: generate future timestamps if not enough data for validation
        last_timestamp = x_timestamp.iloc[-1]
        actual_prediction_timestamps = pd.date_range(
            start=last_timestamp + time_freq,
            periods=pred_len,
            freq=time_freq
        )
    
    # ============================================================
    # Build Predictions DataFrame
    # ============================================================
    
    predictions_df = pred_df.copy()
    # Reset index to ensure clean assignment
    predictions_df = predictions_df.reset_index(drop=True)
    # Assign timestamps properly
    predictions_df['timestamp'] = actual_prediction_timestamps.values
    predictions_df = predictions_df.set_index('timestamp')
    
    print(f"✅ Prediction complete: {pred_len} points generated")
    
    # ============================================================
    # Build Actuals DataFrame (if available)
    # ============================================================
    
    actuals_df = pd.DataFrame()
    
    if start_date:
        start_dt = pd.to_datetime(start_date)
        mask = df['timestamps'] >= start_dt
        time_range_df = df[mask]
        
        if len(time_range_df) >= lookback + pred_len:
            actuals_df = time_range_df.iloc[lookback:lookback+pred_len].copy()
            actuals_df = actuals_df.rename(columns={'timestamps': 'timestamp'})
            actuals_df = actuals_df.set_index('timestamp')
            print(f"📊 Loaded {len(actuals_df)} actual data points for comparison")
    else:
        if y_timestamp is not None:
            # Select the last pred_len rows as actuals to match predicted timestamps
            actuals_df = df.iloc[-pred_len:].copy()
            actuals_df = actuals_df.rename(columns={'timestamps': 'timestamp'})
            actuals_df = actuals_df.set_index('timestamp')
            print(f"📊 Loaded {len(actuals_df)} actual data points for comparison")
    
    # ============================================================
    # CALCULATE PERFECT METRICS
    # ============================================================
    
    metrics: Dict[str, Dict[str, Any]] = {}
    
    if not actuals_df.empty:
        print("\n" + "="*60)
        print("CALCULATING PERFECT ERROR METRICS")
        print("="*60)
        
        try:
            metrics = calculate_error_metrics(predictions_df, actuals_df)
            
            # Print beautiful metrics table
            print("\n📊 ERROR METRICS SUMMARY:\n")
            for col, col_metrics in metrics.items():
                print(f"┌─ {col.upper()} " + "─"*(55-len(col)))
                for metric_name, value in col_metrics.items():
                    if isinstance(value, float):
                        print(f"│  {metric_name:45s}: {value:12.6f}")
                    else:
                        print(f"│  {metric_name:45s}: {value:>12}")
                print("└" + "─"*58)
                print()
        
        except Exception as e:
            print(f"❌ Error calculating metrics: {e}")
            import traceback
            traceback.print_exc()
    
    # ============================================================
    # Save Results
    # ============================================================
    
    save_prediction_results(
        file_path=file_path,
        prediction_type=prediction_type,
        predictions_df=predictions_df,
        actuals_df=actuals_df,
        input_data=x_df,
        prediction_params={
            'lookback': lookback,
            'pred_len': pred_len,
            'temperature': temperature,
            'top_p': top_p,
            'sample_count': sample_count,
            'start_date': derived_start_date,
            'end_date': derived_end_date,
            'time_frequency': str(time_freq)
        },
        metrics=metrics
    )
    
    # ============================================================
    # Print Summary
    # ============================================================
    
    print(f"\n{'='*60}")
    print("PREDICTION SUMMARY")
    print(f"{'='*60}")
    print(f"Predictions generated: {len(predictions_df)}")
    print(f"Actual data available: {len(actuals_df)}")
    
    if not actuals_df.empty:
        print(f"\n📊 Quick accuracy check (first prediction):")
        first_pred = predictions_df.iloc[0]
        first_actual = actuals_df.iloc[0]
        for key in ['open', 'high', 'low', 'close']:
            if key in first_pred.index and key in first_actual.index:
                error = abs(first_pred[key] - first_actual[key])
                error_pct = (error / first_actual[key] * 100) if first_actual[key] != 0 else 0
                print(f"  {key:6s}: pred={first_pred[key]:.2f}, actual={first_actual[key]:.2f}, "
                      f"error={error:.2f} ({error_pct:.2f}%)")
    
    print(f"{'='*60}\n")
    
    return predictions_df, actuals_df, metrics


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("KRONOS PIPELINE - PERFECTED VERSION")
    print("=" * 60)
    print(f"Model availability: {MODEL_AVAILABLE}")
    print(f"Project root: {PROJECT_ROOT}")
    print()
    
    # Example usage
    try:
        # 1. Load model
        print("Step 1: Loading model...")
        load_model(model_key='kronos-mini', device='cpu')
        print()
        
        # 2. Run prediction
        print("Step 2: Running prediction...")
        data_path = os.path.join(PROJECT_ROOT, 'data', 'test.csv')
        
        predictions_df, actuals_df, metrics = predict(
            file_path=data_path,
            lookback=400,
            pred_len=120,
            temperature=1.0,
            top_p=0.9
        )
        
        print()
        print(f"📈 Predictions: {len(predictions_df)} points")
        print(f"📊 Actuals: {len(actuals_df)} points")
        print(f"📉 Metrics calculated: {len(metrics)} columns")
        print(f"✅ Pipeline complete!")
        
        # Display predictions
        print("\n" + "="*60)
        print("FIRST 5 PREDICTIONS")
        print("="*60)
        print(predictions_df.head())
        
        if not actuals_df.empty:
            print("\n" + "="*60)
            print("FIRST 5 ACTUALS")
            print("="*60)
            print(actuals_df.head())
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()