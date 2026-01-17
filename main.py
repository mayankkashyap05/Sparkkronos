"""
KRONOS Sequential Processing Pipeline
======================================

Complete pipeline:
1. Load data (data_loader.py)
2. Load model (app.py)
3. Sequential batch processing (sequential.py → app.py)
4. Results aggregation

FEATURES:
- Supports template variables in config paths: {group}, {lookback}, {device}, etc.
- Handles nested templates (e.g., group containing {lookback})
- Handles dynamic variables (e.g., {timestamp} filled in later)
- Also supports plain paths without templates
- Automatic directory creation
- Windows-safe filenames (sanitizes invalid characters)
"""

from src.data.data_loader import load_csv_data
from src.app import load_model, predict_from_dataframe
from src.sequential import SequentialProcessor, WindowConfig
import json
import os
import re
from datetime import datetime


# Dynamic variables that will be filled in later (not available during initial formatting)
DYNAMIC_VARIABLES = {'timestamp'}


def sanitize_filename(filename):
    """
    Remove invalid characters from filename for Windows/Linux compatibility.
    
    Invalid characters: < > : " / \\ | ? *
    
    Args:
        filename (str): Filename to sanitize
    
    Returns:
        str: Sanitized filename
    """
    # Replace invalid characters with underscore
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, '_', filename)
    
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    return sanitized


def format_template_string(template, skip_missing=None, **kwargs):
    """
    Safely format a string template with provided variables.
    
    Args:
        template (str): String that may contain {variable} placeholders
        skip_missing (set): Set of variable names to skip if missing (won't show warning)
        **kwargs: Variables to substitute into the template
    
    Returns:
        str: Formatted string with variables replaced
        
    Examples:
        format_template_string("output/{group}/data.csv", group="test")
        # Returns: "output/test/data.csv"
        
        format_template_string("output/{timestamp}.csv", skip_missing={'timestamp'})
        # Returns: "output/{timestamp}.csv" (left as-is, no warning)
    """
    if skip_missing is None:
        skip_missing = set()
    
    try:
        return template.format(**kwargs)
    except KeyError as e:
        missing_var = e.args[0]
        
        # Don't warn about dynamic variables that will be filled in later
        if missing_var not in skip_missing:
            print(f"⚠️  Warning: Template variable '{missing_var}' not found in config. Using template as-is.")
        
        return template


def build_template_variables(config):
    """
    Build a dictionary of all available template variables from config.
    
    This handles nested templates (e.g., group containing {lookback}).
    
    Available variables:
    - {group}
    - {lookback}
    - {pred_len}
    - {device}
    - {model_key}
    - {window_size}
    - {step_size}
    - {temperature}
    - {top_p}
    - {sample_count}
    - {date} - current date (YYYYMMDD)
    - {time} - current time (HHMMSS)
    - {datetime} - combined date and time (YYYYMMDD_HHMMSS)
    
    Dynamic variables (filled in later by app.py):
    - {timestamp} - generated when creating prediction files
    
    Args:
        config (dict): Full configuration dictionary
    
    Returns:
        dict: Dictionary of template variables
    """
    now = datetime.now()
    
    # First, build all non-group variables
    base_vars = {
        # Prediction settings (excluding group for now)
        'lookback': config['prediction'].get('lookback', 0),
        'pred_len': config['prediction'].get('pred_len', 0),
        'temperature': config['prediction'].get('temperature', 0.0),
        'top_p': config['prediction'].get('top_p', 0.0),
        'sample_count': config['prediction'].get('sample_count', 0),
        
        # Model settings
        'model_key': config['model'].get('model_key', 'unknown'),
        'device': config['model'].get('device', 'cpu'),
        
        # Sequential processing settings
        'window_size': config['sequential_processing'].get('window_size', 0),
        'step_size': config['sequential_processing'].get('step_size', 0),
        
        # Date/time stamps (pipeline start time)
        'date': now.strftime('%Y%m%d'),
        'time': now.strftime('%H%M%S'),
        'datetime': now.strftime('%Y%m%d_%H%M%S'),
    }
    
    # Now handle group (which might contain template variables itself)
    group_template = config['prediction'].get('group', 'default')
    
    # Format the group string with the base variables
    try:
        formatted_group = group_template.format(**base_vars)
    except KeyError as e:
        print(f"⚠️  Warning: Template variable '{e.args[0]}' not found when formatting group. Using group as-is.")
        formatted_group = group_template
    
    # Add the formatted group to the variables
    base_vars['group'] = formatted_group
    
    return base_vars


def format_output_config(output_config, template_vars, sanitize=True):
    """
    Format all paths in output_config with template variables.
    
    Args:
        output_config (dict): Output configuration section
        template_vars (dict): Template variables to use for formatting
        sanitize (bool): Whether to sanitize filenames for OS compatibility
    
    Returns:
        dict: New output config with formatted paths
    """
    formatted_config = output_config.copy()
    
    # Format summary_file path
    if 'summary_file' in formatted_config:
        formatted_path = format_template_string(
            formatted_config['summary_file'],
            skip_missing=DYNAMIC_VARIABLES,
            **template_vars
        )
        if sanitize:
            # Sanitize only the filename part, keep directory separators
            parts = formatted_path.replace('\\', '/').split('/')
            parts[-1] = sanitize_filename(parts[-1])  # Sanitize filename only
            formatted_path = '/'.join(parts)
        
        formatted_config['summary_file'] = formatted_path
    
    # Format prediction_files_dir path
    if 'prediction_files_dir' in formatted_config:
        formatted_path = format_template_string(
            formatted_config['prediction_files_dir'],
            skip_missing=DYNAMIC_VARIABLES,
            **template_vars
        )
        if sanitize:
            # Sanitize directory names
            parts = formatted_path.replace('\\', '/').split('/')
            parts = [sanitize_filename(part) for part in parts if part]
            formatted_path = '/'.join(parts)
        
        formatted_config['prediction_files_dir'] = formatted_path
    
    # Format prediction_filename_template (keep {timestamp} as-is)
    if 'prediction_filename_template' in formatted_config:
        formatted_path = format_template_string(
            formatted_config['prediction_filename_template'],
            skip_missing=DYNAMIC_VARIABLES,  # ← Skip {timestamp} warning
            **template_vars
        )
        if sanitize:
            formatted_path = sanitize_filename(formatted_path)
        
        formatted_config['prediction_filename_template'] = formatted_path
    
    # Format custom_output_path if exists
    if 'custom_output_path' in formatted_config and formatted_config['custom_output_path']:
        formatted_path = format_template_string(
            formatted_config['custom_output_path'],
            skip_missing=DYNAMIC_VARIABLES,
            **template_vars
        )
        if sanitize:
            parts = formatted_path.replace('\\', '/').split('/')
            parts = [sanitize_filename(part) for part in parts if part]
            formatted_path = '/'.join(parts)
        
        formatted_config['custom_output_path'] = formatted_path
    
    return formatted_config


def ensure_directory_exists(filepath):
    """
    Create directory for a file path if it doesn't exist.
    
    Args:
        filepath (str): Full path to a file
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)


def main():
    print("="*70)
    print("  KRONOS SEQUENTIAL PROCESSING PIPELINE")
    print("="*70)
    
    # ========================================================================
    # LOAD CONFIGURATION
    # ========================================================================
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # ========================================================================
    # BUILD TEMPLATE VARIABLES & FORMAT OUTPUT PATHS
    # ========================================================================
    print("\n[SETUP] Preparing template variables...")
    print("-"*70)
    
    template_vars = build_template_variables(config)
    output_config = format_output_config(config['output'], template_vars, sanitize=True)
    
    print("✓ Available template variables:")
    for key, value in template_vars.items():
        print(f"  {{{key}}}: {value}")
    
    print(f"\n✓ Dynamic variables (filled in later):")
    print(f"  {{timestamp}}: Generated when creating prediction files")
    
    print(f"\n✓ Formatted output paths:")
    print(f"  Summary file: {output_config['summary_file']}")
    print(f"  Predictions dir: {output_config['prediction_files_dir']}")
    if '{timestamp}' in output_config.get('prediction_filename_template', ''):
        print(f"  Filename template: {output_config['prediction_filename_template']} (timestamp added per file)")
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    print("\n[STEP 1] Loading Data...")
    print("-"*70)
    
    df = load_csv_data(file_path=config['data']['file_path'])
    
    print(f"✓ Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    
    # ========================================================================
    # STEP 2: Load Model
    # ========================================================================
    print("\n[STEP 2] Loading Model...")
    print("-"*70)
    
    load_model(
        model_key=config['model']['model_key'], 
        device=config['model']['device']
    )
    
    print("✓ Model loaded and ready")
    
    # ========================================================================
    # STEP 3: Configure Sequential Processing
    # ========================================================================
    print("\n[STEP 3] Configuring Sequential Processor...")
    print("-"*70)
    
    window_config = WindowConfig(
        window_size=config['sequential_processing']['window_size'],
        step_size=config['sequential_processing']['step_size']
    )
    
    processor = SequentialProcessor(window_config)
    
    total_batches = processor.calculate_total_batches(len(df))
    print(f"✓ Configuration set")
    print(f"  Window size: {window_config.window_size}")
    print(f"  Step size: {window_config.step_size}")
    print(f"  Total batches: {total_batches}")
    
    # ========================================================================
    # STEP 4: Process Batches Sequentially
    # ========================================================================
    print("\n[STEP 4] Processing Batches...")
    print("-"*70)
    
    # Process each batch (one completes before next begins)
    for result in processor.process_batches(
        df=df,
        predict_func=predict_from_dataframe,
        lookback=config['prediction']['lookback'],
        pred_len=config['prediction']['pred_len'],
        temperature=config['prediction']['temperature'],
        top_p=config['prediction']['top_p'],
        sample_count=config['prediction']['sample_count'],
        group=template_vars['group'],  # ← Use formatted group
        output_config=output_config    # ← Use formatted config
    ):
        # Each batch is fully processed here before next iteration
        batch_num = result['batch_info']['batch_number']
        total = result['batch_info']['total_batches']
        
        if result['status'] == 'success':
            print(f"✓ Batch {batch_num}/{total}: {len(result['predictions'])} predictions")
        else:
            print(f"✗ Batch {batch_num}/{total}: Failed - {result['error']}")
    
    # ========================================================================
    # STEP 5: Summary
    # ========================================================================
    print("\n" + "="*70)
    print("  PROCESSING COMPLETE")
    print("="*70)
    
    all_results = processor.get_all_results()
    successful = processor.get_successful_results()
    failed = processor.get_failed_results()
    
    print(f"\nTotal batches: {len(all_results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        total_predictions = sum(len(r['predictions']) for r in successful)
        print(f"Total predictions generated: {total_predictions}")
    
    # Build summary
    summary = {
        'total_batches': len(all_results),
        'successful': len(successful),
        'failed': len(failed),
        'config': {
            'window_size': window_config.window_size,
            'step_size': window_config.step_size,
            'lookback': config['prediction']['lookback'],
            'pred_len': config['prediction']['pred_len'],
            'model_key': config['model']['model_key'],
            'device': config['model']['device'],
            'group': template_vars['group'],  # ← Use formatted group
            'temperature': config['prediction']['temperature'],
            'top_p': config['prediction']['top_p'],
            'sample_count': config['prediction']['sample_count'],
        },
        'template_variables': template_vars,  # ← Include all template vars
        'results': [
            {
                'batch': r['batch_info']['batch_number'],
                'status': r['status'],
                'predictions_count': len(r['predictions']) if r['status'] == 'success' else 0,
            }
            for r in all_results
        ]
    }
    
    # Save summary (create directory if needed)
    summary_filepath = output_config['summary_file']
    ensure_directory_exists(summary_filepath)
    
    with open(summary_filepath, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved to {summary_filepath}")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()


# ============================================================================
# SIMPLE USAGE EXAMPLES
# ============================================================================
#
# Example 1: Direct prediction from file
# --------------------------------------
# load_model(model_key='kronos-mini', device='cuda') 
#
# predictions, actuals, pred_df = predict_from_file(
#     file_path='data/processed/test_processed.csv',
#     group="Kronos-mini, SOL, 15min", 
#     lookback=8,
#     pred_len=2,
#     temperature=0.6,
#     top_p=0.95,
#     sample_count=5,
# )
#
# print(f"📈 Predictions: {len(predictions)} | 📊 Actuals: {len(actuals)}")
#
#
# Example 2: Using config.json
# ----------------------------
# This main.py file automatically reads from config.json
# Just run: python main.py
#
#
# Example 3: Template variables in config.json
# --------------------------------------------
# {
#   "prediction": {
#     "group": "SOL_15min-lb{lookback}-pl{pred_len}-t{temperature}",
#     "lookback": 3,
#     "pred_len": 2,
#     "temperature": 0.6
#   },
#   "output": {
#     "summary_file": "analysis/{group}_summary.json",
#     "prediction_files_dir": "analysis/predictions/{group}",
#     "prediction_filename_template": "pred_{group}_{timestamp}.json"
#   }
# }
#
# This will automatically format to:
# - summary_file: "analysis/SOL_15min-lb3-pl2-t0.6_summary.json"
# - prediction_files_dir: "analysis/predictions/SOL_15min-lb3-pl2-t0.6"
# - prediction_filename_template: "pred_SOL_15min-lb3-pl2-t0.6_{timestamp}.json"
#   (timestamp will be filled in by app.py when creating each file)