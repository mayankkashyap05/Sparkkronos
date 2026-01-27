#!/usr/bin/env python3
"""
High-Performance Batch Experiment Runner (Feature Complete)
=========================================================
Runs multiple configuration files using GPU-accelerated batch processing.
Produces identical data inputs and reporting artifacts as the sequential runner.

FEATURES:
1. Speed: Uses process_sequential_batch() (Batch Size 64+).
2. Memory: Smartly reuses loaded models.
3. Parity: Generates the same 'multiconfig_summary' JSON and output structure as sequential.

Usage:
    python multi_config_run_batch.py --dir configs --pattern "config_*.json"
"""

import json
import os
import sys
import logging
import pandas as pd
import time
import re
import traceback
import argparse
from pathlib import Path
from datetime import datetime

# ============================================================================
# IMPORTS (Core Logic)
# ============================================================================
# 1. AUTO-FIX: Add 'src' to Python Path so imports work
project_root = Path(__file__).resolve().parent
src_dir = project_root / "src"

if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

try:
    from app import load_model, predict_batch_windows
    from sequential import process_sequential_batch
except ImportError as e:
    try:
        from src.app import load_model, predict_batch_windows
        from src.sequential import process_sequential_batch
    except ImportError as e2:
        print(f"CRITICAL ERROR: Could not import core modules.")
        print(f"  Error: {e2}")
        sys.exit(1)


# ============================================================================
# LOGGING & UTILS
# ============================================================================

def setup_logging(log_dir="logs"):
    Path(log_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"batch_run_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

def sanitize_filename(filename):
    """Remove invalid characters from filename"""
    # Replace invalid characters with underscore
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, '_', filename)
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    return sanitized

def format_template_string(template, **kwargs):
    """Safely format string templates"""
    if not isinstance(template, str): return template
    try:
        return template.format(**kwargs)
    except KeyError:
        return template

# ============================================================================
# CONFIGURATION TEMPLATE ENGINE
# ============================================================================

def build_template_variables(config):
    """Extract variables for path formatting, matching sequential runner logic"""
    now = datetime.now()
    
    # Base variables
    base_vars = {
        'lookback': config.get('prediction', {}).get('lookback', 0),
        'pred_len': config.get('prediction', {}).get('pred_len', 0),
        'temperature': config.get('prediction', {}).get('temperature', 0.0),
        'top_p': config.get('prediction', {}).get('top_p', 0.0),
        'sample_count': config.get('prediction', {}).get('sample_count', 0),
        'model_key': config.get('model', {}).get('model_key', 'unknown'),
        'device': config.get('model', {}).get('device', 'cpu'),
        'window_size': config.get('sequential_processing', {}).get('window_size', 0),
        'step_size': config.get('sequential_processing', {}).get('step_size', 0),
        'date': now.strftime('%Y%m%d'),
        'time': now.strftime('%H%M%S'),
        'datetime': now.strftime('%Y%m%d_%H%M%S'),
        # TRICK: Add {timestamp} as a literal string so it survives formatting
        # This allows app.py to fill it in later per-file
        'timestamp': '{timestamp}' 
    }
    
    # Resolve Group Name
    group_template = config.get('prediction', {}).get('group', 'default')
    # We format the group immediately
    base_vars['group'] = format_template_string(group_template, **base_vars)
    
    return base_vars

def prepare_output_config(config, template_vars):
    """
    Format all output paths (including filename templates) using variables.
    This ensures files are saved exactly where sequential runner would save them.
    """
    out_conf = config.get('output', {}).copy()
    
    for key in ['summary_file', 'prediction_files_dir', 'prediction_filename_template', 'custom_output_path']:
        if key in out_conf and out_conf[key]:
            # Format the string (resolves {group}, {lookback}, etc.)
            # {timestamp} remains as literal "{timestamp}" due to the trick above
            formatted = format_template_string(out_conf[key], **template_vars)
            
            # Apply Sanitization
            if key == 'prediction_filename_template':
                # Sanitize filename but preserve {timestamp} placeholder if it exists
                # We do this by temporarily removing the placeholder
                if '{timestamp}' in formatted:
                    temp = formatted.replace('{timestamp}', 'TIMESTAMP_PLACEHOLDER')
                    temp = sanitize_filename(temp)
                    formatted = temp.replace('TIMESTAMP_PLACEHOLDER', '{timestamp}')
                else:
                    formatted = sanitize_filename(formatted)
            elif key != 'custom_output_path':
                # Sanitize path components
                parts = formatted.replace('\\', '/').split('/')
                # Don't sanitize drive letters (e.g. C:)
                safe_parts = [sanitize_filename(p) if ':' not in p else p for p in parts]
                formatted = '/'.join(safe_parts)
                
            out_conf[key] = formatted

    return out_conf

# ============================================================================
# BATCH RUNNER CLASS
# ============================================================================

class BatchExperimentRunner:
    def __init__(self, config_dir="configs"):
        self.config_dir = Path(config_dir)
        self.current_model_key = None
        self.results = []
        
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory '{config_dir}' not found.")

    def run_all(self, pattern="config_*.json", batch_size=64):
        config_files = sorted(self.config_dir.glob(pattern))
        
        if not config_files:
            logging.error(f"No config files found matching '{pattern}' in {self.config_dir}")
            return

        logging.info(f"🚀 STARTING BATCH RUN: Found {len(config_files)} configurations")
        logging.info(f"⚡ GPU BATCH SIZE: {batch_size}")

        for i, config_file in enumerate(config_files, 1):
            logging.info(f"\n{'='*60}\n⚙️ EXPERIMENT {i}/{len(config_files)}: {config_file.name}\n{'='*60}")
            self.run_single_config(config_file, batch_size)

        self.print_summary()
        self.save_summary()

    def run_single_config(self, config_path, batch_size):
        start_time = time.time()
        # Initialize result entry matching Sequential structure
        result_entry = {
            "name": config_path.stem.replace("config_", ""),
            "config_file": config_path.name,
            "success": False,
            "duration": 0,
            "timestamp": datetime.now().isoformat(),
            "config_params": {},
            "template_variables": {},
            "windows_processed": 0
        }

        try:
            # 1. Load Config
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 2. Prepare Variables & Output
            vars = build_template_variables(config)
            output_conf = prepare_output_config(config, vars)
            
            # Store metadata for summary
            result_entry["template_variables"] = vars
            result_entry["config_params"] = {
                "group": vars['group'],
                "model_key": config['model']['model_key'],
                "lookback": config['prediction']['lookback'],
                "pred_len": config['prediction']['pred_len'],
                "device": config['model'].get('device', 'cpu'),
                "temperature": config['prediction']['temperature'],
                "top_p": config['prediction']['top_p'],
                "sample_count": config['prediction']['sample_count']
            }

            # 3. Smart Model Loading
            target_model = config['model']['model_key']
            device = config['model'].get('device', 'cuda')
            
            if target_model != self.current_model_key:
                logging.info(f"📥 Loading Model: {target_model} on {device}...")
                load_model(target_model, device=device)
                self.current_model_key = target_model
            else:
                logging.info(f"♻️  Reusing loaded model: {target_model}")

            # 4. Load Data
            data_path = config['data']['file_path']
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            logging.info(f"📂 Loading Data: {data_path}")
            df = pd.read_csv(data_path)
            logging.info(f"   Rows: {len(df):,}")

            # 5. Execution
            logging.info(f"🔥 Running GPU Inference (Group: {vars['group']})...")
            
            results = process_sequential_batch(
                df=df,
                predict_batch_func=predict_batch_windows,
                batch_size=batch_size,
                
                # Parameters
                window_size=config['sequential_processing']['window_size'],
                step_size=config['sequential_processing']['step_size'],
                lookback=config['prediction']['lookback'],
                pred_len=config['prediction']['pred_len'],
                temperature=config['prediction']['temperature'],
                top_p=config['prediction']['top_p'],
                sample_count=config['prediction']['sample_count'],
                
                # Output
                group=vars['group'],
                output_config=output_conf
            )

            # 6. Success
            duration = time.time() - start_time
            result_entry.update({
                "success": True,
                "duration": duration,
                "windows_processed": len(results),
                "speed": len(results) / duration if duration > 0 else 0
            })
            
            logging.info(f"✅ COMPLETED: {len(results)} windows in {duration:.2f}s ({result_entry['speed']:.1f} w/s)")

        except Exception as e:
            logging.error(f"❌ ERROR: {str(e)}")
            logging.error(traceback.format_exc())
            result_entry["error"] = str(e)
        
        self.results.append(result_entry)

    def print_summary(self):
        logging.info(f"\n{'='*60}\n📊 BATCH EXECUTION SUMMARY\n{'='*60}")
        print(f"{'CONFIG FILE':<30} | {'STATUS':<10} | {'WINDOWS':<10} | {'TIME (s)':<10}")
        print("-" * 70)
        
        total_windows = 0
        total_time = 0
        
        for r in self.results:
            status_icon = "✅" if r['success'] else "❌"
            status_text = "SUCCESS" if r['success'] else "FAILED"
            print(f"{r['config_file'][:30]:<30} | {status_icon} {status_text:<7} | {r['windows_processed']:<10} | {r['duration']:<10.2f}")
            if r['success']:
                total_windows += r['windows_processed']
                total_time += r['duration']
                
        print("-" * 70)
        if total_time > 0:
            logging.info(f"TOTAL: {total_windows} windows processed in {total_time:.2f}s")
            logging.info(f"AVG SPEED: {total_windows/total_time:.2f} windows/sec")

    def save_summary(self, summary_dir="analysis/predictions_summary/multiconfig_summaries"):
        """Save summary JSON identical to sequential runner structure"""
        Path(summary_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = Path(summary_dir) / f"multiconfig_summary_batch_{timestamp}.json"
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "runner_type": "batch_gpu",
            "total_experiments": len(self.results),
            "successful": sum(1 for r in self.results if r["success"]),
            "failed": sum(1 for r in self.results if not r["success"]),
            "total_duration": sum(r["duration"] for r in self.results),
            "results": self.results
        }
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            logging.info(f"💾 Summary saved to: {summary_file}")
        except Exception as e:
            logging.error(f"Failed to save summary: {e}")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kronos Batch Experiment Runner")
    parser.add_argument("--dir", default="configs", help="Directory containing config files")
    parser.add_argument("--pattern", default="config_*.json", help="File pattern to match")
    parser.add_argument("--batch", type=int, default=8, help="GPU Batch Size (default: 64)")
    
    args = parser.parse_args()
    
    setup_logging()
    runner = BatchExperimentRunner(config_dir=args.dir)
    
    try:
        runner.run_all(pattern=args.pattern, batch_size=args.batch)
    except KeyboardInterrupt:
        logging.warning("\n🛑 Execution interrupted by user.")