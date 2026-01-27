#!/usr/bin/env python3
"""
Experiment Runner for Multiple Config Files
Runs main.py with different pre-defined configuration files.

FEATURES:
- Supports template variables in config paths: {group}, {lookback}, {device}, etc.
- Handles nested templates (e.g., group containing {lookback})
- Handles dynamic variables (e.g., {timestamp} filled in later)
- Automatically formats paths before running each experiment
- Works with both templated and plain configs
- Windows-safe filenames (sanitizes invalid characters)
"""

import json
import subprocess
import shutil
import argparse
from pathlib import Path
from datetime import datetime
import logging
import sys
import os
import re
import threading
import queue


def setup_logging(log_dir="logs"):
    """Setup logging to file only (console shows real-time output)"""
    Path(log_dir).mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"experiment_run_{timestamp}.log"
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler with UTF-8
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Setup root logger (file only for structured logs)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    return log_file


# ============================================================================
# TEMPLATE FORMATTING FUNCTIONS (Same as main.py)
# ============================================================================

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
    """
    if not isinstance(template, str):
        return template
    
    if skip_missing is None:
        skip_missing = set()
    
    try:
        return template.format(**kwargs)
    except KeyError as e:
        missing_var = e.args[0]
        
        # Don't warn about dynamic variables that will be filled in later
        if missing_var not in skip_missing:
            logging.warning(f"Template variable '{missing_var}' not found. Using template as-is.")
        
        return template


def build_template_variables(config):
    """
    Build a dictionary of all available template variables from config.
    
    This handles nested templates (e.g., group containing {lookback}).
    
    Available variables:
    - {group}, {lookback}, {pred_len}, {temperature}, {top_p}, {sample_count}
    - {model_key}, {device}
    - {window_size}, {step_size}
    - {date}, {time}, {datetime}
    
    Dynamic variables (filled in later):
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
        'lookback': config.get('prediction', {}).get('lookback', 0),
        'pred_len': config.get('prediction', {}).get('pred_len', 0),
        'temperature': config.get('prediction', {}).get('temperature', 0.0),
        'top_p': config.get('prediction', {}).get('top_p', 0.0),
        'sample_count': config.get('prediction', {}).get('sample_count', 0),
        
        # Model settings
        'model_key': config.get('model', {}).get('model_key', 'unknown'),
        'device': config.get('model', {}).get('device', 'cpu'),
        
        # Sequential processing settings
        'window_size': config.get('sequential_processing', {}).get('window_size', 0),
        'step_size': config.get('sequential_processing', {}).get('step_size', 0),
        
        # Date/time stamps
        'date': now.strftime('%Y%m%d'),
        'time': now.strftime('%H%M%S'),
        'datetime': now.strftime('%Y%m%d_%H%M%S'),
    }
    
    # Now handle group (which might contain template variables itself)
    group_template = config.get('prediction', {}).get('group', 'default')
    
    # Format the group string with the base variables
    try:
        formatted_group = group_template.format(**base_vars)
    except KeyError as e:
        logging.warning(f"Template variable '{e.args[0]}' not found when formatting group. Using group as-is.")
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


def format_config_templates(config):
    """
    Format all template variables in a config dictionary.
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        dict: Config with all templates formatted
    """
    formatted_config = config.copy()
    
    # Build template variables from this config
    template_vars = build_template_variables(config)
    
    # Format output paths
    if 'output' in formatted_config:
        formatted_config['output'] = format_output_config(
            formatted_config['output'], 
            template_vars,
            sanitize=True
        )
    
    return formatted_config


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

class ConfigExperimentRunner:
    """Run experiments with different pre-defined config files"""
    
    def __init__(self, config_dir="configs", backup_original=True):
        self.config_dir = Path(config_dir)
        self.main_config = Path("config.json")
        self.backup_original = backup_original
        self.original_config_backup = None
        self.results = []
        
        # Validate config directory exists
        if not self.config_dir.exists():
            raise FileNotFoundError(
                f"Config directory not found: {self.config_dir}\n"
                f"Please create it and add your config files."
            )
        
    def backup_current_config(self):
        """Backup the current config.json"""
        if self.main_config.exists() and self.backup_original:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.original_config_backup = Path(f"config.json.backup_{timestamp}")
            shutil.copy(self.main_config, self.original_config_backup)
            msg = f"[BACKUP] Original config saved to: {self.original_config_backup}"
            print(msg)
            logging.info(msg)
    
    def restore_original_config(self):
        """Restore the original config.json"""
        if self.original_config_backup and self.original_config_backup.exists():
            shutil.copy(self.original_config_backup, self.main_config)
            msg = f"[RESTORE] Original config restored from: {self.original_config_backup}"
            print(msg)
            logging.info(msg)
    
    def get_config_files(self, pattern="config_*.json"):
        """Get all config files matching the pattern"""
        config_files = sorted(self.config_dir.glob(pattern))
        
        if not config_files:
            msg = f"[ERROR] No config files found matching: {pattern}"
            print(msg)
            logging.error(msg)
            logging.error(f"[ERROR] Looking in directory: {self.config_dir.absolute()}")
            return []
        
        print(f"[FOUND] {len(config_files)} config files:")
        logging.info(f"[FOUND] {len(config_files)} config files:")
        for cf in config_files:
            print(f"        - {cf.name}")
            logging.info(f"        - {cf.name}")
        
        return config_files
    
    def validate_config(self, config_path):
        """Validate that config file is valid JSON"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return True, config
        except json.JSONDecodeError as e:
            logging.error(f"[ERROR] Invalid JSON in {config_path}: {e}")
            return False, None
        except Exception as e:
            logging.error(f"[ERROR] Error reading {config_path}: {e}")
            return False, None
    
    def _reader_thread(self, pipe, queue, pipe_name):
        """Thread function to read from a pipe and put lines in a queue"""
        try:
            for line in iter(pipe.readline, ''):
                if line:
                    queue.put((pipe_name, line))
        finally:
            pipe.close()
    
    def run_with_config(self, config_path):
        """Run main.py with a specific config file"""
        experiment_name = config_path.stem.replace("config_", "")
        
        header = "\n" + "="*70 + "\n" + f"[START] EXPERIMENT: {experiment_name}\n" + f"[CONFIG] {config_path.name}\n" + "="*70
        print(header)
        logging.info(header)
        
        # Validate and load config
        is_valid, config_data = self.validate_config(config_path)
        if not is_valid:
            msg = f"[SKIP] {experiment_name} - invalid config"
            print(msg)
            logging.error(msg)
            self.results.append({
                "name": experiment_name,
                "config_file": config_path.name,
                "success": False,
                "error": "Invalid config file",
                "duration": 0
            })
            return False
        
        # ====================================================================
        # FORMAT TEMPLATE VARIABLES IN CONFIG
        # ====================================================================
        try:
            # Build template variables
            template_vars = build_template_variables(config_data)
            
            # Format the config
            formatted_config = format_config_templates(config_data)
            
            # Log template formatting info
            print("\n[TEMPLATES] Formatting output paths...")
            logging.info("[TEMPLATES] Template variables:")
            for key, value in template_vars.items():
                logging.info(f"            {{{key}}}: {value}")
            
            # Show formatted paths
            if 'output' in formatted_config:
                output = formatted_config['output']
                print(f"[TEMPLATES] Summary file: {output.get('summary_file', 'N/A')}")
                print(f"[TEMPLATES] Predictions dir: {output.get('prediction_files_dir', 'N/A')}")
                logging.info(f"[TEMPLATES] Formatted summary_file: {output.get('summary_file', 'N/A')}")
                logging.info(f"[TEMPLATES] Formatted prediction_files_dir: {output.get('prediction_files_dir', 'N/A')}")
        
        except Exception as e:
            logging.warning(f"[WARNING] Template formatting failed: {e}")
            formatted_config = config_data
        
        # ====================================================================
        # LOG KEY PARAMETERS
        # ====================================================================
        try:
            params = []
            # Safely get configuration sections with fallback to empty dict
            sequential_processing = {} if formatted_config is None else formatted_config.get('sequential_processing', {})
            prediction = {} if formatted_config is None else formatted_config.get('prediction', {})
            output = {} if formatted_config is None else formatted_config.get('output', {})
            model = {} if formatted_config is None else formatted_config.get('model', {})
            
            params = [
                f"[PARAMS] Model: {model.get('model_key', 'N/A')} on {model.get('device', 'N/A')}",
                f"[PARAMS] Window Size: {sequential_processing.get('window_size', 'N/A')}",
                f"[PARAMS] Step Size: {sequential_processing.get('step_size', 'N/A')}",
                f"[PARAMS] Lookback: {prediction.get('lookback', 'N/A')}",
                f"[PARAMS] Pred Length: {prediction.get('pred_len', 'N/A')}",
                f"[PARAMS] Temperature: {prediction.get('temperature', 'N/A')}",
                f"[PARAMS] Group: {template_vars.get('group', 'N/A')}",  # ← Use formatted group
            ]
            
            for param in params:
                print(param)
                logging.info(param)
                
        except KeyError as e:
            logging.warning(f"[WARNING] Missing key in config: {e}")
        
        # ====================================================================
        # SAVE FORMATTED CONFIG TO config.json
        # ====================================================================
        try:
            with open(self.main_config, 'w', encoding='utf-8') as f:
                json.dump(formatted_config, f, indent=2)
            msg = f"[COPY] {config_path.name} -> config.json (with formatted templates)"
            print(msg)
            logging.info(msg)
        except Exception as e:
            logging.error(f"[ERROR] Failed to write config.json: {e}")
            return False
        
        # ====================================================================
        # RUN MAIN.PY
        # ====================================================================
        
        # Prepare environment with UTF-8 encoding
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        
        # Run main.py
        start_time = datetime.now()
        msg = f"[RUN] Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        print(msg)
        logging.info(msg)
        
        # Separator for console output
        print("\n" + "-"*70)
        print(f"  Running: {experiment_name}")
        print("-"*70)
        
        try:
            # Use Popen for real-time output
            process = subprocess.Popen(
                [sys.executable, "main.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                universal_newlines=True
            )
            
            # Create queue for thread-safe communication
            output_queue = queue.Queue()
            
            # Start reader threads for stdout and stderr
            stdout_thread = threading.Thread(
                target=self._reader_thread,
                args=(process.stdout, output_queue, 'stdout')
            )
            stderr_thread = threading.Thread(
                target=self._reader_thread,
                args=(process.stderr, output_queue, 'stderr')
            )
            
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()
            
            stdout_lines = []
            stderr_lines = []
            
            # Helper function to check if line is a progress bar
            def is_progress_bar(line):
                return bool(line.strip() and ('it/s]' in line or ('|' in line and '%|' in line)))
            
            # Read output from queue and process in real-time
            while process.poll() is None or not output_queue.empty():
                try:
                    pipe_name, line = output_queue.get(timeout=0.1)
                    
                    # Print to console immediately
                    if pipe_name == 'stdout':
                        print(line, end='')
                        sys.stdout.flush()
                        stdout_lines.append(line.rstrip())
                    else:  # stderr
                        print(line, end='', file=sys.stderr)
                        sys.stderr.flush()
                        stderr_lines.append(line.rstrip())
                        
                except queue.Empty:
                    continue
            
            # Wait for threads to finish
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)
            
            # Get return code
            return_code = process.returncode
            duration = (datetime.now() - start_time).total_seconds()
            success = return_code == 0
            
            # Log filtered output (without progress bars)
            filtered_stdout = [line for line in stdout_lines if line.strip() and not is_progress_bar(line)]
            if filtered_stdout:
                logging.info("[OUTPUT] STDOUT:")
                for line in filtered_stdout:
                    logging.info(f"         {line}")
            
            filtered_stderr = [line for line in stderr_lines if line.strip() and not is_progress_bar(line)]
            if filtered_stderr:
                logging.warning("[OUTPUT] STDERR:")
                for line in filtered_stderr:
                    logging.warning(f"         {line}")
            
            # Record result
            result_data = {
                "name": experiment_name,
                "config_file": config_path.name,
                "success": success,
                "duration": duration,
                "timestamp": start_time.isoformat(),
                "return_code": return_code,
                "config_params": {
                    "group": template_vars.get('group', 'N/A'),  # ← Use formatted group
                    "lookback": prediction.get('lookback', 'N/A'),
                    "pred_len": prediction.get('pred_len', 'N/A'),
                    "window_size": sequential_processing.get('window_size', 'N/A'),
                    "device": model.get('device', 'N/A'),
                    "temperature": prediction.get('temperature', 'N/A'),
                    "top_p": prediction.get('top_p', 'N/A'),
                    "sample_count": prediction.get('sample_count', 'N/A'),
                },
                "template_variables": template_vars  # ← Include all template vars
            }
            
            self.results.append(result_data)
            
            # Console separator
            print("-"*70)
            
            if success:
                msg = f"✓ {experiment_name} completed in {duration:.2f}s"
                print(msg)
                logging.info(f"[SUCCESS] {experiment_name} completed in {duration:.2f}s")
            else:
                msg = f"✗ {experiment_name} failed with return code {return_code}"
                print(msg)
                logging.error(f"[FAILED] {experiment_name} - return code {return_code}")
            
            print("-"*70 + "\n")
            
            return success
            
        except subprocess.TimeoutExpired:
            duration = (datetime.now() - start_time).total_seconds()
            logging.error(f"[TIMEOUT] {experiment_name} timed out after {duration:.2f}s")
            self.results.append({
                "name": experiment_name,
                "config_file": config_path.name,
                "success": False,
                "error": "Timeout",
                "duration": duration
            })
            return False
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logging.error(f"[ERROR] Running {experiment_name}: {e}")
            self.results.append({
                "name": experiment_name,
                "config_file": config_path.name,
                "success": False,
                "error": str(e),
                "duration": duration
            })
            return False
    
    def run_all(self, config_pattern="config_*.json", stop_on_error=False):
        """Run all experiments"""
        # Backup original config
        self.backup_current_config()
        
        # Get config files
        config_files = self.get_config_files(config_pattern)
        
        if not config_files:
            print("[ERROR] No config files to process!")
            logging.error("[ERROR] No config files to process!")
            return
        
        msg = f"\n[START] Running {len(config_files)} experiments...\n"
        print(msg)
        logging.info(msg)
        
        # Run each experiment
        for i, config_file in enumerate(config_files, 1):
            header = f"\n{'#'*70}\nEXPERIMENT {i}/{len(config_files)}\n{'#'*70}"
            print(header)
            logging.info(header)
            
            success = self.run_with_config(config_file)
            
            if not success and stop_on_error:
                msg = f"[STOP] Stopping due to error in {config_file.name}"
                print(msg)
                logging.error(msg)
                break
        
        # Restore original config
        self.restore_original_config()
        
        # Print summary
        self.print_summary()
        
        # Save summary
        self.save_summary()
    
    def print_summary(self):
        """Print experiment summary"""
        header = "\n" + "="*70 + "\nEXPERIMENT SUMMARY\n" + "="*70
        print(header)
        logging.info(header)
        
        total = len(self.results)
        successful = sum(1 for r in self.results if r["success"])
        failed = total - successful
        total_duration = sum(r["duration"] for r in self.results)
        
        summary_lines = [
            f"\nTotal Experiments: {total}",
            f"Successful: {successful}",
            f"Failed: {failed}",
            f"Total Duration: {total_duration:.2f}s ({total_duration/60:.2f} min)",
            "\nDetailed Results:",
            "-" * 70
        ]
        
        for line in summary_lines:
            print(line)
            logging.info(line)
        
        for result in self.results:
            status = "✓" if result["success"] else "✗"
            line = (
                f"{status} {result['name']:30s} | "
                f"{result['duration']:>8.2f}s | "
                f"{result['config_file']}"
            )
            print(line)
            logging.info(line)
            
            if not result["success"] and "error" in result:
                error_line = f"  Error: {result['error']}"
                print(error_line)
                logging.info(error_line)
        
        footer = "="*70
        print(footer)
        logging.info(footer)
    
    def save_summary(self, summary_dir="analysis/predictions_summary/multiconfig_summaries"):
        """Save summary to JSON file"""
        Path(summary_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = Path(summary_dir) / f"multiconfig_summary_{timestamp}.json"
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(self.results),
            "successful": sum(1 for r in self.results if r["success"]),
            "failed": sum(1 for r in self.results if not r["success"]),
            "total_duration": sum(r["duration"] for r in self.results),
            "results": self.results
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        msg = f"\n[SAVE] Summary saved to: {summary_file}"
        print(msg)
        logging.info(msg)


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments with different config files (supports template variables)"
    )
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Directory containing config files (default: configs)"
    )
    parser.add_argument(
        "--pattern",
        default="config_*.json",
        help="Pattern to match config files (default: config_*.json)"
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop running experiments if one fails"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't backup original config.json"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging()
    msg = f"[LOG] Logging to: {log_file}"
    print(msg)
    logging.info(msg)
    
    # Run experiments
    try:
        runner = ConfigExperimentRunner(
            config_dir=args.config_dir,
            backup_original=not args.no_backup
        )
        
        runner.run_all(
            config_pattern=args.pattern,
            stop_on_error=args.stop_on_error
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        logging.error(f"[ERROR] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Experiment run interrupted by user")
        logging.warning("[INTERRUPTED] Experiment run interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()