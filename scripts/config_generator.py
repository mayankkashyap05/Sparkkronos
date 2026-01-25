import json
import os
import itertools
from pathlib import Path

def get_range_input(param_name, default_min, default_max, default_step=1):
    """Get range input from user for a parameter"""
    print(f"\n{param_name}:")
    print(f"Default range: {default_min} to {default_max} (step: {default_step})")
    
    use_default = input("Use default? (y/n): ").strip().lower()
    
    if use_default == 'y':
        return list(range(default_min, default_max + 1, default_step))
    else:
        min_val = float(input(f"Enter minimum {param_name}: "))
        max_val = float(input(f"Enter maximum {param_name}: "))
        step = float(input(f"Enter step size for {param_name}: "))
        
        # Generate list based on whether values are integers or floats
        if min_val == int(min_val) and max_val == int(max_val) and step == int(step):
            return list(range(int(min_val), int(max_val) + 1, int(step)))
        else:
            values = []
            current = min_val
            while current <= max_val:
                values.append(round(current, 2))
                current += step
            return values

def get_float_range_input(param_name, default_values):
    """Get range input for float parameters"""
    print(f"\n{param_name}:")
    print(f"Default values: {default_values}")
    
    use_default = input("Use default? (y/n): ").strip().lower()
    
    if use_default == 'y':
        return default_values
    else:
        min_val = float(input(f"Enter minimum {param_name}: "))
        max_val = float(input(f"Enter maximum {param_name}: "))
        step = float(input(f"Enter step size for {param_name}: "))
        
        values = []
        current = min_val
        while current <= max_val + 0.0001:  # Small epsilon for float comparison
            values.append(round(current, 2))
            current += step
        return values

def generate_config(base_config, lookback, pred_len, temperature, top_p, sample_count, config_number):
    """Generate a single config with given parameters"""
    config = json.loads(json.dumps(base_config))  # Deep copy
    
    # Update sequential_processing window_size
    config["sequential_processing"]["window_size"] = lookback + pred_len
    
    # Update prediction parameters
    config["prediction"]["lookback"] = lookback
    config["prediction"]["pred_len"] = pred_len
    config["prediction"]["temperature"] = temperature
    config["prediction"]["top_p"] = top_p
    config["prediction"]["sample_count"] = sample_count
    
    # Update group with new values
    group_template = config["prediction"]["group"]
    group = group_template.format(
        model_key=config["model"]["model_key"],
        lookback=lookback,
        pred_len=pred_len,
        temperature=temperature,
        top_p=top_p,
        sample_count=sample_count
    )
    config["prediction"]["group"] = group
    
    # Update output paths with new group
    config["output"]["summary_file"] = f"analysis/predictions_summary/{group}_summary.json"
    config["output"]["prediction_files_dir"] = f"analysis/predictions/{group}"
    config["output"]["prediction_filename_template"] = f"prediction_{group}_{{timestamp}}.json"
    
    return config

def main():
    # Base configuration template
    base_config = {
        "data": {
            "file_path": "data/processed/SOLUSD_15m_processed_3-months_log_return.csv"
        },
        "model": {
            "model_key": "kronos-mini",
            "device": "cuda"
        },
        "sequential_processing": {
            "window_size": 4,  # This will be updated
            "step_size": 1
        },
        "prediction": {
            "lookback": 3,
            "pred_len": 1,
            "temperature": 0.6,
            "top_p": 0.95,
            "sample_count": 5,
            "group": "SOL_15min-model={model_key}-lookback={lookback}-pred_len={pred_len}-temperature={temperature}-top_p={top_p}-sample_count={sample_count}"
        },
        "output": {
            "summary_file": "analysis/predictions_summary/{group}_summary.json",
            "prediction_files_dir": "analysis/predictions/{group}",
            "prediction_filename_template": "prediction_{group}_{timestamp}.json",
            "create_timestamped_subdirs": False,
            "custom_output_path": None
        }
    }
    
    print("=" * 60)
    print("CONFIG FILE GENERATOR")
    print("=" * 60)
    print("\nThis script will generate all possible combinations of:")
    print("- lookback")
    print("- pred_len")
    print("- temperature")
    print("- top_p")
    print("- sample_count")
    print("\nNote: window_size will be automatically set as lookback + pred_len")
    print("=" * 60)
    
    # Get ranges from user
    lookback_values = get_range_input("lookback", 1, 5, 1)
    pred_len_values = get_range_input("pred_len", 1, 3, 1)
    temperature_values = get_float_range_input("temperature", [0.3, 0.5, 0.7, 0.9])
    top_p_values = get_float_range_input("top_p", [0.90, 0.95, 0.99])
    sample_count_values = get_range_input("sample_count", 3, 10, 1)
    
    # Calculate total combinations
    total_combinations = (len(lookback_values) * len(pred_len_values) * 
                         len(temperature_values) * len(top_p_values) * 
                         len(sample_count_values))
    
    print(f"\n" + "=" * 60)
    print(f"Total combinations to generate: {total_combinations}")
    
    proceed = input("\nProceed with generation? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Generation cancelled.")
        return
    
    # Create output directory
    output_dir = Path("configs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all combinations
    combinations = itertools.product(
        lookback_values,
        pred_len_values,
        temperature_values,
        top_p_values,
        sample_count_values
    )
    
    config_count = 0
    
    for lookback, pred_len, temperature, top_p, sample_count in combinations:
        config_count += 1
        
        # Generate config
        config = generate_config(
            base_config,
            lookback,
            pred_len,
            temperature,
            top_p,
            sample_count,
            config_count
        )
        
        # Create filename
        filename = f"config_lb{lookback}_pl{pred_len}_t{temperature}_tp{top_p}_sc{sample_count}.json"
        filepath = output_dir / filename
        
        # Save config
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Generated: {filename} ({config_count}/{total_combinations})")
    
    print(f"\n" + "=" * 60)
    print(f"✅ Successfully generated {config_count} config files")
    print(f"📁 Saved to: {output_dir.absolute()}")
    print("=" * 60)

if __name__ == "__main__":
    main()