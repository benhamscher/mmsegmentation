#!/usr/bin/env python3
"""
Script to export MMSegmentation config state similar to R's save.image()
"""

import json
import pickle
from pathlib import Path
from mmengine.config import Config
import argparse


def export_config_state(config_path, output_dir="./config_exports"):
    """
    Export all configuration variables to various formats
    Similar to R's save.image() functionality
    """
    
    # Load the config
    cfg = Config.fromfile(config_path)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    config_name = Path(config_path).stem
    
    # 1. Export as JSON (human readable)
    json_path = output_dir / f"{config_name}_state.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dict(cfg), f, indent=2, default=str)
    print(f"✅ Exported JSON state to: {json_path}")
    
    # 2. Export as Python pickle (complete state)
    pickle_path = output_dir / f"{config_name}_state.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(dict(cfg), f)
    print(f"✅ Exported pickle state to: {pickle_path}")
    
    # 3. Export as Python file (executable)
    python_path = output_dir / f"{config_name}_state.py"
    with open(python_path, 'w', encoding='utf-8') as f:
        f.write("# Exported configuration state\n")
        f.write("# Can be imported as: from config_state import *\n\n")
        
        def write_variable(key, value, indent=0):
            spaces = "    " * indent
            if isinstance(value, dict):
                f.write(f"{spaces}{key} = {{\n")
                for k, v in value.items():
                    if isinstance(v, dict):
                        f.write(f"{spaces}    '{k}': {{\n")
                        for k2, v2 in v.items():
                            f.write(f"{spaces}        '{k2}': {repr(v2)},\n")
                        f.write(f"{spaces}    }},\n")
                    else:
                        f.write(f"{spaces}    '{k}': {repr(v)},\n")
                f.write(f"{spaces}}}\n\n")
            else:
                f.write(f"{spaces}{key} = {repr(value)}\n\n")
        
        for key, value in dict(cfg).items():
            write_variable(key, value)
    
    print(f"✅ Exported Python state to: {python_path}")
    
    # 4. Export variable summary (like R's ls())
    summary_path = output_dir / f"{config_name}_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Configuration Summary for: {config_path}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Variables defined:\n")
        f.write("-" * 20 + "\n")
        config_dict = dict(cfg)
        for key, value in config_dict.items():
            f.write(f"{key:<25} : {type(value).__name__}\n")
        
        f.write(f"\nTotal variables: {len(config_dict)}\n\n")
        
        # Show detailed structure for complex objects
        for key, value in config_dict.items():
            if isinstance(value, dict):
                f.write(f"\n{key} structure:\n")
                f.write("-" * (len(key) + 11) + "\n")
                for k, v in value.items():
                    f.write(f"  {k}: {type(v).__name__}\n")
    
    print(f"✅ Exported summary to: {summary_path}")
    
    return {
        'json': json_path,
        'pickle': pickle_path, 
        'python': python_path,
        'summary': summary_path
    }


def load_config_state(pickle_path):
    """
    Load configuration state from pickle file
    Similar to R's load()
    """
    with open(pickle_path, 'rb') as f:
        config_dict = pickle.load(f)
    
    print(f"✅ Loaded configuration state from: {pickle_path}")
    print(f"Available variables: {list(config_dict.keys())}")
    
    return config_dict


def compare_configs(config1_path, config2_path, output_dir="./config_exports"):
    """
    Compare two configurations and export differences
    Similar to R's all.equal()
    """
    cfg1 = Config.fromfile(config1_path)
    cfg2 = Config.fromfile(config2_path)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    name1 = Path(config1_path).stem
    name2 = Path(config2_path).stem
    
    diff_path = output_dir / f"diff_{name1}_vs_{name2}.txt"
    
    def find_differences(dict1, dict2, path=""):
        diffs = []
        
        # Check for keys in dict1 but not in dict2
        for key in dict1:
            current_path = f"{path}.{key}" if path else key
            if key not in dict2:
                diffs.append(f"MISSING in config2: {current_path} = {dict1[key]}")
            elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                diffs.extend(find_differences(dict1[key], dict2[key], current_path))
            elif dict1[key] != dict2[key]:
                diffs.append(f"DIFFERENT: {current_path}")
                diffs.append(f"  Config1: {dict1[key]}")
                diffs.append(f"  Config2: {dict2[key]}")
        
        # Check for keys in dict2 but not in dict1
        for key in dict2:
            current_path = f"{path}.{key}" if path else key
            if key not in dict1:
                diffs.append(f"MISSING in config1: {current_path} = {dict2[key]}")
        
        return diffs
    
    found_differences = find_differences(dict(cfg1), dict(cfg2))
    
    with open(diff_path, 'w', encoding='utf-8') as f:
        f.write("Configuration Comparison\n")
        f.write(f"Config1: {config1_path}\n")
        f.write(f"Config2: {config2_path}\n")
        f.write("=" * 60 + "\n\n")
        
        if found_differences:
            f.write(f"Found {len(found_differences)} differences:\n\n")
            for diff in found_differences:
                f.write(f"{diff}\n")
        else:
            f.write("No differences found!\n")
    
    print(f"✅ Comparison saved to: {diff_path}")
    return found_differences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MMSeg config state")
    parser.add_argument('config', help='Path to config file')
    parser.add_argument('--output-dir', default='./config_exports', 
                       help='Output directory for exports')
    parser.add_argument('--compare', help='Compare with another config file')
    parser.add_argument('--load', help='Load state from pickle file')
    
    args = parser.parse_args()
    
    if args.load:
        state = load_config_state(args.load)
        # You can now access variables like: state['model'], state['train_cfg'], etc.
        
    elif args.compare:
        config_differences = compare_configs(args.config, args.compare, args.output_dir)
        print(f"\nFound {len(config_differences)} differences")
        
    else:
        export_paths = export_config_state(args.config, args.output_dir)
        print("\n✅ All exports completed successfully!")
        print(f"Files created: {list(export_paths.values())}")
