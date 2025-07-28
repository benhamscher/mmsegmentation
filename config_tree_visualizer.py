#!/usr/bin/env python3
"""
Enhanced config export script with tree visualization
"""

from pathlib import Path
from mmengine.config import Config
import argparse
from anytree import Node, RenderTree


def display_dict_as_tree(dictionary, name="config", max_depth=None):
    """
    Display a dictionary as a tree structure using anytree.
    
    Args:
        dictionary: Dict to visualize
        name: Root name for the tree
        max_depth: Maximum depth to display (None for unlimited)
    """
    if not isinstance(dictionary, dict):
        if hasattr(dictionary, '__dict__'):
            dictionary = dictionary.__dict__
        else:
            dictionary = dict(dictionary)
    
    def make_tree(data, parent=None, current_depth=0):
        if max_depth is not None and current_depth >= max_depth:
            return
            
        for key, value in data.items():
            if isinstance(value, dict):
                node = Node(f"{key} (dict, {len(value)} items)", parent=parent)
                make_tree(value, parent=node, current_depth=current_depth + 1)
            elif isinstance(value, (list, tuple)):
                node = Node(f"{key} ({type(value).__name__}, {len(value)} items)", parent=parent)
                # Optionally show first few items of lists
                if len(value) <= 3:
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            Node(f"[{i}] (dict, {len(item)} items)", parent=node)
                        else:
                            Node(f"[{i}] = {repr(item)}", parent=node)
            else:
                # Show the actual value for simple types
                value_str = repr(value)
                if len(value_str) > 50:
                    value_str = value_str[:47] + "..."
                node = Node(f"{key} = {value_str}", parent=parent)

    root = Node(name)
    make_tree(dictionary, parent=root)

    tree_output = []
    for pre, _, node in RenderTree(root):
        tree_output.append(f"{pre}{node.name}")
    
    return "\n".join(tree_output)


def export_config_with_tree(config_path, output_dir="./config_exports", max_tree_depth=3):
    """
    Export configuration with tree visualization
    """
    # Load the config
    cfg = Config.fromfile(config_path)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    config_name = Path(config_path).stem
    
    # Generate tree visualization
    tree_viz = display_dict_as_tree(dict(cfg), name=config_name, max_depth=max_tree_depth)
    
    # Save tree visualization
    tree_path = output_dir / f"{config_name}_tree.txt"
    with open(tree_path, 'w', encoding='utf-8') as f:
        f.write(f"Configuration Tree for: {config_path}\n")
        f.write("=" * 60 + "\n\n")
        f.write(tree_viz)
    
    print(f"✅ Tree visualization saved to: {tree_path}")
    
    # Also print tree to console with limited depth
    print(f"\n📊 Configuration Tree (depth={max_tree_depth}):")
    print("-" * 50)
    console_tree = display_dict_as_tree(dict(cfg), name=config_name, max_depth=2)
    print(console_tree)
    
    return tree_path


def compare_configs_detailed(config1_path, config2_path, output_dir="./config_exports"):
    """
    Enhanced config comparison with tree visualization
    """
    cfg1 = Config.fromfile(config1_path)
    cfg2 = Config.fromfile(config2_path)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    name1 = Path(config1_path).stem
    name2 = Path(config2_path).stem
    
    # Generate tree visualizations for both configs
    tree1 = display_dict_as_tree(dict(cfg1), name=name1, max_depth=4)
    tree2 = display_dict_as_tree(dict(cfg2), name=name2, max_depth=4)
    
    # Save individual trees
    tree1_path = output_dir / f"{name1}_detailed_tree.txt"
    tree2_path = output_dir / f"{name2}_detailed_tree.txt"
    
    with open(tree1_path, 'w', encoding='utf-8') as f:
        f.write(tree1)
    with open(tree2_path, 'w', encoding='utf-8') as f:
        f.write(tree2)
    
    def find_differences(dict1, dict2, path=""):
        diffs = []
        
        # Check for keys in dict1 but not in dict2
        for key in dict1:
            current_path = f"{path}.{key}" if path else key
            if key not in dict2:
                diffs.append({
                    'type': 'missing_in_config2',
                    'path': current_path,
                    'value': dict1[key],
                    'details': f"MISSING in config2: {current_path} = {repr(dict1[key])}"
                })
            elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                diffs.extend(find_differences(dict1[key], dict2[key], current_path))
            elif dict1[key] != dict2[key]:
                diffs.append({
                    'type': 'different',
                    'path': current_path,
                    'value1': dict1[key],
                    'value2': dict2[key],
                    'details': f"DIFFERENT: {current_path}\n  Config1: {repr(dict1[key])}\n  Config2: {repr(dict2[key])}"
                })
        
        # Check for keys in dict2 but not in dict1
        for key in dict2:
            current_path = f"{path}.{key}" if path else key
            if key not in dict1:
                diffs.append({
                    'type': 'missing_in_config1',
                    'path': current_path,
                    'value': dict2[key],
                    'details': f"MISSING in config1: {current_path} = {repr(dict2[key])}"
                })
        
        return diffs
    
    found_differences = find_differences(dict(cfg1), dict(cfg2))
    
    # Create detailed comparison report
    comparison_path = output_dir / f"detailed_comparison_{name1}_vs_{name2}.txt"
    
    with open(comparison_path, 'w', encoding='utf-8') as f:
        f.write("DETAILED CONFIGURATION COMPARISON\n")
        f.write("=" * 60 + "\n")
        f.write(f"Config1: {config1_path}\n")
        f.write(f"Config2: {config2_path}\n")
        f.write(f"Total differences found: {len(found_differences)}\n\n")
        
        # Group differences by type
        missing_in_2 = [d for d in found_differences if d['type'] == 'missing_in_config2']
        missing_in_1 = [d for d in found_differences if d['type'] == 'missing_in_config1']
        different = [d for d in found_differences if d['type'] == 'different']
        
        if missing_in_2:
            f.write(f"🔍 MISSING IN CONFIG2 ({len(missing_in_2)} items):\n")
            f.write("-" * 40 + "\n")
            for diff in missing_in_2:
                f.write(f"{diff['details']}\n\n")
        
        if missing_in_1:
            f.write(f"🔍 MISSING IN CONFIG1 ({len(missing_in_1)} items):\n")
            f.write("-" * 40 + "\n")
            for diff in missing_in_1:
                f.write(f"{diff['details']}\n\n")
        
        if different:
            f.write(f"🔍 DIFFERENT VALUES ({len(different)} items):\n")
            f.write("-" * 40 + "\n")
            for diff in different:
                f.write(f"{diff['details']}\n\n")
        
        # Add tree visualizations
        f.write("\n" + "=" * 60 + "\n")
        f.write("CONFIGURATION TREE STRUCTURES\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"CONFIG1 TREE ({name1}):\n")
        f.write("-" * 30 + "\n")
        f.write(tree1)
        f.write("\n\n")
        
        f.write(f"CONFIG2 TREE ({name2}):\n")
        f.write("-" * 30 + "\n")
        f.write(tree2)
    
    print(f"✅ Detailed comparison saved to: {comparison_path}")
    print(f"✅ Tree visualizations saved to: {tree1_path}, {tree2_path}")
    
    # Print summary to console
    print("📊 COMPARISON SUMMARY:")
    print(f"  Missing in config2: {len(missing_in_2)}")
    print(f"  Missing in config1: {len(missing_in_1)}")
    print(f"  Different values: {len(different)}")
    print(f"  Total differences: {len(found_differences)}")
    
    return found_differences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced MMSeg config analysis with tree visualization")
    parser.add_argument('config', help='Path to config file')
    parser.add_argument('--output-dir', default='./config_exports', 
                       help='Output directory for exports')
    parser.add_argument('--compare', help='Compare with another config file')
    parser.add_argument('--tree-only', action='store_true',
                       help='Only generate tree visualization')
    parser.add_argument('--max-depth', type=int, default=3,
                       help='Maximum tree depth to display')
    
    args = parser.parse_args()
    
    if args.tree_only:
        export_config_with_tree(args.config, args.output_dir, args.max_depth)
        
    elif args.compare:
        differences = compare_configs_detailed(args.config, args.compare, args.output_dir)
        
    else:
        # Generate both tree and full export
        export_config_with_tree(args.config, args.output_dir, args.max_depth)
        print("\n" + "="*50)
        print("For detailed comparison, use: --compare other_config.py")
