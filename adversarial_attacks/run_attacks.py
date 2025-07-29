import os
import subprocess
import time
from datetime import timedelta
from pathlib import Path
import itertools

def run_adversarial_attack(img_path, config, checkpoint, attack, epsilon, modelname, save_path, target=None):
    """Run the adversarial attack script with given parameters."""
    cmd = [
        "python", "./adversarial_attacks/inference_attack.py",
        "--img-path", img_path,
        "--config", config,
        "--checkpoint", checkpoint,
        "--out-path", save_path,
        "--dataset", "cityscapes",
        "--attack", attack,
        "--modelname", modelname,
        "--device", "cuda:1",
    ]
    
    # Only add epsilon for FGSM attacks
    if "FGSM" in attack:
        cmd.extend(["--epsilon", str(epsilon)])
    
    if target:
        cmd.extend(["--target", target])
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    # Base configuration
    base_config = {
        "img_path": "/home/maag/datasets/Cityscapes",
        "config": "/home/hamscher/projects/MA/mmsegmentation/work_dirs/my_deeplabv3plus_config/20250717_153125/vis_data/config.py",
        "checkpoint": "/home/hamscher/projects/MA/mmsegmentation/work_dirs/my_deeplabv3plus_config/iter_240000.pth",
    }
    
    # FGSM attacks that use epsilon
    fgsm_attacks = [
        "FGSM_untargeted",
        "FGSM_targeted", 
        "FGSM_untargeted_iterative",
        "FGSM_targeted_iterative",
    ]
    
    # Other attacks that don't use epsilon
    other_attacks = [
        "PGD_untarget",
        "PGD_target",
        "DAG_untarget_99",
        "DAG_target_pedestrians", 
        "DAG_target_cars",
        "DAG_target_1train",
        "ALMA_prox_untarget",
        "ALMA_prox_target",
        "smm_static",
        "smm_dynamic"
    ]
    
    epsilons = [2, 4, 8, 16]  # Only used for FGSM attacks
    
    modelnames = [
        "deeplabv3plus",
        # Add more model names as needed
    ]
    
    # Optional: different targets for targeted attacks
    targets = [
        None,  # No target (for untargeted attacks)
        # "frankfurt_000000_000294_leftImg8bit.png",  # Example target
        # Add more targets as needed
    ]
    
    # Generate combinations for FGSM attacks (with epsilon)
    fgsm_combinations = list(itertools.product(fgsm_attacks, epsilons, modelnames, targets))
    
    # Generate combinations for other attacks (epsilon set to None)
    other_combinations = list(itertools.product(other_attacks, [None], modelnames, targets))
    
    # Combine all combinations
    all_combinations = fgsm_combinations + other_combinations
    
    # Filter combinations
    filtered_combinations = []
    for attack, epsilon, modelname, target in all_combinations:
        if "targeted" in attack.lower() and target is None:
            continue  # Skip targeted attacks without targets
        if "untarget" in attack.lower() and target is not None:
            continue  # Skip untargeted attacks with targets
        filtered_combinations.append((attack, epsilon, modelname, target))
    
    total_runs = len(filtered_combinations)
    completed_runs = 0
    start_time = time.time()
    
    print(f"Starting batch adversarial attack generation with {total_runs} configurations...")
    
    for attack, epsilon, modelname, target in filtered_combinations:
        run_start = time.time()
        
        # Extract config name from config path
        config_name = Path(base_config["config"]).stem
        
        # Create save directory with appropriate naming
        if "FGSM" in attack:
            save_dir = f"/home/hamscher/datasets/Cityscapes/{config_name}/{attack}_epsilon_{epsilon}"
        else:
            save_dir = f"/home/hamscher/datasets/Cityscapes/{config_name}/{attack}"
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\nRunning configuration {completed_runs + 1}/{total_runs}:")
        print(f"  Attack: {attack}")
        print(f"  Epsilon: {epsilon if epsilon else 'N/A'}")
        print(f"  Model: {modelname}")
        print(f"  Target: {target if target else 'None'}")
        print(f"  Save path: {save_dir}")
        
        try:
            run_adversarial_attack(
                img_path=base_config["img_path"],
                config=base_config["config"],
                checkpoint=base_config["checkpoint"],
                attack=attack,
                epsilon=epsilon if epsilon else 2,  # Default epsilon for function call
                modelname=modelname,
                save_path=save_dir,
                target=target
            )
            
            print(f"✓ Configuration completed successfully")
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Configuration failed with error: {e}")
            continue
        
        # Calculate remaining time
        completed_runs += 1
        avg_time_per_run = (time.time() - start_time) / completed_runs
        remaining_runs = total_runs - completed_runs
        remaining_time = timedelta(seconds=int(avg_time_per_run * remaining_runs))
        
        # Format remaining time
        days = remaining_time.days
        hours = remaining_time.seconds // 3600
        minutes = (remaining_time.seconds % 3600) // 60
        seconds = remaining_time.seconds % 60
        time_str = f"{days}d {hours:02d}h {minutes:02d}m {seconds:02d}s"
        
        print(f"Progress: {completed_runs}/{total_runs}")
        print(f"Estimated remaining time: {time_str}")
        print("-" * 80)
    
    print(f"\nBatch adversarial attack generation completed!")
    print(f"Total configurations run: {completed_runs}/{total_runs}")
    
    total_time = timedelta(seconds=int(time.time() - start_time))
    print(f"Total execution time: {total_time}")

if __name__ == "__main__":
    main()