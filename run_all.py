import os
import sys
import subprocess
import time
import re
import argparse

def natural_sort_key(filename):
    if filename == "test_config.yaml":
        return (0, 0, filename)
    match = re.match(r'^exp(\d+)', filename)
    if match:
        return (1, int(match.group(1)), filename)
    return (2, 0, filename)

def natural_sort(files):
    return sorted(files, key=natural_sort_key)

def main():
    config_dir = "training_config"
    if not os.path.exists(config_dir):
        print(f"Error: Configuration directory '{config_dir}' not found.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Run all solver configurations and generate plots.")
    parser.add_argument("--start", "-s", type=str, help="Experiment name, prefix, or number to start from.")
    args = parser.parse_args()

    # Find all yaml/yml files, skipping common.yaml and original.yaml
    configs = []
    for f in os.listdir(config_dir):
        if (f.endswith(".yaml") or f.endswith(".yml")) and f not in ("common.yaml", "original.yaml"):
            configs.append(f)
    
    configs = natural_sort(configs)

    if not configs:
        print(f"No configuration files found in '{config_dir}' (excluding common.yaml and original.yaml).")
        return

    # Handle start-from logic
    if args.start:
        start_idx = None
        # 1. Exact match
        for idx, c in enumerate(configs):
            if c == args.start:
                start_idx = idx
                break
        
        # 2. Prefix match
        if start_idx is None:
            for idx, c in enumerate(configs):
                if c.startswith(args.start):
                    start_idx = idx
                    break
        
        # 3. Number match (e.g. "5" -> "exp5_")
        if start_idx is None and args.start.isdigit():
            target_prefix = f"exp{args.start}_"
            for idx, c in enumerate(configs):
                if c.startswith(target_prefix):
                    start_idx = idx
                    break

        if start_idx is None:
            print(f"Error: Starting configuration matching '{args.start}' was not found.")
            print("Available configurations:")
            for c in configs:
                print(f"  - {c}")
            sys.exit(1)
            
        print(f"Starting execution from configuration: {configs[start_idx]}")
        configs = configs[start_idx:]

    print(f"Found {len(configs)} configurations to run:\n" + "\n".join(f"-> {c}" for c in configs) + "\n")
    
    results = {}
    python_exe = sys.executable  # Use current Python interpreter

    total_start = time.time()

    for idx, config in enumerate(configs, 1):
        print("=" * 80)
        print(f"[{idx}/{len(configs)}] Processing Configuration: {config}")
        print("=" * 80)
        
        config_path = os.path.join(config_dir, config)
        
        # 1. Run training
        print(f"\n--- Running Training for {config} ---")
        train_start = time.time()
        train_cmd = [python_exe, "train.py", "--config", config_path]
        train_res = subprocess.run(train_cmd)
        train_time = time.time() - train_start
        
        if train_res.returncode != 0:
            print(f"Error: Training failed for configuration {config} (Exit code: {train_res.returncode})")
            results[config] = {"status": "Failed (Training)", "time": train_time}
            continue
            
        # 2. Run plotting
        print(f"\n--- Generating Plots for {config} ---")
        plot_start = time.time()
        plot_cmd = [python_exe, "plot_parallel.py", "--config", config_path]
        plot_res = subprocess.run(plot_cmd)
        plot_time = time.time() - plot_start
        
        total_config_time = train_time + plot_time
        
        if plot_res.returncode != 0:
            print(f"Error: Plotting failed for configuration {config} (Exit code: {plot_res.returncode})")
            results[config] = {"status": "Failed (Plotting)", "time": total_config_time}
        else:
            print(f"Successfully processed configuration {config} in {total_config_time:.2f}s")
            results[config] = {"status": "Success", "time": total_config_time}

    # Final summary
    total_time = time.time() - total_start
    print("\n" + "=" * 80)
    print("RUN ALL CONFIGS SUMMARY")
    print("=" * 80)
    print(f"Total time elapsed: {total_time:.2f} seconds\n")
    
    success_count = 0
    for config, info in results.items():
        status = info["status"]
        elapsed = info["time"]
        print(f"  - {config:<25}: {status:<20} (took {elapsed:.2f}s)")
        if status == "Success":
            success_count += 1
            
    print("-" * 80)
    print(f"Completed: {success_count}/{len(configs)} configurations succeeded.")
    print("=" * 80)

if __name__ == "__main__":
    main()
