#!/usr/bin/env python
"""
Unified Master Orchestrator for PINN, KAN, and IGA Benchmark Suite.

Coordinates:
  1. Single-method execution via train.py (with GPU sequential queue & CPU thread pool).
  2. Per-run concurrent diagnostic plotting via plot_run.py.
  3. Problem-level cross-method comparison suite via plot_problem_comparisons.py.
  4. Global multi-problem Pareto & synthesis suite via plot_comparison_suite.py.

Usage:
    python run_all.py --test
    python run_all.py --config-dir training_config/poisson_sine/
    python run_all.py --solvers pinn kan
"""

import os
import sys
import glob
import time
import argparse
import subprocess
import yaml
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional


def natural_sort(files: List[str]) -> List[str]:
    return sorted(files)


def find_all_configs(config_dir: str, test_mode: bool = False, solvers_filter: Optional[List[str]] = None) -> List[str]:
    """Finds all YAML configuration files in config_dir recursively."""
    if test_mode:
        test_dir = os.path.join("training_config", "test")
        if os.path.isdir(test_dir):
            yaml_files = glob.glob(os.path.join(test_dir, "**", "*.yaml"), recursive=True)
        else:
            yaml_files = glob.glob(os.path.join(config_dir, "**", "*test*.yaml"), recursive=True)
    else:
        yaml_files = glob.glob(os.path.join(config_dir, "**", "*.yaml"), recursive=True)

    # Filter out common / template configs
    filtered = []
    for yf in natural_sort(yaml_files):
        base = os.path.basename(yf)
        if base in ("common.yaml", "config_schema.yaml"):
            continue

        if solvers_filter:
            try:
                with open(yf, "r") as f:
                    data = yaml.safe_load(f) or {}
                solv = data.get("SOLVER", "").lower()
                if solv not in [s.lower() for s in solvers_filter]:
                    continue
            except Exception:
                pass

        filtered.append(yf)

    return filtered


def run_training_method(
    python_exe: str,
    config_path: str,
    wall_time: Optional[float] = None,
    skip_existing: bool = False,
    optimized: bool = False
) -> Tuple[int, float, str]:
    """Invokes train.py for a single configuration."""
    cmd = [python_exe, "train.py", "--config", config_path]
    if wall_time is not None:
        cmd.extend(["--wall-time", str(wall_time)])
    if skip_existing:
        cmd.append("--skip-existing")
    if optimized:
        cmd.append("--optimized")

    start_t = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - start_t
    return res.returncode, elapsed, config_path


def run_plot_method(python_exe: str, config_path: str) -> Tuple[int, float]:
    """Invokes plot_run.py for a completed method run."""
    cmd = [python_exe, "plot_run.py", "--config", config_path]
    start_t = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - start_t
    return res.returncode, elapsed


def run_problem_comparisons(python_exe: str, problem_dir: str) -> Tuple[int, float]:
    """Invokes plot_problem_comparisons.py for a problem directory."""
    cmd = [python_exe, "plot_problem_comparisons.py", "--problem-dir", problem_dir]
    start_t = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - start_t
    return res.returncode, elapsed


def run_global_comparisons(python_exe: str, output_dir: str = "output") -> Tuple[int, float]:
    """Invokes plot_comparison_suite.py and generate_summary_tables.py."""
    cmd1 = [python_exe, "generate_summary_tables.py", "--output-dir", output_dir]
    cmd2 = [python_exe, "plot_comparison_suite.py", "--output-dir", output_dir]
    start_t = time.time()
    rc1 = subprocess.run(cmd1).returncode
    rc2 = subprocess.run(cmd2).returncode
    elapsed = time.time() - start_t
    return max(rc1, rc2), elapsed


def main():
    parser = argparse.ArgumentParser(description="Master Orchestrator for PINN, KAN, and IGA Benchmark Suite")
    parser.add_argument("--test", action="store_true", help="Fast smoke test mode on minimal configs")
    parser.add_argument("--config-dir", type=str, default="training_config", help="Base configuration directory")
    parser.add_argument("--solvers", nargs="+", help="Filter by solver types (e.g. pinn kan iga)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs whose outcomes.npz already exists")
    parser.add_argument("--optimized", action="store_true", help="Enable performance optimizations across solvers")
    parser.add_argument("--wall-time", type=float, default=None, help="Global wall-clock time limit per method in minutes")
    parser.add_argument("--cpu-workers", type=int, default=None, help="Number of CPU worker threads for IGA")
    parser.add_argument("--sequential", action="store_true", help="Force strictly sequential execution of all tasks")

    args = parser.parse_args()
    python_exe = sys.executable
    total_start = time.time()

    # Discover target configurations
    configs = find_all_configs(args.config_dir, test_mode=args.test, solvers_filter=args.solvers)

    if not configs:
        print("No configuration files found matching criteria.")
        sys.exit(1)

    print("=" * 80)
    print("=== MASTER BENCHMARK ORCHESTRATOR ===")
    print(f"  Mode:            {'TEST (Smoke Test)' if args.test else 'FULL BENCHMARK'}")
    print(f"  Total Configs:   {len(configs)}")
    print(f"  Optimization:    {'OPTIMIZED' if args.optimized else 'BASELINE'}")
    print(f"  Skip Existing:   {args.skip_existing}")
    print("=" * 80)
    for c in configs:
        print(f"  [CONFIG] {os.path.relpath(c)}")
    print("=" * 80 + "\n")

    # Partition into GPU (PINN/KAN) and CPU (IGA)
    gpu_configs = []
    cpu_configs = []
    affected_problem_dirs = set()

    for c in configs:
        try:
            with open(c, "r") as f:
                d = yaml.safe_load(f) or {}
            solv = d.get("SOLVER", "pinn").lower()
            out_dir = d.get("output_dir", "")
            if out_dir:
                problem_dir = os.path.dirname(out_dir)
                affected_problem_dirs.add(problem_dir)
            else:
                prob = d.get("PROBLEM_NAME", "poisson_sine")
                affected_problem_dirs.add(os.path.join("output", prob))
        except Exception:
            solv = "pinn"

        if solv in ("pinn", "kan"):
            gpu_configs.append(c)
        else:
            cpu_configs.append(c)

    results = {}

    def run_single_pipeline(cfg_path):
        """Train method, then immediately run plot_run.py."""
        rel_name = os.path.relpath(cfg_path)
        print(f"\n--- [TRAIN START] {rel_name} ---")
        t_rc, t_time, _ = run_training_method(
            python_exe,
            cfg_path,
            wall_time=args.wall_time,
            skip_existing=args.skip_existing,
            optimized=args.optimized
        )

        if t_rc == 0:
            print(f"--- [PLOT START] {rel_name} ---")
            p_rc, p_time = run_plot_method(python_exe, cfg_path)
            total_time = t_time + p_time
            return cfg_path, {"status": "Success", "time": total_time}
        else:
            return cfg_path, {"status": f"Failed (Train Exit: {t_rc})", "time": t_time}

    # Execution Phase
    if args.sequential:
        # Strictly sequential
        for c in configs:
            _, res_dict = run_single_pipeline(c)
            results[c] = res_dict
    else:
        # Hardware-aware: GPU sequential stream + CPU thread pool simultaneously
        max_cpu = args.cpu_workers or max(1, min(6, (os.cpu_count() or 4) - 1))
        print(f"Hardware-aware execution: 1 GPU stream ({len(gpu_configs)} runs) + {max_cpu} CPU threads ({len(cpu_configs)} runs)\n")

        with ThreadPoolExecutor(max_workers=max_cpu + 1) as master_executor:
            futures = []

            # 1. GPU Sequential Worker
            def gpu_worker():
                for gc in gpu_configs:
                    path, r_info = run_single_pipeline(gc)
                    results[path] = r_info

            if gpu_configs:
                futures.append(master_executor.submit(gpu_worker))

            # 2. CPU Parallel Pool
            def cpu_worker_task(cc):
                path, r_info = run_single_pipeline(cc)
                results[path] = r_info

            for cc in cpu_configs:
                futures.append(master_executor.submit(cpu_worker_task, cc))

            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    print(f"Task worker raised exception: {e}")

    # Problem-Level Comparisons
    print("\n" + "=" * 80)
    print("=== GENERATING PROBLEM-LEVEL CROSS-METHOD COMPARISONS ===")
    print("=" * 80)
    for p_dir in sorted(list(affected_problem_dirs)):
        if os.path.exists(p_dir):
            print(f"Generating comparisons for: {p_dir}")
            run_problem_comparisons(python_exe, p_dir)

    # Global Comparisons
    print("\n" + "=" * 80)
    print("=== GENERATING GLOBAL BENCHMARK SYNTHESIS & METRICS ===")
    print("=" * 80)
    run_global_comparisons(python_exe, output_dir="output")

    # Summary Report
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print("=== BENCHMARK EXECUTION SUMMARY ===")
    print("=" * 80)
    print(f"Total time elapsed: {total_elapsed:.2f} seconds\n")

    success_cnt = 0
    for c, r_dict in results.items():
        status = r_dict.get("status", "Unknown")
        t = r_dict.get("time", 0.0)
        rel_c = os.path.relpath(c)
        print(f"  - {rel_c:<50}: {status:<15} ({t:.2f}s)")
        if status == "Success":
            success_cnt += 1

    print("-" * 80)
    print(f"Total Completed: {success_cnt}/{len(configs)} configurations succeeded.")
    print("=" * 80)


if __name__ == "__main__":
    main()
