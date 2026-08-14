#!/usr/bin/env python
"""
High-Performance Concurrent Plot & Synthesis Suite Regenerator.
Uses Python's native concurrent.futures (ProcessPoolExecutor) for maximum cross-platform speed.

Usage:
    python regenerate_all_plots.py
    python regenerate_all_plots.py --output-dir output --workers 8
"""

import argparse
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from plot_run import plot_single_run
from src.visualization.global_comparison import GlobalComparator
from src.visualization.problem_comparison import ProblemComparator


def _worker_plot_single(run_dir: str):
    try:
        plot_single_run(run_dir, max_workers=1)
        return run_dir, True, None
    except Exception as e:
        return run_dir, False, str(e)


def _worker_problem_comparison(problem_dir: str):
    try:
        comparator = ProblemComparator(problem_dir=problem_dir)
        comparator.generate_all()
        return problem_dir, True, None
    except Exception as e:
        return problem_dir, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Regenerate All Benchmark Visualizations Concurrently")
    parser.add_argument("--output-dir", type=str, default="output", help="Root output directory")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes")
    args = parser.parse_args()

    root_output = os.path.abspath(args.output_dir)
    workers = args.workers or max(2, (os.cpu_count() or 4) - 1)
    total_start = time.time()

    print("=" * 80)
    print(f"=== PARALLEL PLOT REGENERATOR (Workers: {workers}) ===")
    print(f"  Target Root: {root_output}")
    print("=" * 80)

    # 1. Discover all method output directories containing outcomes.npz
    all_outcomes = glob.glob(os.path.join(root_output, "*", "*", "outcomes.npz"))
    method_dirs = []
    problem_dirs_set = set()

    for p in all_outcomes:
        run_dir = os.path.dirname(p)
        problem_dir = os.path.dirname(run_dir)
        problem_id = os.path.basename(problem_dir)
        method_name = os.path.basename(run_dir)

        if problem_id in ("global_comparisons", "test_runs") or method_name == "comparisons":
            continue

        method_dirs.append(run_dir)
        problem_dirs_set.add(problem_dir)

    print(f"\n[Phase 1/3] Found {len(method_dirs)} completed method runs across {len(problem_dirs_set)} problems.")
    print("Launching concurrent per-run visualizers (6 diagnostic plots each)...")

    start_p1 = time.time()
    success_cnt = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker_plot_single, m): m for m in method_dirs}
        for f in as_completed(futures):
            m_dir, ok, err = f.result()
            rel_m = os.path.relpath(m_dir, root_output)
            if ok:
                success_cnt += 1
                print(f"  [+] {rel_m:<45} -> Diagnostic Plots Ready")
            else:
                print(f"  [-] {rel_m:<45} -> ERROR: {err}")

    print(f"Phase 1 complete in {time.time() - start_p1:.2f}s ({success_cnt}/{len(method_dirs)} runs succeeded).\n")

    # 2. Problem-Level Comparisons
    print("=" * 80)
    print("[Phase 2/3] Generating Problem-Level Comparisons in Parallel...")
    print("=" * 80)

    start_p2 = time.time()
    problem_dirs = sorted(list(problem_dirs_set))
    with ProcessPoolExecutor(max_workers=min(workers, len(problem_dirs) or 1)) as executor:
        futures = {executor.submit(_worker_problem_comparison, p): p for p in problem_dirs}
        for f in as_completed(futures):
            p_dir, ok, err = f.result()
            rel_p = os.path.relpath(p_dir, root_output)
            if ok:
                print(f"  [+] Comparisons Ready for [{rel_p}]")
            else:
                print(f"  [-] Error for [{rel_p}]: {err}")

    print(f"Phase 2 complete in {time.time() - start_p2:.2f}s.\n")

    # 3. Global Comparisons
    print("=" * 80)
    print("[Phase 3/3] Generating Global Multi-Problem Benchmark Suite...")
    print("=" * 80)

    start_p3 = time.time()
    try:
        global_comp = GlobalComparator(root_output_dir=root_output)
        global_comp.generate_all()
        print(f"Phase 3 complete in {time.time() - start_p3:.2f}s.\n")
    except Exception as e:
        print(f"Error generating global comparisons: {e}")

    total_elapsed = time.time() - total_start
    print("=" * 80)
    print(f"ALL BENCHMARK VISUALIZATIONS REGENERATED SUCCESSFULLY in {total_elapsed:.2f}s!")
    print("=" * 80)


if __name__ == "__main__":
    main()
