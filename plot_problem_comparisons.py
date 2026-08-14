#!/usr/bin/env python
"""
Problem-Level Cross-Method Comparison Suite CLI.
Aggregates all method runs within a given problem directory (e.g. output/poisson_sine/)
and generates convergence overlays, side-by-side grids, boundary layer slices, and markdown/csv tables.

Usage:
    python plot_problem_comparisons.py --problem-dir output/poisson_sine
"""

import argparse
import os
import sys

from src.visualization.problem_comparison import ProblemComparator


def main():
    parser = argparse.ArgumentParser(description="Problem-Level Cross-Method Comparison Suite")
    parser.add_argument(
        "--problem-dir",
        type=str,
        required=True,
        help="Path to problem output directory (e.g. output/poisson_sine)"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.problem_dir):
        print(f"Error: Problem directory not found: {args.problem_dir}")
        sys.exit(1)

    comparator = ProblemComparator(problem_dir=args.problem_dir)
    comparator.generate_all()


if __name__ == "__main__":
    main()
