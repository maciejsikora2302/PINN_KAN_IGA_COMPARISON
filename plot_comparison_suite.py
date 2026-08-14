#!/usr/bin/env python
"""
Global Multi-Problem Benchmark Suite Entrypoint.
Scans output/ and produces Pareto frontiers, ablation comparisons, and global synthesis figures.

Usage:
    python plot_comparison_suite.py [--output-dir output]
"""

import os
import argparse
from src.visualization.global_comparison import GlobalComparator


def main():
    parser = argparse.ArgumentParser(description="Global Multi-Problem Benchmark Comparison Suite")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Root output directory containing problem subfolders (default: output)"
    )
    args = parser.parse_args()

    comparator = GlobalComparator(root_output_dir=args.output_dir)
    comparator.generate_all()


if __name__ == "__main__":
    main()
