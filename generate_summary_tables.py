#!/usr/bin/env python
"""
Global Benchmark Summary Table Generator.
Generates comprehensive Markdown and CSV tables synthesizing all runs across problems.

Usage:
    python generate_summary_tables.py [--output-dir output]
"""

import os
import argparse
from src.visualization.global_comparison import GlobalComparator


def main():
    parser = argparse.ArgumentParser(description="Generate Global Benchmark Summary Tables")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Root output directory containing problem subfolders (default: output)"
    )
    args = parser.parse_args()

    comparator = GlobalComparator(root_output_dir=args.output_dir)
    comparator.generate_global_metrics_table()


if __name__ == "__main__":
    main()
