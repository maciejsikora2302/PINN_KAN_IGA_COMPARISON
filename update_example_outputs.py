#!/usr/bin/env python
"""
Update Example Outputs Showcase Gallery.
Copies representative high-quality benchmark visualization artifacts from output/
into example_outputs/ for documentation embedding in README.md.

Usage:
    python update_example_outputs.py
    python update_example_outputs.py --output-dir output --target-dir example_outputs
"""

import argparse
import os
import shutil


def update_gallery(output_dir: str = "output", target_dir: str = "example_outputs"):
    src_root = os.path.abspath(output_dir)
    dst_root = os.path.abspath(target_dir)
    os.makedirs(dst_root, exist_ok=True)

    print("=" * 80)
    print("=== UPDATING README EXAMPLE OUTPUTS GALLERY ===")
    print(f"  Source Root: {src_root}")
    print(f"  Target Dir:  {dst_root}")
    print("=" * 80)

    # Key benchmark artifacts showcasing top-performing results
    showcase_mappings = [
        (
            os.path.join(src_root, "eriksson_johnson_eps1.0", "pinn_uniform", "surface_3d.png"),
            os.path.join(dst_root, "sample_3d_surface.png"),
            "3D Solution & Pointwise Error Elevation (PINN Uniform — H1 error 3.91e-04)"
        ),
        (
            os.path.join(src_root, "eriksson_johnson_eps1.0", "comparisons", "boundary_layer_slice_comparison.png"),
            os.path.join(dst_root, "sample_boundary_layer_slice.png"),
            "1D Boundary Layer Slice Comparison (Eriksson-Johnson eps=1.0)"
        ),
        (
            os.path.join(src_root, "eriksson_johnson_eps1.0", "comparisons", "side_by_side_solution_uniform.png"),
            os.path.join(dst_root, "sample_side_by_side_solution.png"),
            "2D Solution Fields Comparison (Uniform Methods vs Exact)"
        ),
        (
            os.path.join(src_root, "eriksson_johnson_eps1.0", "comparisons", "side_by_side_3d_solution_uniform.png"),
            os.path.join(dst_root, "sample_side_by_side_3d_solution.png"),
            "3D Elevation Solutions Multi-Panel Comparison (Uniform Methods vs Exact)"
        ),
        (
            os.path.join(src_root, "eriksson_johnson_eps1.0", "comparisons", "side_by_side_3d_error_uniform.png"),
            os.path.join(dst_root, "sample_side_by_side_3d_error.png"),
            "3D Pointwise Error Elevation Multi-Panel Comparison"
        ),
        (
            os.path.join(src_root, "global_comparisons", "pareto_efficiency_h1_vs_time.png"),
            os.path.join(dst_root, "sample_pareto_efficiency.png"),
            "Pareto Efficiency Frontier (H1 Semi-Norm vs Wall-Clock Time)"
        ),
        (
            os.path.join(src_root, "global_comparisons", "computational_efficiency_ranking.png"),
            os.path.join(dst_root, "sample_ces_ranking.png"),
            "Overall Computational Efficiency Score (CES) Ranking"
        ),
        (
            os.path.join(src_root, "global_comparisons", "formulation_performance_analysis.png"),
            os.path.join(dst_root, "sample_formulation_analysis.png"),
            "Formulation Performance Analysis (Standard vs Robust Variational)"
        ),
    ]

    copied_count = 0
    for src_path, dst_path, desc in showcase_mappings:
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied_count += 1
            print(f"  [+] Copied: {os.path.basename(dst_path):<32} -> {desc}")
        else:
            print(f"  [!] Missing: {os.path.relpath(src_path, src_root)} (Skipped)")

    print("-" * 80)
    print(f"Gallery updated: {copied_count}/{len(showcase_mappings)} files copied into {dst_root}\n")


def main():
    parser = argparse.ArgumentParser(description="Update README Example Outputs Gallery")
    parser.add_argument("--output-dir", type=str, default="output", help="Root benchmark output directory")
    parser.add_argument("--target-dir", type=str, default="example_outputs", help="Showcase target directory")
    args = parser.parse_args()

    update_gallery(output_dir=args.output_dir, target_dir=args.target_dir)


if __name__ == "__main__":
    main()
