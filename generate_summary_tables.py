import argparse
import os
import glob
import yaml
from typing import List, Dict, Any

def format_sci(val: Any) -> str:
    """Formats floating point values to scientific notation or standard format."""
    if val is None or val == "N/A":
        return "N/A"
    try:
        val_f = float(val)
        if abs(val_f) < 1e-3 or abs(val_f) >= 1e4:
            return f"{val_f:.4e}"
        return f"{val_f:.4f}"
    except (ValueError, TypeError):
        return str(val)

def parse_metadata_files(output_dir: str) -> List[Dict[str, Any]]:
    """Scans subdirectories in output_dir for metadata.yaml files and extracts key metrics."""
    records = []
    yaml_paths = glob.glob(os.path.join(output_dir, "*", "metadata.yaml"))

    for yaml_path in yaml_paths:
        try:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)

            config = data.get("config", {})
            results = data.get("results", {})
            config_name = os.path.basename(os.path.dirname(yaml_path))

            example = config.get("EXAMPLE", "N/A")
            epsilon = config.get("EPSILON", "N/A")

            for solver_name, solver_info in results.items():
                if not isinstance(solver_info, dict):
                    continue

                if solver_name == "PINN":
                    method_str = "PINN" if not config.get("RPINN") else "R-PINN"
                    mesh_sampler = str(solver_info.get("sampler_type", config.get("SAMPLER_TYPE", "uniform")))
                    dofs_params = solver_info.get("trainable_parameters", "N/A")
                elif solver_name == "KAN":
                    spline_type = solver_info.get("spline_type", config.get("KAN_SPLINE_TYPE", "nurbs")).upper()
                    method_str = f"{spline_type}-KAN" if not config.get("RPINN") else f"R-{spline_type}-KAN"
                    mesh_sampler = str(solver_info.get("sampler_type", config.get("SAMPLER_TYPE", "uniform")))
                    dofs_params = solver_info.get("trainable_parameters", "N/A")
                elif solver_name == "IGA":
                    method_str = f"IGA-{str(solver_info.get('method', 'standard')).upper()}"
                    mesh_sampler = str(solver_info.get("mesh_type", "uniform"))
                    dofs_params = solver_info.get("degrees_of_freedom", "N/A")
                else:
                    method_str = solver_name
                    mesh_sampler = "N/A"
                    dofs_params = "N/A"

                records.append({
                    "config_name": config_name,
                    "example": example,
                    "epsilon": epsilon,
                    "solver": solver_name,
                    "method": method_str,
                    "mesh_sampler": mesh_sampler,
                    "dofs_params": dofs_params,
                    "l2_error": solver_info.get("final_l2_error", "N/A"),
                    "h1_error": solver_info.get("final_h1_error", "N/A"),
                    "linf_error": solver_info.get("final_linf_error", "N/A"),
                    "time_s": solver_info.get("elapsed_seconds", "N/A")
                })
        except Exception as e:
            print(f"Warning: Failed to parse metadata file {yaml_path}: {e}")

    # Sort by example, epsilon, solver
    records.sort(key=lambda r: (str(r["example"]), str(r["epsilon"]), r["solver"], r["method"]))
    return records

def generate_markdown_table(records: List[Dict[str, Any]]) -> str:
    """Generates a formatted GitHub-style Markdown comparison table."""
    headers = [
        "Config / Run", "Problem", "Epsilon", "Method", "Mesh / Sampler",
        "DoFs / Params", "L2 Error", "H1 Error", "Linf Peak Error", "Time (s)"
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]

    for r in records:
        row = [
            str(r["config_name"]),
            f"Ex {r['example']}",
            str(r["epsilon"]),
            str(r["method"]),
            str(r["mesh_sampler"]),
            str(r["dofs_params"]),
            format_sci(r["l2_error"]),
            format_sci(r["h1_error"]),
            format_sci(r["linf_error"]),
            format_sci(r["time_s"])
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)

def generate_latex_table(records: List[Dict[str, Any]]) -> str:
    """Generates a publication-ready LaTeX table using booktabs package."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Performance and Error Norm Comparison across Solvers}",
        r"\label{tab:solver_comparison}",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"Config & Problem & $\epsilon$ & Method & Mesh/Sampler & DoFs/Params & $L^2$ Error & $H^1$ Semi-Norm & $L^\infty$ Max Error \\",
        r"\midrule"
    ]

    for r in records:
        config_clean = str(r["config_name"]).replace("_", r"\_")
        method_clean = str(r["method"]).replace("_", r"\_")
        mesh_clean = str(r["mesh_sampler"]).replace("_", r"\_")
        row_str = (
            f"{config_clean} & Ex {r['example']} & {r['epsilon']} & {method_clean} & "
            f"{mesh_clean} & {r['dofs_params']} & {format_sci(r['l2_error'])} & "
            f"{format_sci(r['h1_error'])} & {format_sci(r['linf_error'])} \\\\"
        )
        lines.append(row_str)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Generate Markdown and LaTeX summary comparison tables from metadata.yaml files.")
    parser.add_argument("--output_dir", type=str, default="output", help="Path to output directory containing experiment subfolders.")
    parser.add_argument("--save_md", type=str, default=None, help="Path to save Markdown summary table.")
    parser.add_argument("--save_tex", type=str, default=None, help="Path to save LaTeX booktabs summary table.")
    args = parser.parse_args()

    records = parse_metadata_files(args.output_dir)
    if not records:
        print(f"No experiment metadata found in '{args.output_dir}'.")
        return

    md_table = generate_markdown_table(records)
    print("\n=== Experiment Performance & Error Summary Table ===")
    print(md_table)

    save_md = args.save_md if args.save_md else os.path.join(args.output_dir, "summary.md")
    save_tex = args.save_tex if args.save_tex else os.path.join(args.output_dir, "summary_table.tex")

    os.makedirs(os.path.dirname(os.path.abspath(save_md)), exist_ok=True)
    with open(save_md, "w") as f:
        f.write(md_table + "\n")
    print(f"\nMarkdown summary saved to {save_md}")

    tex_table = generate_latex_table(records)
    os.makedirs(os.path.dirname(os.path.abspath(save_tex)), exist_ok=True)
    with open(save_tex, "w") as f:
        f.write(tex_table + "\n")
    print(f"LaTeX summary table saved to {save_tex}")

if __name__ == "__main__":
    main()
