import os
import sys
import yaml
from typing import Any, Dict, List

def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def save_yaml(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

def generate_configs_from_matrix(matrix_path: str, base_out_dir: str = "training_config") -> List[str]:
    """
    Parses a declarative benchmark/test sweep matrix YAML and compiles clean,
    single-run configuration files structured by problem.
    """
    matrix_data = load_yaml(matrix_path)
    if not matrix_data:
        raise FileNotFoundError(f"Sweep matrix file not found or empty: {matrix_path}")

    global_params = matrix_data.get("global", {})
    problems = matrix_data.get("problems", [])
    methods = matrix_data.get("methods", [])
    matrix_name = matrix_data.get("name", os.path.splitext(os.path.basename(matrix_path))[0])

    is_test_matrix = "test" in matrix_name.lower() or "test" in matrix_path.lower()
    created_configs = []

    print(f"\nGenerating configurations from: {matrix_path}")
    print(f"Matrix Name: {matrix_name} | Problems: {len(problems)} | Methods: {len(methods)}")

    for prob in problems:
        prob_id = prob.get("id")
        if not prob_id:
            ex_id = prob.get("EXAMPLE", 1)
            eps = prob.get("EPSILON", 0.01)
            if ex_id == 1:
                prob_id = "poisson_sine"
            elif ex_id == 2:
                prob_id = "poisson_exp"
            else:
                prob_id = f"eriksson_johnson_eps{eps}"

        target_dir = os.path.join(base_out_dir, "test" if is_test_matrix else prob_id)
        os.makedirs(target_dir, exist_ok=True)

        for meth in methods:
            meth_name = meth.get("name")
            if not meth_name:
                continue

            config_filename = f"{meth_name}.yaml" if not is_test_matrix else f"test_{prob_id.replace('test_', '')}_{meth_name}.yaml"
            config_filepath = os.path.join(target_dir, config_filename)

            # Combine global defaults -> problem specs -> method specs
            combined: Dict[str, Any] = {}
            for k, v in global_params.items():
                combined[k] = v
            for k, v in prob.items():
                if k != "id":
                    combined[k] = v
            for k, v in meth.items():
                if k != "name":
                    combined[k] = v

            combined["METHOD_NAME"] = meth_name

            save_yaml(config_filepath, combined)
            created_configs.append(config_filepath)

    print(f"Successfully generated {len(created_configs)} configuration files.")
    return created_configs

def main():
    if len(sys.argv) > 1 and sys.argv[1] not in ("-h", "--help"):
        matrix_arg = sys.argv[1]
    else:
        matrix_arg = "sweeps"

    if os.path.isdir(matrix_arg):
        matrices = [os.path.join(matrix_arg, f) for f in sorted(os.listdir(matrix_arg)) if f.endswith((".yaml", ".yml"))]
    elif os.path.isfile(matrix_arg):
        matrices = [matrix_arg]
    else:
        matrices = [
            os.path.join("sweeps", "test_matrix.yaml"),
            os.path.join("sweeps", "benchmark_matrix.yaml")
        ]

    for m in matrices:
        if os.path.exists(m):
            generate_configs_from_matrix(m)

if __name__ == "__main__":
    main()
