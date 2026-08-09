import os
import sys
import yaml
import itertools
from typing import Any, Dict

def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def save_yaml(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

def main():
    # 1. Parse CLI arguments
    if "-h" in sys.argv or "--help" in sys.argv:
        print("Usage: python manage_configs.py [template_yaml_path_or_dir] [-y/--force] [--prefix PREFIX] [--proceed]")
        sys.exit(1)

    template_path = None
    force = False
    prefix_override = None
    proceed = False

    idx = 1
    while idx < len(sys.argv):
        arg = sys.argv[idx]
        if arg in ("-y", "--force"):
            force = True
        elif arg == "--prefix" and idx + 1 < len(sys.argv):
            prefix_override = sys.argv[idx + 1]
            idx += 1
        elif arg == "--proceed":
            proceed = True
        else:
            if template_path is None and not arg.startswith("-"):
                template_path = arg
        idx += 1

    if not template_path:
        template_path = "sweeps"

    if not os.path.exists(template_path):
        print(f"Error: Path '{template_path}' not found.")
        sys.exit(1)

    # 2. Load common.yaml reference
    config_dir = "training_config"
    common_path = os.path.join(config_dir, "common.yaml")
    if not os.path.exists(common_path):
        print(f"Error: Base configuration file '{common_path}' not found.")
        sys.exit(1)

    common_data = load_yaml(common_path)
    updated_common_data = dict(common_data)

    # Resolve templates list
    templates = []
    if os.path.isdir(template_path):
        for f in os.listdir(template_path):
            if (f.endswith(".yaml") or f.endswith(".yml")) and f not in ("common.yaml", "original.yaml"):
                templates.append(os.path.join(template_path, f))
    else:
        templates = [template_path]

    if not templates:
        print(f"No sweep templates found in '{template_path}'.")
        sys.exit(0)

    proposed_files = {}
    global_idx = 1

    for t_path in sorted(templates):
        # 3. Load the template
        template = load_yaml(t_path)
        global_params = template.get("global", {})
        sweep = template.get("sweep", {})
        scenarios = template.get("scenarios", [])

        # 4. Handle global parameters comparison with common.yaml
        for key, template_val in global_params.items():
            common_val = updated_common_data.get(key)
            if common_val != template_val:
                if force:
                    updated_common_data[key] = template_val
                else:
                    print(f"[{os.path.basename(t_path)}] Parameter '{key}' has value '{template_val}' in template, but '{common_val}' in common.yaml.")
                    ans = input(f"Do you want to override common.yaml globally? [y/N]: ").strip().lower()
                    if ans in ("y", "yes"):
                        updated_common_data[key] = template_val
                        print(f"-> Selected global override. '{key}' will be updated in common.yaml.")
                    else:
                        print(f"-> Leaving as local override for generated configs.")

        # 5. Generate combinations
        sweep_keys = sorted(sweep.keys())
        sweep_lists = [sweep[k] for k in sweep_keys]
        
        combinations = []
        if sweep_lists:
            for comb in itertools.product(*sweep_lists):
                combinations.append(dict(zip(sweep_keys, comb)))
        else:
            combinations = [{}]

        # Derive sweep prefix
        base_name = os.path.splitext(os.path.basename(t_path))[0]
        if base_name.startswith("sweep_"):
            sweep_prefix = base_name[6:]
        else:
            sweep_prefix = base_name

        # NN-only and IGA-only keys to prune combinations
        nn_only_keys = {
            "rpinn", "sampler_type", "sampler_gamma", "layers", "neurons_per_layer", 
            "kan_layers", "kan_neurons_per_layer", "kan_spline_type", "kan_grid_size", 
            "kan_spline_order", "activation", "epochs", "rpinn_epochs", "kan_epochs", 
            "kan_rpinn_epochs", "learning_rate", "kan_learning_rate"
        }
        iga_only_keys = {
            "iga_degree", "iga_elements", "iga_method", "iga_mesh_type", 
            "iga_adaptive_gamma", "iga_test_degree_enrichment"
        }

        # Generate scenario configurations
        for scenario in scenarios:
            custom_name = scenario.get("name")
            scenario_params = {k: v for k, v in scenario.items() if k != "name"}

            # Determine active solvers for this scenario
            global_solvers = global_params.get("SOLVERS", updated_common_data.get("SOLVERS", ["pinn", "kan", "iga"]))
            scenario_solvers = scenario.get("SOLVERS", global_solvers)
            if isinstance(scenario_solvers, str):
                scenario_solvers = [scenario_solvers]
            scenario_solvers = [s.lower() for s in scenario_solvers]

            has_nn = any(s in scenario_solvers for s in ("pinn", "kan"))
            has_iga = "iga" in scenario_solvers

            # Deduplicate combinations based on active solvers
            seen_combinations = []
            deduped_combinations = []
            for comb in combinations:
                projected = {}
                for k, v in comb.items():
                    kl = k.lower()
                    if not has_nn and kl in nn_only_keys:
                        continue
                    if not has_iga and kl in iga_only_keys:
                        continue
                    projected[k] = v
                if projected not in seen_combinations:
                    seen_combinations.append(projected)
                    deduped_combinations.append(projected)

            for projected_comb in deduped_combinations:
                file_values = {}
                for k, v in global_params.items():
                    file_values[k] = v
                for k, v in scenario_params.items():
                    file_values[k] = v
                for k, v in projected_comb.items():
                    file_values[k] = v

                file_overrides = {}
                for k, v in file_values.items():
                    if updated_common_data.get(k) != v:
                        file_overrides[k] = v

                # Deterministic filename generation with prefix
                name_part = f"{custom_name}_" if custom_name else ""
                
                # Include active solvers list in name to make it clear
                solver_part = "_".join(scenario_solvers)
                suffix = "_".join(f"{k.lower()}{v}" for k, v in sorted(projected_comb.items()))
                
                # Determine prefix: if prefix_override is set, use it. Otherwise, use sweep_prefix.
                pfx = prefix_override if prefix_override else sweep_prefix
                filename_parts = [pfx, f"exp{global_idx}", solver_part, name_part + suffix]
                filename = "_".join(part for part in filename_parts if part).strip("_") + ".yaml"

                file_path = os.path.join(config_dir, filename)
                global_idx += 1

                action = "MODIFY" if os.path.exists(file_path) else "NEW"

                proposed_files[filename] = {
                    "path": file_path,
                    "action": action,
                    "content": file_overrides
                }

    common_modified = updated_common_data != common_data
    if common_modified:
        proposed_files["common.yaml"] = {
            "path": common_path,
            "action": "MODIFY",
            "content": updated_common_data
        }

    # 6. Show summary of proposed changes
    print("\n" + "=" * 80)
    print("PROPOSED CONFIGURATION CHANGES")
    print("=" * 80)

    for filename, info in sorted(proposed_files.items()):
        action = info["action"]
        content = info["content"]
        print(f"[{action}] {os.path.join(config_dir, filename)}")
        if not content:
            print("  (Empty config - will use defaults from common.yaml)")
        else:
            for k, v in content.items():
                print(f"  {k}: {v}")
        print("-" * 50)

    # 7. Await confirmation unless forced
    if not force and not proceed:
        confirm = input("\nDo you want to apply these changes? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Cancelled. No files were written.")
            sys.exit(0)

    # 8. Write the files
    print("\nCleaning up old generated configuration files...")
    protected_files = {"common.yaml", "original.yaml", "test_config.yaml", "extended_test_config.yaml"}
    for f in os.listdir(config_dir):
        if (f.endswith(".yaml") or f.endswith(".yml")) and f not in protected_files:
            file_to_del = os.path.join(config_dir, f)
            try:
                os.remove(file_to_del)
                # Avoid verbose prints for large directory cleanup
            except Exception as e:
                print(f"Warning: Failed to delete {file_to_del}: {e}")

    print("\nWriting configuration files...")
    for filename, info in proposed_files.items():
        path = info["path"]
        content = info["content"]
        if filename == "common.yaml":
            save_yaml(path, content)
            print(f"Updated {path}")
        else:
            if not content:
                with open(path, "w") as f:
                    f.write("# Use all defaults from common.yaml\n")
            else:
                save_yaml(path, content)
            print(f"Created/Updated {path}")

    print("\nAll configuration changes applied successfully.")

if __name__ == "__main__":
    main()
