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
    if len(sys.argv) < 2 or "-h" in sys.argv or "--help" in sys.argv:
        print("Usage: python manage_configs.py <template_yaml_path> [-y/--force] [--prefix PREFIX]")
        sys.exit(1)

    template_path = None
    force = False
    prefix_override = None

    idx = 1
    while idx < len(sys.argv):
        arg = sys.argv[idx]
        if arg in ("-y", "--force"):
            force = True
        elif arg == "--prefix" and idx + 1 < len(sys.argv):
            prefix_override = sys.argv[idx + 1]
            idx += 1
        else:
            if template_path is None and not arg.startswith("-"):
                template_path = arg
        idx += 1

    if not template_path:
        print("Error: Missing template configuration file path.")
        print("Usage: python manage_configs.py <template_yaml_path> [-y/--force] [--prefix PREFIX]")
        sys.exit(1)

    if not os.path.exists(template_path):
        print(f"Error: Template file '{template_path}' not found.")
        sys.exit(1)

    # 2. Load common.yaml reference
    config_dir = "training_config"
    common_path = os.path.join(config_dir, "common.yaml")
    if not os.path.exists(common_path):
        print(f"Error: Base configuration file '{common_path}' not found.")
        sys.exit(1)

    common_data = load_yaml(common_path)
    updated_common_data = dict(common_data)

    # 3. Load the template
    template = load_yaml(template_path)
    global_params = template.get("global", {})
    sweep = template.get("sweep", {})
    scenarios = template.get("scenarios", [])

    # 4. Handle single-valued parameter comparison with common.yaml
    local_global_overrides = {}
    for key, template_val in global_params.items():
        common_val = common_data.get(key)
        if common_val != template_val:
            if force:
                updated_common_data[key] = template_val
            else:
                print(f"Parameter '{key}' has value '{template_val}' in template, but '{common_val}' in common.yaml.")
                ans = input(f"Do you want to override common.yaml globally? [y/N]: ").strip().lower()
                if ans in ("y", "yes"):
                    updated_common_data[key] = template_val
                    print(f"-> Selected global override. '{key}' will be updated in common.yaml.")
                else:
                    local_global_overrides[key] = template_val
                    print(f"-> Selected local override. '{key}' will be written to generated configs where applicable.")

    # 5. Generate combinations
    sweep_keys = sorted(sweep.keys())
    sweep_lists = [sweep[k] for k in sweep_keys]
    
    combinations = []
    if sweep_lists:
        for comb in itertools.product(*sweep_lists):
            combinations.append(dict(zip(sweep_keys, comb)))
    else:
        combinations = [{}]

    proposed_files = {}

    common_modified = updated_common_data != common_data
    if common_modified:
        proposed_files["common.yaml"] = {
            "path": common_path,
            "action": "MODIFY",
            "content": updated_common_data
        }

    # Generate scenario configurations
    global_idx = 1
    for scenario in scenarios:
        custom_name = scenario.get("name")
        scenario_params = {k: v for k, v in scenario.items() if k != "name"}

        for comb in combinations:
            file_values = {}
            for k, v in global_params.items():
                file_values[k] = v
            for k, v in scenario_params.items():
                file_values[k] = v
            for k, v in comb.items():
                file_values[k] = v

            file_overrides = {}
            for k, v in file_values.items():
                if updated_common_data.get(k) != v:
                    file_overrides[k] = v

            # Deterministic filename generation
            name_part = f"{custom_name}_" if custom_name else ""
            suffix = "_".join(f"{k.lower()}{v}" for k, v in sorted(comb.items()))
            if prefix_override:
                filename = f"{prefix_override}_{name_part}{suffix}".strip("_") + ".yaml"
            else:
                filename = f"exp{global_idx}_{name_part}{suffix}".strip("_") + ".yaml"

            file_path = os.path.join(config_dir, filename)
            global_idx += 1

            action = "MODIFY" if os.path.exists(file_path) else "NEW"

            proposed_files[filename] = {
                "path": file_path,
                "action": action,
                "content": file_overrides
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
    if not force:
        confirm = input("\nDo you want to apply these changes? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Cancelled. No files were written.")
            sys.exit(0)

    # 8. Write the files
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
