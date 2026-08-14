import unittest
import os
import subprocess
import glob
import yaml

class TestPhase5ConfigManagement(unittest.TestCase):

    def test_manage_configs_generation(self):
        """Run manage_configs.py on sweeps/sweep_poisson.yaml and verify clean YAML configs are generated."""
        cmd = [
            os.path.join("venv", "Scripts", "python.exe"),
            "manage_configs.py",
            "sweeps/sweep_poisson.yaml",
            "--force"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"manage_configs.py failed with stderr: {result.stderr}")

        # Check generated configs in training_config
        config_files = glob.glob(os.path.join("training_config", "exp*_poisson_*.yaml"))
        self.assertGreater(len(config_files), 0, "No experiment configs generated for Poisson sweep.")

        for cfg_file in config_files:
            with open(cfg_file, "r") as f:
                data = yaml.safe_load(f)
            self.assertIsInstance(data, dict, f"Failed to parse generated config: {cfg_file}")

    def test_run_all_ordering(self):
        """Assert run_all.py naturally sorts experiment config files sequentially (exp1, exp2, ...)."""
        from run_all import natural_sort
        sample_files = [
            "exp10_test.yaml", "exp1_test.yaml", "exp2_test.yaml", "exp20_test.yaml"
        ]
        sorted_files = natural_sort(sample_files)
        expected = ["exp1_test.yaml", "exp2_test.yaml", "exp10_test.yaml", "exp20_test.yaml"]
        self.assertEqual(sorted_files, expected)

if __name__ == "__main__":
    unittest.main()
