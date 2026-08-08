import unittest
import os
import torch
import numpy as np

from src.samplers import BoundaryLayerSampler, UniformGridSampler

class TestPhase3Samplers(unittest.TestCase):

    def test_uniform_vs_boundary_layer_sampling(self):
        """Verify boundary layer sampler places points inside [0, 1] with higher density near t=1.0."""
        sampler = BoundaryLayerSampler(stretch_gamma=3.0)
        x_grid, t_grid = sampler.sample_points(n_points_x=10, n_points_t=10, length=1.0, total_time=1.0)

        # Check bounds
        self.assertGreaterEqual(x_grid.min().item(), 0.0)
        self.assertLessEqual(x_grid.max().item(), 1.0)
        self.assertGreaterEqual(t_grid.min().item(), 0.0)
        self.assertLessEqual(t_grid.max().item(), 1.0)

        # Verify density near t=1.0 is higher than uniform spacing
        t_coords = torch.unique(t_grid)
        diff_first = float((t_coords[1] - t_coords[0]).detach())
        diff_last = float((t_coords[-1] - t_coords[-2]).detach())
        self.assertLess(diff_last, diff_first, "Spacing near t=1.0 should be tighter than near t=0.0")

    def test_non_uniform_gram_matrix_solvability(self):
        """Verify non-uniform Gram matrix G_LU solver produces finite non-zero outputs."""
        sampler = BoundaryLayerSampler(stretch_gamma=2.5)
        n_x, n_t = 10, 10
        G_LU = sampler.build_gram_matrix(n_points_x=n_x, n_points_t=n_t)

        self.assertIsNotNone(G_LU)
        rhs = torch.randn(n_x * n_t, 1)
        sol = torch.linalg.lu_solve(*G_LU, rhs)

        self.assertFalse(torch.isnan(sol).any().item(), "Gram matrix linear solve produced NaNs.")
        self.assertFalse(torch.isinf(sol).any().item(), "Gram matrix linear solve produced Infs.")
        self.assertGreater(torch.abs(sol).sum().item(), 0.0)

    def test_extended_run_execution(self):
        """Test full train.py pipeline using extended_test_config.yaml."""
        import subprocess

        cmd = [
            os.path.join("venv", "Scripts", "python.exe"),
            "train.py",
            "training_config/extended_test_config.yaml"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"train.py failed with stderr: {result.stderr}")

        output_dir = os.path.join("output", "extended_test_config")
        self.assertTrue(os.path.exists(os.path.join(output_dir, "pinn.npz")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "kan.npz")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "iga.npz")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "metadata.yaml")))

if __name__ == "__main__":
    unittest.main()
