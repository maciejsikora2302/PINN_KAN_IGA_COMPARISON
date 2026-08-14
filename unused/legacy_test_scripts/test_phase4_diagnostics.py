import unittest
import os
import tempfile
import numpy as np
import yaml

from visualizer import Visualizer
from src.problems import get_problem
from src.solvers.iga import BaseIGASolver

class TestPhase4Diagnostics(unittest.TestCase):

    def test_error_norms_computation(self):
        """Verify error norms computation produces positive floats for L2, H1, and Linf."""
        problem = get_problem(1, epsilon=0.01)
        x_grid = np.linspace(0.0, 1.0, 20).reshape(-1, 1)
        t_grid = np.linspace(0.0, 1.0, 20).reshape(-1, 1)

        # Mock prediction and derivatives
        z_pred = np.zeros(20)
        dzdx = np.zeros(20)
        dzdt = np.zeros(20)

        solver = BaseIGASolver
        # Instantiate dummy subclass to call compute_error_norms
        from src.solvers.iga import StandardIGASolver
        iga_solver = StandardIGASolver()
        metrics = iga_solver.compute_error_norms(x_grid, t_grid, problem, z_pred, dzdx, dzdt)

        self.assertTrue(hasattr(metrics, "l2_error"))
        self.assertTrue(hasattr(metrics, "h1_error"))
        self.assertTrue(hasattr(metrics, "linf_error"))

        self.assertGreater(metrics.l2_error, 0.0)
        self.assertGreater(metrics.h1_error, 0.0)
        self.assertGreater(metrics.linf_error, 0.0)

    def test_boundary_layer_slice_plot(self):
        """Call Visualizer().plot_boundary_layer_slice(...) with dummy predictions and check image creation."""
        viz = Visualizer()
        dummy_x = np.tile(np.linspace(0.0, 1.0, 10), 10)
        dummy_t = np.repeat(np.linspace(0.0, 1.0, 10), 10)
        dummy_z = np.sin(np.pi * dummy_x) * np.sin(np.pi * dummy_t)

        predictions = {
            "PINN": {"x": dummy_x, "t": dummy_t, "z": dummy_z},
            "KAN": {"x": dummy_x, "t": dummy_t, "z": dummy_z},
            "IGA": {"x": dummy_x, "t": dummy_t, "z": dummy_z}
        }

        save_path = os.path.join(tempfile.gettempdir(), "test_boundary_slice_plot.png")
        viz.plot_boundary_layer_slice(
            predictions=predictions,
            example=3,
            epsilon=0.01,
            x_cut=0.5,
            save_path=save_path
        )

        self.assertTrue(os.path.exists(save_path))
        os.remove(save_path)

    def test_summary_table_generation(self):
        """Run generate_summary_tables functions on temporary metadata and verify headers."""
        temp_dir = tempfile.mkdtemp()
        sub_dir = os.path.join(temp_dir, "test_exp_1")
        os.makedirs(sub_dir, exist_ok=True)

        meta_data = {
            "config": {"EXAMPLE": 3, "EPSILON": 0.01, "SAMPLER_TYPE": "boundary_layer"},
            "results": {
                "PINN": {"final_l2_error": 0.0012, "final_h1_error": 0.054, "final_linf_error": 0.015, "elapsed_seconds": 1.2},
                "KAN": {"final_l2_error": 0.0008, "final_h1_error": 0.042, "final_linf_error": 0.009, "elapsed_seconds": 2.1},
                "IGA": {"final_l2_error": 0.0001, "final_h1_error": 0.012, "final_linf_error": 0.002, "elapsed_seconds": 0.3}
            }
        }

        with open(os.path.join(sub_dir, "metadata.yaml"), "w") as f:
            yaml.dump(meta_data, f)

        records = parse_metadata_files(temp_dir)
        self.assertEqual(len(records), 3)

        md_table = generate_markdown_table(records)
        self.assertIn("Config / Run", md_table)
        self.assertIn("Linf Peak Error", md_table)

        tex_table = generate_latex_table(records)
        self.assertIn(r"\begin{table}", tex_table)
        self.assertIn(r"L^2", tex_table)

    def test_poisson_exp_residual(self):
        """Verify that u_xx + u_yy + f = 0 for Poisson Exp exact solution."""
        problem = get_problem(2)
        x = np.linspace(0.0, 1.0, 50)
        y = np.linspace(0.0, 1.0, 50)
        X, Y = np.meshgrid(x, y)

        u_xx = problem.exact_dx2(X, Y)
        u_yy = problem.exact_dy2(X, Y)
        f = problem.rhs(X, Y)

        residual = u_xx + u_yy + f
        np.testing.assert_allclose(residual, 0.0, atol=1e-12)

if __name__ == "__main__":
    unittest.main()
