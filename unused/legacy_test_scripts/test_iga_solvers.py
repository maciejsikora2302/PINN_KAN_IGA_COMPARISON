import unittest
import os
import tempfile
import numpy as np
import yaml

from src.problems import get_problem
from src.solvers.iga import (
    BaseIGASolver,
    StandardIGASolver,
    SUPGIGASolver,
    ResidualMinimizationIGASolver,
    IGAExperiment,
)

class TestIGASolvers(unittest.TestCase):

    def test_standard_iga_poisson(self):
        """Test StandardIGASolver on Poisson Example 1 with p=2, M=8."""
        problem = get_problem(1, epsilon=0.01)
        solver = StandardIGASolver()
        sol = solver.solve(
            problem=problem,
            p=2,
            M=8,
            mesh_type="uniform",
            n_points_x=20,
            n_points_t=20
        )
        self.assertFalse(np.isnan(sol.sol_coeffs).any(), "Solver output contains NaNs.")
        self.assertLess(sol.metrics.l2_error, 0.1, f"L2 error too high: {sol.metrics.l2_error}")
        self.assertLess(sol.metrics.h1_error, 2.0, f"H1 error too high: {sol.metrics.h1_error}")

    def test_supg_eriksson_johnson(self):
        """Test SUPGIGASolver on Eriksson-Johnson Example 3 with epsilon=0.01, p=2, M=8."""
        problem = get_problem(3, epsilon=0.01)
        solver = SUPGIGASolver()
        sol = solver.solve(
            problem=problem,
            p=2,
            M=8,
            mesh_type="uniform",
            n_points_x=20,
            n_points_t=20
        )
        self.assertFalse(np.isnan(sol.sol_coeffs).any(), "SUPG solver output contains NaNs.")
        self.assertLess(sol.metrics.linf_error, 5.0, f"Linf error unbounded: {sol.metrics.linf_error}")

    def test_igrm_eriksson_johnson(self):
        """Test ResidualMinimizationIGASolver on Example 3 with Delta p=1."""
        problem = get_problem(3, epsilon=0.01)
        solver = ResidualMinimizationIGASolver()
        sol = solver.solve(
            problem=problem,
            p=2,
            M=8,
            mesh_type="uniform",
            test_degree_enrichment=1,
            n_points_x=20,
            n_points_t=20
        )
        self.assertFalse(np.isnan(sol.sol_coeffs).any(), "iGRM solver output contains NaNs.")
        self.assertTrue(hasattr(sol.metrics, "h1_error"))

    def test_adaptive_knot_mesh(self):
        """Test graded knot vector generation clustered towards 1.0."""
        p = 2
        M = 8
        gamma = 3.0
        knots = BaseIGASolver.open_graded_knots(p, M, gamma=gamma)

        # Assert bounds [0, 1]
        self.assertEqual(knots[0], 0.0)
        self.assertEqual(knots[-1], 1.0)
        self.assertTrue(np.all(np.diff(knots) >= 0.0), "Knot vector is not non-decreasing.")

        # Check clustering towards 1.0 (internal knot spacing near 1.0 should be smaller than near 0.0)
        internal_knots = knots[p+1 : -(p+1)]
        first_step = internal_knots[1] - internal_knots[0]
        last_step = internal_knots[-1] - internal_knots[-2]
        self.assertLess(last_step, first_step, "Knots are not clustered towards 1.0.")

    def test_iga_config_dispatch(self):
        """Test loading config with IGA_METHOD: supg and IGA_MESH_TYPE: adaptive via IGAExperiment."""
        config_data = {
            "RPINN": 0,
            "LENGTH": 1.0,
            "TOTAL_TIME": 1.0,
            "N_POINTS_X": 15,
            "N_POINTS_T": 15,
            "LAYERS": 1,
            "NEURONS_PER_LAYER": 10,
            "EPOCHS": 10,
            "LEARNING_RATE": 0.01,
            "EXAMPLE": 3,
            "EPSILON": 0.01,
            "ACTIVATION": "tanh",
            "IGA_DEGREE": 2,
            "IGA_ELEMENTS": 8,
            "IGA_METHOD": "supg",
            "IGA_MESH_TYPE": "adaptive",
            "IGA_ADAPTIVE_GAMMA": 3.0
        }

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_config_path = f.name

        try:
            experiment = IGAExperiment(temp_config_path)
            experiment.train()

            self.assertIsNotNone(experiment.sol_coeffs)
            self.assertEqual(experiment.config.IGA_METHOD, "supg")
            self.assertEqual(experiment.config.IGA_MESH_TYPE, "adaptive")
            self.assertIsNotNone(experiment.final_h1_error)
        finally:
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

if __name__ == "__main__":
    unittest.main()
