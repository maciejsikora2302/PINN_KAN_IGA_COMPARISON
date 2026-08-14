import os
import tempfile
import unittest

import numpy as np
import torch
import yaml

from src.solvers.kan import KANExperiment, NURBSLinear


class TestKANNURBS(unittest.TestCase):

    def test_nurbs_positive_weights(self):
        """Assert all projective weights w = softplus(raw_weights) + 1e-4 are strictly > 0."""
        layer = NURBSLinear(in_features=2, out_features=4)
        w = layer.positive_weights
        self.assertTrue(torch.all(w > 0.0).item(), "Projective weights contain non-positive values.")
        self.assertGreaterEqual(w.min().item(), 1e-4)

    def test_nurbs_partition_of_unity(self):
        """Assert that if w_i = 1, rational B-spline basis sums to 1.0 everywhere in [-1, 1]."""
        layer = NURBSLinear(in_features=2, out_features=3, grid_range=[-1.0, 1.0])
        # Set raw_weights so softplus gives 1.0
        with torch.no_grad():
            layer.raw_weights.fill_(0.5413)

        x_eval = torch.linspace(-0.9, 0.9, 50).unsqueeze(1).repeat(1, 2)
        bases = layer.b_splines(x_eval)  # (N, in_features, num_bases)
        w = layer.positive_weights  # (out_features, in_features, num_bases)

        bases_exp = bases.unsqueeze(1)
        weighted_bases = bases_exp * w.unsqueeze(0)
        den = torch.sum(weighted_bases, dim=-1, keepdim=True) + 1e-8
        rational_bases = weighted_bases / den

        sum_rational = rational_bases.sum(dim=-1)  # (N, out_features, in_features)
        # Should sum to 1.0 everywhere on active domain
        max_diff = torch.max(torch.abs(sum_rational - 1.0)).item()
        self.assertLess(max_diff, 1e-4, f"Partition of unity violated: max diff {max_diff}")

    def test_nurbs_autograd_gradients(self):
        """Assert 1st and 2nd derivatives via torch.autograd.grad are finite and non-NaN."""
        layer = NURBSLinear(in_features=2, out_features=1)
        x = torch.linspace(-0.8, 0.8, 20, requires_grad=True).unsqueeze(1).repeat(1, 2)
        out = layer(x)

        dfdx = torch.autograd.grad(out.sum(), x, create_graph=True)[0]
        self.assertFalse(torch.isnan(dfdx).any().item(), "1st derivative contains NaNs.")
        self.assertFalse(torch.isinf(dfdx).any().item(), "1st derivative contains Infs.")

        d2fdx2 = torch.autograd.grad(dfdx.sum(), x, create_graph=False)[0]
        self.assertFalse(torch.isnan(d2fdx2).any().item(), "2nd derivative contains NaNs.")
        self.assertFalse(torch.isinf(d2fdx2).any().item(), "2nd derivative contains Infs.")

    def test_kan_experiment_quick_run(self):
        """Run 10 training epochs of KANExperiment with KAN_SPLINE_TYPE: nurbs."""
        config_data = {
            "RPINN": 0,
            "LENGTH": 1.0,
            "TOTAL_TIME": 1.0,
            "N_POINTS_X": 10,
            "N_POINTS_T": 10,
            "KAN_LAYERS": 1,
            "KAN_NEURONS_PER_LAYER": 8,
            "EPOCHS": 10,
            "KAN_EPOCHS": 10,
            "LEARNING_RATE": 0.01,
            "EXAMPLE": 1,
            "EPSILON": 0.01,
            "ACTIVATION": "tanh",
            "KAN_SPLINE_TYPE": "nurbs",
            "KAN_GRID_SIZE": 5,
            "KAN_SPLINE_ORDER": 3
        }

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_config_path = f.name

        try:
            experiment = KANExperiment(temp_config_path)
            experiment.train()

            self.assertIsNotNone(experiment.final_loss)
            self.assertFalse(np.isnan(experiment.final_loss))

            outcomes_path = os.path.join(tempfile.gettempdir(), "test_kan_nurbs_outcomes.npz")
            experiment.save_outcomes(outcomes_path)
            self.assertTrue(os.path.exists(outcomes_path))

            with np.load(outcomes_path) as data:
                self.assertIn("kan_layer_0_nurbs_weights", data.files)

            if os.path.exists(outcomes_path):
                os.remove(outcomes_path)
        finally:
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

if __name__ == "__main__":
    unittest.main()
