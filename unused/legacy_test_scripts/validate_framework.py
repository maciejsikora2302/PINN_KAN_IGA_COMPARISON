import os
import sys

import yaml

# Add the parent directory to sys.path to resolve model and visualization imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import ExperimentInterface, SolverMetrics, SolverOutcome
from visualizer import Visualizer


class MockExperiment(ExperimentInterface):
    """
    A concrete mock implementation of ExperimentInterface for validation purposes.
    Generates dummy physics simulation outcomes (e.g., a simple harmonic motion wave).
    """

    def load_config(self, config_path: str) -> None:
        """Loads configuration from a mock YAML file or standardizes config parameters."""
        print(f"[Mock] Loading configuration from: {config_path}")
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        else:
            self.config = {
                "amplitude": 1.0,
                "frequency": 2.0,
                "num_points": 100,
                "epochs": 10
            }
            # Save the default mock config to file for demonstration
            with open(config_path, 'w') as f:
                yaml.safe_dump(self.config, f)
            print(f"[Mock] Created default YAML configuration file at: {config_path}")
            
        print(f"[Mock] Loaded Config: {self.config}")

    def train(self) -> SolverOutcome:
        """Simulates training or running the physics simulation."""
        print("[Mock] Starting training simulation...")
        epochs = self.config.get("epochs", 5)
        for epoch in range(1, epochs + 1):
            print(f"  [Mock] Epoch {epoch}/{epochs} - Loss: {1.0 / epoch:.4f}")
        
        # Hardcode some simulation outcomes
        import numpy as np
        num_points = self.config.get("num_points", 100)
        amp = self.config.get("amplitude", 1.0)
        freq = self.config.get("frequency", 2.0)
        
        # Generate a wave
        x_arr = np.linspace(0, 2 * np.pi, num_points)
        y_arr = amp * np.sin(freq * x_arr)
        self.x_data = x_arr.tolist()
        self.y_data = y_arr.tolist()
        print("[Mock] Training simulation complete.")

        metrics = SolverMetrics(
            final_loss=0.01,
            final_interior_loss=0.005,
            final_h1_error=0.02,
            final_l2_error=0.01,
            final_linf_error=0.03,
            trainable_parameters_or_dofs=10,
            elapsed_seconds=1.2,
            epochs_trained=epochs,
            epochs_total=epochs
        )
        self.outcome = SolverOutcome(
            x_grid=x_arr,
            t_grid=np.zeros_like(x_arr),
            z_pred=y_arr,
            loss_history=[1.0 / i for i in range(1, epochs + 1)],
            h1_error_history=[0.1 / i for i in range(1, epochs + 1)],
            h1_time_history=[0.1 * i for i in range(1, epochs + 1)],
            h1_epoch_history=list(range(1, epochs + 1)),
            h1_progress_history=[0.1 * i for i in range(1, epochs + 1)],
            metrics=metrics
        )
        return self.outcome

    def save_model(self, path: str) -> None:
        """Saves a dummy model state file."""
        print(f"[Mock] Saving model weights to: {path}")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as f:
            f.write("mock_model_state_vector: [0.1, -0.5, 1.2, 0.08]")
        print("[Mock] Model saved successfully.")

    def save_outcomes(self, path: str) -> None:
        """Saves the simulated wave outcomes to a YAML or text file."""
        print(f"[Mock] Saving simulation outcomes to: {path}")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        outcomes = {
            "x": self.x_data,
            "y": self.y_data,
            "metrics": {
                "max_amplitude": max(self.y_data),
                "min_amplitude": min(self.y_data)
            }
        }
        with open(path, 'w') as f:
            yaml.safe_dump(outcomes, f)
        print("[Mock] Outcomes saved successfully.")

def main():
    print("=== Starting Framework Validation ===")
    
    # Establish a 'test' directory relative to this script
    test_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(test_dir, exist_ok=True)
    
    config_file = os.path.join(test_dir, "mock_config.yaml")
    model_file = os.path.join(test_dir, "mock_model.txt")
    outcome_file = os.path.join(test_dir, "mock_outcomes.yaml")
    plot_file = os.path.join(test_dir, "mock_simulation_plot.png")
    
    # Initialize implementation
    experiment = MockExperiment(config_path=config_file)
    
    # Run simulation
    experiment.train()
    
    # Save model and outcomes
    experiment.save_model(model_file)
    experiment.save_outcomes(outcome_file)
    
    # Visualize outcomes
    print("[Validation] Plotting outcomes using the Visualizer class...")
    visualizer = Visualizer()
    visualizer.plot(
        x=experiment.x_data,
        y=experiment.y_data,
        title="Mock Physics Simulation Outcome (Simple Harmonic Motion)",
        xlabel="Time / Position",
        ylabel="Amplitude",
        save_path=plot_file
    )
    
    print("\nValidation runs successfully completed.")
    print(f"All generated mock files saved to: {test_dir}")

if __name__ == "__main__":
    main()
