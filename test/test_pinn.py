import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinn import PINNExperiment
from visualizer import Visualizer

def main():
    print("=== Testing PINNExperiment ===")
    config_path = os.path.join(os.path.dirname(__file__), "pinn_test_config.yaml")
    model_save_path = os.path.join(os.path.dirname(__file__), "output", "pinn_test_model.pth")
    outcomes_save_path = os.path.join(os.path.dirname(__file__), "output", "pinn_test_outcomes.yaml")
    plot_save_path = os.path.join(os.path.dirname(__file__), "output", "pinn_test_loss_plot.png")
    
    # Initialize PINNExperiment
    experiment = PINNExperiment(config_path=config_path)
    
    # Train model
    experiment.train()
    
    # Save model and outcomes
    experiment.save_model(model_save_path)
    experiment.save_outcomes(outcomes_save_path)
    
    # Plot training loss using Visualizer
    print("Plotting loss history with Visualizer...")
    epochs = list(range(1, len(experiment.loss_history) + 1))
    visualizer = Visualizer()
    visualizer.plot(
        x=epochs,
        y=experiment.loss_history,
        title="PINN Training Loss",
        xlabel="Epoch",
        ylabel="Loss",
        save_path=plot_save_path
    )
    
    print("PINNExperiment test completed successfully.")

if __name__ == "__main__":
    main()
