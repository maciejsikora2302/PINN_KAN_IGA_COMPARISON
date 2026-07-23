from abc import ABC, abstractmethod
import yaml
from typing import Any, Dict

class ExperimentInterface(ABC):
    """
    Abstract base interface for running physics simulation experiments using
    different numerical/machine learning methods.
    """

    def __init__(self, config_path: str | None = None):
        self.config: Any = None
        if config_path:
            self.load_config(config_path)

    @abstractmethod
    def load_config(self, config_path: str) -> None:
        """
        Loads configuration settings from a YAML file.
        
        Args:
            config_path (str): Path to the YAML configuration file.
        """
        pass

    @abstractmethod
    def train(self) -> None:
        """
        Performs the training loop or runs the physics simulation.
        """
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        """
        Saves the model weights/parameters/state to a file.
        
        Args:
            path (str): File path where the model should be saved.
        """
        pass

    @abstractmethod
    def save_outcomes(self, path: str) -> None:
        """
        Saves the outcome/results/metrics of the simulation to a file.
        
        Args:
            path (str): File path where the outcomes should be saved.
        """
        pass
