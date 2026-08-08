from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional
import torch

class CollocationSampler(ABC):
    """
    Abstract Base Class for collocation point sampling and discrete Gram matrix construction.
    """

    @abstractmethod
    def sample_points(self, n_points_x: int, n_points_t: int, length: float = 1.0, total_time: float = 1.0, device: Any = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates (x_grid, t_grid) collocation coordinates."""
        pass

    @abstractmethod
    def build_gram_matrix(self, n_points_x: int, n_points_t: int, device: Any = "cpu") -> Optional[Any]:
        """Constructs discrete Gram matrix factor G_LU for robust loss calculation."""
        pass
