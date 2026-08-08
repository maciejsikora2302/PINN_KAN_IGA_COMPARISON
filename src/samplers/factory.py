from .uniform_grid import UniformGridSampler
from .boundary_layer import BoundaryLayerSampler

def get_sampler(sampler_type: str = "uniform", gamma: float = 3.0):
    """
    Factory function for obtaining collocation samplers.
    Supported types: 'uniform', 'boundary_layer'
    """
    if str(sampler_type).lower() == "boundary_layer":
        return BoundaryLayerSampler(stretch_gamma=gamma)
    return UniformGridSampler()
