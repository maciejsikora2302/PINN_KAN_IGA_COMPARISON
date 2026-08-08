import os
import yaml

class Config:
    """Base class containing configuration logic"""

    def to_dict(self):
        """Returns a dictionary containing all public configuration parameters."""
        return {key: value for key, value in self.__dict__.items() if not key.startswith('_')}
    
    def load_config(self, path: str):
        # 1. Load common configuration if it exists
        common_path = os.path.join(os.path.dirname(os.path.abspath(path)), "common.yaml")
        if not os.path.exists(common_path):
            common_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_config", "common.yaml")
        
        if os.path.exists(common_path):
            with open(common_path, 'r') as f:
                common_data = yaml.safe_load(f) or {}
            for key, value in common_data.items():
                setattr(self, key, value)
        
        # 2. Load the specific config file to override values
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        for key, value in data.items():
            setattr(self, key, value)

        # Apply fallback defaults for KAN configurations
        if hasattr(self, 'KAN_LAYERS') and self.KAN_LAYERS is None:
            self.KAN_LAYERS = self.LAYERS
        if hasattr(self, 'KAN_NEURONS_PER_LAYER') and self.KAN_NEURONS_PER_LAYER is None:
            if self.NEURONS_PER_LAYER is not None:
                self.KAN_NEURONS_PER_LAYER = max(5, self.NEURONS_PER_LAYER // 5)
        if hasattr(self, 'KAN_EPOCHS') and getattr(self, 'KAN_EPOCHS', None) is None:
            self.KAN_EPOCHS = getattr(self, 'EPOCHS', None)
        if hasattr(self, 'KAN_LEARNING_RATE') and getattr(self, 'KAN_LEARNING_RATE', None) is None:
            self.KAN_LEARNING_RATE = getattr(self, 'LEARNING_RATE', None)

        # 3. Handle default OUTPUT_DIR dynamically based on config name if not explicitly set
        if not getattr(self, 'OUTPUT_DIR', None):
            config_name = os.path.splitext(os.path.basename(path))[0]
            self.OUTPUT_DIR = os.path.join("output", config_name)

    def assign_from_args(self, args):
        """Override this method to assign configurations from parsed arguments"""
        raise NotImplementedError("Subclasses must implement assign_from_args")

    def validate_config(self):
        # validate which configs are still none after loading
        none_configs = [key for key, value in self.__dict__.items() if value is None]
        if none_configs:
            raise ValueError(f"Found missing configs!\nConfig: \n{'-'*50}\n{'\n'.join(none_configs)}\n{'-'*50}")
        print("Config validated successfully!")
        print(self)
        
    def __str__(self):
        config_parameters = [f"{key}={value}" for key, value in self.__dict__.items()]
        return "Config: \n" + '='*50 + '\n' + '\n'.join(config_parameters) + '\n' + '='*50
 
 
class SharedConfig(Config):
    """Shared configuration properties"""
    
    def __init__(self):
        self.LENGTH = None
        self.TOTAL_TIME = None
        self.N_POINTS_X = None
        self.N_POINTS_T = None
        self.LAYERS = None
        self.NEURONS_PER_LAYER = None
        self.KAN_LAYERS = None
        self.KAN_NEURONS_PER_LAYER = None
        self.EPOCHS = None
        self.KAN_EPOCHS = None
        self.LEARNING_RATE = None
        self.KAN_LEARNING_RATE = None
        self.RPINN = None
        self.EXAMPLE = None
        self.EPSILON = None
        self.ACTIVATION = None
        self.OUTPUT_DIR = None
        self.DEBUG = False
        self.H1_CALC_EVERY = 100
        self.IGA_DEGREE = 3
        self.IGA_ELEMENTS = 32
        self.IGA_METHOD = "standard"
        self.IGA_MESH_TYPE = "uniform"
        self.IGA_ADAPTIVE_GAMMA = 3.0
        self.IGA_TEST_DEGREE_ENRICHMENT = 1
        self.KAN_SPLINE_TYPE = "nurbs"
        self.KAN_GRID_SIZE = 5
        self.KAN_SPLINE_ORDER = 3
        self.SAMPLER_TYPE = "uniform"
        self.SAMPLER_GAMMA = 3.0

    def assign_from_args(self, args):
        # Assign arguments to CONFIG namespace
        self.LENGTH = args.length if hasattr(args, 'length') and args.length is not None else self.LENGTH
        self.TOTAL_TIME = args.total_time if hasattr(args, 'total_time') and args.total_time is not None else self.TOTAL_TIME
        self.N_POINTS_X = args.n_points_x if hasattr(args, 'n_points_x') and args.n_points_x is not None else self.N_POINTS_X
        self.N_POINTS_T = args.n_points_t if hasattr(args, 'n_points_t') and args.n_points_t is not None else self.N_POINTS_T
        self.LAYERS = args.layers if hasattr(args, 'layers') and args.layers is not None else self.LAYERS
        self.NEURONS_PER_LAYER = args.neurons_per_layer if hasattr(args, 'neurons_per_layer') and args.neurons_per_layer is not None else self.NEURONS_PER_LAYER
        self.KAN_LAYERS = args.kan_layers if hasattr(args, 'kan_layers') and args.kan_layers is not None else self.KAN_LAYERS
        self.KAN_NEURONS_PER_LAYER = args.kan_neurons_per_layer if hasattr(args, 'kan_neurons_per_layer') and args.kan_neurons_per_layer is not None else self.KAN_NEURONS_PER_LAYER
        self.EPOCHS = args.epochs if hasattr(args, 'epochs') and args.epochs is not None else self.EPOCHS
        self.KAN_EPOCHS = args.kan_epochs if hasattr(args, 'kan_epochs') and args.kan_epochs is not None else self.KAN_EPOCHS
        self.LEARNING_RATE = args.learning_rate if hasattr(args, 'learning_rate') and args.learning_rate is not None else self.LEARNING_RATE
        self.KAN_LEARNING_RATE = args.kan_learning_rate if hasattr(args, 'kan_learning_rate') and args.kan_learning_rate is not None else self.KAN_LEARNING_RATE
        self.RPINN = args.rpinn if hasattr(args, 'rpinn') and args.rpinn is not None else self.RPINN
        self.EXAMPLE = args.example if hasattr(args, 'example') and args.example is not None else self.EXAMPLE
        self.EPSILON = args.epsilon if hasattr(args, 'epsilon') and args.epsilon is not None else self.EPSILON
        self.ACTIVATION = args.activation if hasattr(args, 'activation') and args.activation is not None else self.ACTIVATION
        self.OUTPUT_DIR = args.output_dir if hasattr(args, 'output_dir') and args.output_dir is not None else self.OUTPUT_DIR
        self.DEBUG = args.debug if hasattr(args, 'debug') and args.debug is not None else self.DEBUG
        self.H1_CALC_EVERY = args.h1_calc_every if hasattr(args, 'h1_calc_every') and args.h1_calc_every is not None else self.H1_CALC_EVERY
        self.IGA_DEGREE = args.iga_degree if hasattr(args, 'iga_degree') and args.iga_degree is not None else self.IGA_DEGREE
        self.IGA_ELEMENTS = args.iga_elements if hasattr(args, 'iga_elements') and args.iga_elements is not None else self.IGA_ELEMENTS
        self.IGA_METHOD = args.iga_method if hasattr(args, 'iga_method') and args.iga_method is not None else self.IGA_METHOD
        self.IGA_MESH_TYPE = args.iga_mesh_type if hasattr(args, 'iga_mesh_type') and args.iga_mesh_type is not None else self.IGA_MESH_TYPE
        self.IGA_ADAPTIVE_GAMMA = args.iga_adaptive_gamma if hasattr(args, 'iga_adaptive_gamma') and args.iga_adaptive_gamma is not None else self.IGA_ADAPTIVE_GAMMA
        self.IGA_TEST_DEGREE_ENRICHMENT = args.iga_test_degree_enrichment if hasattr(args, 'iga_test_degree_enrichment') and args.iga_test_degree_enrichment is not None else self.IGA_TEST_DEGREE_ENRICHMENT
        self.KAN_SPLINE_TYPE = args.kan_spline_type if hasattr(args, 'kan_spline_type') and args.kan_spline_type is not None else self.KAN_SPLINE_TYPE
        self.KAN_GRID_SIZE = args.kan_grid_size if hasattr(args, 'kan_grid_size') and args.kan_grid_size is not None else self.KAN_GRID_SIZE
        self.KAN_SPLINE_ORDER = args.kan_spline_order if hasattr(args, 'kan_spline_order') and args.kan_spline_order is not None else self.KAN_SPLINE_ORDER
        self.SAMPLER_TYPE = args.sampler_type if hasattr(args, 'sampler_type') and args.sampler_type is not None else self.SAMPLER_TYPE
        self.SAMPLER_GAMMA = args.sampler_gamma if hasattr(args, 'sampler_gamma') and args.sampler_gamma is not None else self.SAMPLER_GAMMA


class PINNConfig(SharedConfig):
    """Specific configuration for PINN solver"""
    
    def __init__(self):
        super().__init__()


class KANConfig(SharedConfig):
    """Specific configuration for KAN solver"""
    
    def __init__(self):
        super().__init__()


class IGAConfig(SharedConfig):
    """Specific configuration for IGA solver"""
    
    def __init__(self):
        super().__init__()





