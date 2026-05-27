"""Core inference engine"""
from typing import Optional, Dict, Any
from .model import Model
from .session import Session
from .optimizer import ModelOptimizer

class EngineConfig:
    """Engine configuration"""
    def __init__(self, device: str = "gpu", num_streams: int = 4,
                 max_batch_size: int = 32, enable_profiling: bool = False):
        self.device = device
        self.num_streams = num_streams
        self.max_batch_size = max_batch_size
        self.enable_profiling = enable_profiling

class Engine:
    """Main inference engine"""
    
    def __init__(self, device: str = "gpu", config: Optional[EngineConfig] = None):
        self.device = device
        self.config = config or EngineConfig(device=device)
        self._optimizer = ModelOptimizer()
        self._sessions: Dict[str, Session] = {}
        self._initialized = False
    
    def initialize(self):
        """Initialize engine runtime"""
        if self._initialized:
            return
        # Initialize GPU context
        self._initialized = True
    
    def optimize(self, model: Model, quantize: bool = False,
                 optimize_graph: bool = True) -> Model:
        """Optimize model for inference"""
        return self._optimizer.optimize(model, quantize=quantize, 
                                        optimize_graph=optimize_graph)
    
    def create_session(self, model: Model) -> Session:
        """Create inference session"""
        session = Session(model, self.config)
        self._sessions[model.name] = session
        return session
    
    def load_model(self, path: str) -> Model:
        """Load model from file"""
        return Model.load(path)
    
    def benchmark(self, model: Model, input_shape: tuple, 
                  iterations: int = 100) -> Dict[str, float]:
        """Benchmark model performance"""
        session = self.create_session(model)
        return session.benchmark(input_shape, iterations)
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, *args):
        pass
