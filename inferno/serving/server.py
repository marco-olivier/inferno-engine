"""Inference server"""
from typing import Optional, Dict
from ..engine import Engine
from ..model import Model

class ServerConfig:
    """Server configuration"""
    def __init__(self, host: str = "0.0.0.0", port: int = 8000,
                 max_workers: int = 4, enable_metrics: bool = True):
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.enable_metrics = enable_metrics

class InferenceServer:
    """gRPC/REST inference server"""
    
    def __init__(self, engine: Engine, config: Optional[ServerConfig] = None):
        self.engine = engine
        self.config = config or ServerConfig()
        self._models: Dict[str, Model] = {}
        self._running = False
    
    def register_model(self, name: str, model: Model):
        """Register model for serving"""
        self._models[name] = model
    
    def start(self):
        """Start inference server"""
        self._running = True
        print(f"Server starting on {self.config.host}:{self.config.port}")
    
    def stop(self):
        """Stop server"""
        self._running = False
    
    def health_check(self) -> bool:
        """Check server health"""
        return self._running
