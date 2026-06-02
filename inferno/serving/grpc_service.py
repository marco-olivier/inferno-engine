"""gRPC service for inference"""
from typing import Dict, Any
import numpy as np

class GrpcService:
    """gRPC inference service"""
    
    def __init__(self, server):
        self.server = server
        self._handlers = {}
    
    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prediction request"""
        model_name = request.get('model')
        inputs = request.get('inputs', {})
        
        if model_name not in self.server._models:
            raise ValueError(f"Model not found: {model_name}")
        
        model = self.server._models[model_name]
        outputs = model.predict(inputs)
        
        return {
            'outputs': {k: v.tolist() for k, v in outputs.items()},
            'model': model_name
        }
    
    def model_metadata(self, model_name: str) -> Dict[str, Any]:
        """Get model metadata"""
        if model_name not in self.server._models:
            raise ValueError(f"Model not found: {model_name}")
        
        model = self.server._models[model_name]
        return {
            'name': model.name,
            'inputs': model.metadata.input_shapes,
            'outputs': model.metadata.output_shapes,
        }
