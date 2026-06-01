"""PyTorch backend"""
from typing import Dict, Any
import numpy as np

class TorchBackend:
    """PyTorch inference backend"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._models = {}
    
    def load_model(self, path: str) -> Any:
        """Load PyTorch model"""
        import torch
        model = torch.jit.load(path, map_location=self.device)
        model.eval()
        self._models[path] = model
        return model
    
    def predict(self, model, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run inference"""
        import torch
        
        # Convert to tensors
        tensors = {k: torch.from_numpy(v).to(self.device) 
                  for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**tensors)
        
        # Convert back to numpy
        if isinstance(outputs, torch.Tensor):
            return {"output": outputs.cpu().numpy()}
        return {k: v.cpu().numpy() for k, v in outputs.items()}
