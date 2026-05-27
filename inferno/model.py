"""Model loading and management"""
from typing import Optional, Dict, Any, List
from pathlib import Path
from enum import Enum
import numpy as np

class ModelFormat(Enum):
    ONNX = "onnx"
    PYTORCH = "pytorch"
    TENSORRT = "tensorrt"
    UNKNOWN = "unknown"

class ModelMetadata:
    """Model metadata"""
    def __init__(self, name: str, format: ModelFormat, 
                 input_shapes: Dict[str, tuple],
                 output_shapes: Dict[str, tuple]):
        self.name = name
        self.format = format
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes

class Model:
    """Neural network model"""
    
    def __init__(self, name: str, model_data: Any, 
                 format: ModelFormat, metadata: ModelMetadata):
        self.name = name
        self._data = model_data
        self.format = format
        self.metadata = metadata
    
    @classmethod
    def load(cls, path: str) -> 'Model':
        """Load model from file"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        format = cls._detect_format(path)
        
        if format == ModelFormat.ONNX:
            return cls._load_onnx(path)
        elif format == ModelFormat.PYTORCH:
            return cls._load_pytorch(path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def _detect_format(path: Path) -> ModelFormat:
        """Detect model format from extension"""
        ext = path.suffix.lower()
        mapping = {
            '.onnx': ModelFormat.ONNX,
            '.pt': ModelFormat.PYTORCH,
            '.pth': ModelFormat.PYTORCH,
            '.trt': ModelFormat.TENSORRT,
        }
        return mapping.get(ext, ModelFormat.UNKNOWN)
    
    @classmethod
    def _load_onnx(cls, path: Path) -> 'Model':
        """Load ONNX model"""
        import onnxruntime as ort
        session = ort.InferenceSession(str(path))
        
        inputs = {inp.name: inp.shape for inp in session.get_inputs()}
        outputs = {out.name: out.shape for out in session.get_outputs()}
        
        metadata = ModelMetadata(
            name=path.stem,
            format=ModelFormat.ONNX,
            input_shapes=inputs,
            output_shapes=outputs
        )
        
        return cls(path.stem, session, ModelFormat.ONNX, metadata)
    
    @classmethod
    def _load_pytorch(cls, path: Path) -> 'Model':
        """Load PyTorch model"""
        import torch
        model = torch.jit.load(str(path))
        
        metadata = ModelMetadata(
            name=path.stem,
            format=ModelFormat.PYTORCH,
            input_shapes={},
            output_shapes={}
        )
        
        return cls(path.stem, model, ModelFormat.PYTORCH, metadata)
    
    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run inference"""
        if self.format == ModelFormat.ONNX:
            return self._data.run(None, inputs)
        else:
            raise NotImplementedError(f"Predict not implemented for {self.format}")
