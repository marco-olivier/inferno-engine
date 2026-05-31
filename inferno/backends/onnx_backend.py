"""ONNX Runtime backend"""
from typing import Dict, Any
import numpy as np

class OnnxBackend:
    """ONNX Runtime inference backend"""
    
    def __init__(self, providers: list = None):
        self.providers = providers or ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self._sessions = {}
    
    def load_model(self, path: str) -> Any:
        """Load ONNX model"""
        import onnxruntime as ort
        session = ort.InferenceSession(path, providers=self.providers)
        self._sessions[path] = session
        return session
    
    def predict(self, session, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run inference"""
        output_names = [o.name for o in session.get_outputs()]
        results = session.run(output_names, inputs)
        return dict(zip(output_names, results))
    
    def get_input_info(self, session) -> Dict[str, tuple]:
        """Get model input information"""
        return {inp.name: inp.shape for inp in session.get_inputs()}
    
    def get_output_info(self, session) -> Dict[str, tuple]:
        """Get model output information"""
        return {out.name: out.shape for out in session.get_outputs()}
