"""Request/Response objects for inference"""
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np
import time

@dataclass
class Request:
    """Inference request"""
    inputs: Dict[str, np.ndarray]
    request_id: str = ""
    priority: int = 0
    timeout: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.request_id:
            import uuid
            self.request_id = str(uuid.uuid4())[:8]

@dataclass
class Response:
    """Inference response"""
    request_id: str
    outputs: Dict[str, np.ndarray]
    latency_ms: float = 0.0
    model_name: str = ""
    batch_size: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def output(self) -> np.ndarray:
        """Get first output"""
        return next(iter(self.outputs.values()))

class ResponseBuilder:
    """Build response from model output"""
    
    def __init__(self, request: Request):
        self.request = request
        self._start_time = time.perf_counter()
    
    def build(self, outputs: Dict[str, np.ndarray], 
              model_name: str = "") -> Response:
        """Build response"""
        latency = (time.perf_counter() - self._start_time) * 1000
        return Response(
            request_id=self.request.request_id,
            outputs=outputs,
            latency_ms=latency,
            model_name=model_name
        )
