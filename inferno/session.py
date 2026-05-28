"""Inference session management"""
from typing import Dict, Optional, List
import numpy as np
import time
from .model import Model
from .request import Request, Response, ResponseBuilder
from .batching import DynamicBatcher

class SessionConfig:
    """Session configuration"""
    def __init__(self, max_batch_size: int = 32, timeout_ms: float = 100.0,
                 num_streams: int = 4):
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.num_streams = num_streams

class Session:
    """Inference session for model execution"""
    
    def __init__(self, model: Model, config=None):
        self.model = model
        self.config = SessionConfig()
        self._batcher = DynamicBatcher(
            max_batch_size=self.config.max_batch_size,
            timeout_ms=self.config.timeout_ms
        )
        self._request_count = 0
    
    def predict(self, request: Request) -> Response:
        """Run inference for single request"""
        builder = ResponseBuilder(request)
        
        # Run model
        outputs = self.model.predict(request.inputs)
        
        return builder.build(outputs, self.model.name)
    
    def predict_batch(self, requests: List[Request]) -> List[Response]:
        """Run inference for batch of requests"""
        # Batch requests
        batched = self._batcher.batch(requests)
        
        responses = []
        for batch in batched:
            # Merge inputs
            merged_inputs = self._merge_inputs(batch)
            
            # Run inference
            outputs = self.model.predict(merged_inputs)
            
            # Split outputs
            for i, req in enumerate(batch):
                builder = ResponseBuilder(req)
                single_output = self._extract_output(outputs, i)
                responses.append(builder.build(single_output, self.model.name))
        
        return responses
    
    def _merge_inputs(self, requests: List[Request]) -> Dict[str, np.ndarray]:
        """Merge request inputs into batch"""
        merged = {}
        for key in requests[0].inputs:
            merged[key] = np.stack([r.inputs[key] for r in requests])
        return merged
    
    def _extract_output(self, outputs: Dict[str, np.ndarray], 
                        index: int) -> Dict[str, np.ndarray]:
        """Extract single output from batch"""
        return {k: v[index] for k, v in outputs.items()}
    
    def benchmark(self, input_shape: tuple, iterations: int = 100) -> Dict[str, float]:
        """Benchmark session performance"""
        # Create dummy input
        dummy_input = {k: np.random.randn(*v).astype(np.float32) 
                      for k, v in self.model.metadata.input_shapes.items()}
        
        latencies = []
        for _ in range(iterations):
            request = Request(inputs=dummy_input)
            start = time.perf_counter()
            self.predict(request)
            latencies.append((time.perf_counter() - start) * 1000)
        
        return {
            'mean_ms': np.mean(latencies),
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'throughput_rps': 1000 / np.mean(latencies),
        }
