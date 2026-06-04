"""Session tests"""
import pytest
import numpy as np
from inferno.session import Session
from inferno.model import Model, ModelFormat, ModelMetadata

class MockModel:
    """Mock model for testing"""
    def __init__(self):
        self.name = "test_model"
        self.format = ModelFormat.ONNX
        self.metadata = ModelMetadata(
            name="test",
            format=ModelFormat.ONNX,
            input_shapes={"input": (1, 3, 224, 224)},
            output_shapes={"output": (1, 1000)}
        )
    
    def predict(self, inputs):
        return {"output": np.random.randn(1, 1000)}

class TestSession:
    def test_predict(self):
        model = MockModel()
        session = Session(model)
        
        from inferno.request import Request
        request = Request(inputs={"input": np.random.randn(1, 3, 224, 224)})
        response = session.predict(request)
        
        assert response.output.shape == (1, 1000)
        assert response.latency_ms > 0
    
    def test_benchmark(self):
        model = MockModel()
        session = Session(model)
        
        results = session.benchmark((1, 3, 224, 224), iterations=10)
        assert "mean_ms" in results
        assert "p95_ms" in results
