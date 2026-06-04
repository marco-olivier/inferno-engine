"""Test fixtures"""
import pytest
import numpy as np

@pytest.fixture
def dummy_model_path(tmp_path):
    """Create dummy model file"""
    # Create minimal ONNX model for testing
    path = tmp_path / "test.onnx"
    path.touch()
    return str(path)

@pytest.fixture
def sample_input():
    """Sample input data"""
    return {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)}
