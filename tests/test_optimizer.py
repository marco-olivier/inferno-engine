"""Optimizer tests"""
import pytest
from inferno.optimizer import ModelOptimizer
from inferno.optimizer.quantization import Quantizer
from inferno.optimizer.graph_optimizer import GraphOptimizer

class TestModelOptimizer:
    def test_init(self):
        optimizer = ModelOptimizer()
        assert optimizer._quantizer is not None
        assert optimizer._graph_optimizer is not None

class TestQuantizer:
    def test_init(self):
        quantizer = Quantizer()
        assert quantizer.config.precision == "int8"

class TestGraphOptimizer:
    def test_passes(self):
        optimizer = GraphOptimizer()
        assert len(optimizer.passes) == 3
