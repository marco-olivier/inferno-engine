"""Batching tests"""
import pytest
import numpy as np
from inferno.batching import DynamicBatcher
from inferno.request import Request

class TestDynamicBatcher:
    def test_batch(self):
        batcher = DynamicBatcher(max_batch_size=4)
        requests = [
            Request(inputs={"x": np.array([i])})
            for i in range(10)
        ]
        batches = batcher.batch(requests)
        assert len(batches) == 3  # 4+4+2
        assert len(batches[0]) == 4
    
    def test_empty_batch(self):
        batcher = DynamicBatcher()
        batches = batcher.batch([])
        assert len(batches) == 0
