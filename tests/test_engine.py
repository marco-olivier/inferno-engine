"""Engine tests"""
import pytest
from inferno.engine import Engine, EngineConfig

class TestEngine:
    def test_init(self):
        engine = Engine(device="cpu")
        assert engine.device == "cpu"
    
    def test_config(self):
        config = EngineConfig(device="gpu", num_streams=8)
        engine = Engine(config=config)
        assert engine.config.num_streams == 8
    
    def test_initialize(self):
        engine = Engine()
        engine.initialize()
        assert engine._initialized == True
