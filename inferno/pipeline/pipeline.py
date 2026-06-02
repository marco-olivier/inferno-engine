"""End-to-end inference pipeline"""
from typing import Dict, Any, List, Callable
import numpy as np
from ..session import Session
from ..request import Request

class PipelineStep:
    """Single pipeline step"""
    def __init__(self, name: str, func: Callable):
        self.name = name
        self.func = func
    
    def run(self, data: Any) -> Any:
        return self.func(data)

class Pipeline:
    """End-to-end inference pipeline"""
    
    def __init__(self, name: str = "default"):
        self.name = name
        self._steps: List[PipelineStep] = []
        self._session = None
    
    def add_preprocess(self, func: Callable) -> 'Pipeline':
        """Add preprocessing step"""
        self._steps.append(PipelineStep("preprocess", func))
        return self
    
    def set_session(self, session: Session) -> 'Pipeline':
        """Set inference session"""
        self._session = session
        return self
    
    def add_postprocess(self, func: Callable) -> 'Pipeline':
        """Add postprocessing step"""
        self._steps.append(PipelineStep("postprocess", func))
        return self
    
    def predict(self, input_data: Any) -> Any:
        """Run full pipeline"""
        data = input_data
        
        # Run preprocessing
        for step in self._steps:
            if step.name == "preprocess":
                data = step.run(data)
        
        # Run inference
        if self._session:
            request = Request(inputs=data)
            response = self._session.predict(request)
            data = response.outputs
        
        # Run postprocessing
        for step in self._steps:
            if step.name == "postprocess":
                data = step.run(data)
        
        return data
