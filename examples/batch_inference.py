#!/usr/bin/env python3
"""Batch inference example"""
import numpy as np
from inferno import Engine, Model, Request

def main():
    model = Model.load("model.onnx")
    
    with Engine(device="gpu") as engine:
        session = engine.create_session(model)
        
        # Create batch of requests
        requests = [
            Request(inputs={"input": np.random.randn(1, 3, 224, 224).astype(np.float32)})
            for _ in range(16)
        ]
        
        # Batch inference
        responses = session.predict_batch(requests)
        
        print(f"Processed {len(responses)} requests")
        print(f"Avg latency: {np.mean([r.latency_ms for r in responses]):.2f}ms")

if __name__ == "__main__":
    main()
