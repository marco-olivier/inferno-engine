#!/usr/bin/env python3
"""Image classification example"""
import numpy as np
from inferno import Engine, Model, Request

def main():
    # Load model
    model = Model.load("resnet50.onnx")
    
    # Create engine
    with Engine(device="gpu") as engine:
        # Optimize
        optimized = engine.optimize(model, quantize=True)
        
        # Create session
        session = engine.create_session(optimized)
        
        # Prepare input
        image = np.random.randn(1, 3, 224, 224).astype(np.float32)
        request = Request(inputs={"input": image})
        
        # Run inference
        response = session.predict(request)
        
        # Get results
        logits = response.output
        predicted_class = np.argmax(logits)
        
        print(f"Predicted class: {predicted_class}")
        print(f"Latency: {response.latency_ms:.2f}ms")

if __name__ == "__main__":
    main()
