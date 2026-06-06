#!/usr/bin/env python3
"""Server deployment example"""
from inferno import Engine, Model
from inferno.serving import InferenceServer

def main():
    # Load model
    model = Model.load("model.onnx")
    
    # Create engine
    engine = Engine(device="gpu")
    engine.initialize()
    
    # Create server
    server = InferenceServer(engine, host="0.0.0.0", port=8000)
    server.register_model("classifier", model)
    
    # Start serving
    print("Starting inference server...")
    server.start()

if __name__ == "__main__":
    main()
