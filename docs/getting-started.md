# Getting Started

## Installation

```bash
pip install inferno-engine
```

## Basic Usage

```python
from inferno import Engine, Model, Request

# Load model
model = Model.load("model.onnx")

# Create engine
engine = Engine(device="gpu")

# Optimize and create session
optimized = engine.optimize(model, quantize=True)
session = engine.create_session(optimized)

# Run inference
request = Request(inputs={"input": data})
response = session.predict(request)
```

## Model Formats

- ONNX (.onnx)
- PyTorch (.pt, .pth)
- TensorRT (.trt)
