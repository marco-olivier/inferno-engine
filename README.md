# Inferno Engine

High-performance inference engine for AI/ML workloads with low-latency execution.

## Features

- **Model Loading**: Support for ONNX, PyTorch, TensorRT formats
- **Dynamic Batching**: Automatic request batching for throughput
- **Model Optimization**: Quantization, pruning, graph optimization
- **Multi-GPU**: Distributed inference across devices
- **Real-time Serving**: Low-latency model serving with gRPC/REST

## Architecture

```
┌─────────────────────────────────────────┐
│            Client Requests              │
├─────────────────────────────────────────┤
│           Serving Layer (gRPC)          │
├─────────────────────────────────────────┤
│         Dynamic Batching Engine         │
├─────────────────────────────────────────┤
│          Model Optimization             │
├─────────────────────────────────────────┤
│        Inference Runtime (GPU)          │
└─────────────────────────────────────────┘
```

## Quick Start

```python
from inferno import Engine, Model, Request

# Load model
model = Model.from_onnx("model.onnx")
engine = Engine(device="gpu")

# Optimize model
optimized = engine.optimize(model, quantize=True)

# Create inference session
session = engine.create_session(optimized)

# Run inference
request = Request(input_data=image)
response = session.predict(request)
print(response.output)
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- ONNX Runtime 1.15+

## Installation

```bash
pip install inferno-engine
```

## Documentation

- [Getting Started](docs/getting-started.md)
- [Model Formats](docs/models.md)
- [Serving Guide](docs/serving.md)
- [Examples](examples/)

## License

MIT License
