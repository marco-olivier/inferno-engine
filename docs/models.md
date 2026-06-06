# Model Formats

## Supported Formats

### ONNX
```python
from inferno import Model
model = Model.load("model.onnx")
```

### PyTorch
```python
model = Model.load("model.pt")
```

### TensorRT
```python
model = Model.load("model.trt")
```

## Model Optimization

### Quantization
```python
from inferno.optimizer import Quantizer
quantizer = Quantizer(precision="int8")
optimized = quantizer.quantize(model)
```

### Graph Optimization
```python
from inferno.optimizer import GraphOptimizer
optimizer = GraphOptimizer()
optimized = optimizer.optimize(model)
```

## Custom Models

Implement `Model` interface for custom formats.
