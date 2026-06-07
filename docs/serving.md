# Serving Guide

## gRPC Server

```python
from inferno import Engine, Model
from inferno.serving import InferenceServer

engine = Engine(device="gpu")
model = Model.load("model.onnx")

server = InferenceServer(engine)
server.register_model("my_model", model)
server.start()
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| host | 0.0.0.0 | Server host |
| port | 8000 | Server port |
| max_workers | 4 | Worker threads |
| enable_metrics | true | Prometheus metrics |

## Health Check

```bash
curl http://localhost:8000/health
```

## Metrics

```bash
curl http://localhost:8000/metrics
```
