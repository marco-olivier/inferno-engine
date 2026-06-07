# Performance Guide

## Batching

Use dynamic batching for higher throughput:
```python
session.predict_batch(requests)  # More efficient than individual calls
```

## GPU Optimization

- Use FP16 precision when possible
- Enable TensorRT for supported models
- Use multiple streams for concurrent inference

## Memory Management

- Reuse buffers across requests
- Enable memory pooling
- Monitor GPU memory usage

## Benchmarks

| Model | Batch Size | Latency (ms) | Throughput (RPS) |
|-------|-----------|---------------|------------------|
| ResNet-50 | 1 | 2.1 | 476 |
| ResNet-50 | 32 | 12.5 | 2560 |
| BERT-base | 1 | 5.2 | 192 |
| BERT-base | 16 | 28.4 | 563 |
