"""Metrics collection"""
from typing import Dict, List
from dataclasses import dataclass
import time

@dataclass
class LatencyMetric:
    """Latency measurement"""
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float

class MetricsCollector:
    """Collect and report metrics"""
    
    def __init__(self):
        self._latencies: List[float] = []
        self._request_count = 0
        self._error_count = 0
    
    def record_latency(self, latency_ms: float):
        """Record request latency"""
        self._latencies.append(latency_ms)
        self._request_count += 1
    
    def record_error(self):
        """Record error"""
        self._error_count += 1
    
    def get_latency_stats(self) -> LatencyMetric:
        """Get latency statistics"""
        import numpy as np
        if not self._latencies:
            return LatencyMetric(0, 0, 0, 0)
        
        latencies = np.array(self._latencies)
        return LatencyMetric(
            mean_ms=float(np.mean(latencies)),
            p50_ms=float(np.percentile(latencies, 50)),
            p95_ms=float(np.percentile(latencies, 95)),
            p99_ms=float(np.percentile(latencies, 99))
        )
    
    def get_throughput(self, window_seconds: float = 60) -> float:
        """Get requests per second"""
        if not self._latencies:
            return 0.0
        return self._request_count / window_seconds
    
    def reset(self):
        """Reset metrics"""
        self._latencies.clear()
        self._request_count = 0
        self._error_count = 0
