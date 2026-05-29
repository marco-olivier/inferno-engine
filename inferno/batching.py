"""Dynamic batching engine"""
from typing import List, Optional
from dataclasses import dataclass
import time
import threading
from queue import Queue

@dataclass
class BatchConfig:
    """Batch configuration"""
    max_batch_size: int = 32
    timeout_ms: float = 100.0
    max_queue_size: int = 1000

class DynamicBatcher:
    """Batches requests for efficient GPU utilization"""
    
    def __init__(self, max_batch_size: int = 32, timeout_ms: float = 100.0):
        self.config = BatchConfig(
            max_batch_size=max_batch_size,
            timeout_ms=timeout_ms
        )
        self._queue = Queue(maxsize=self.config.max_queue_size)
        self._batch = []
        self._lock = threading.Lock()
    
    def add_request(self, request) -> None:
        """Add request to batch queue"""
        self._queue.put(request)
    
    def batch(self, requests: list) -> List[list]:
        """Split requests into batches"""
        batches = []
        for i in range(0, len(requests), self.config.max_batch_size):
            batch = requests[i:i + self.config.max_batch_size]
            batches.append(batch)
        return batches
    
    def get_batch(self) -> Optional[list]:
        """Get next batch of requests"""
        with self._lock:
            if not self._batch:
                return None
            batch = self._batch
            self._batch = []
            return batch
    
    def _should_flush(self) -> bool:
        """Check if batch should be flushed"""
        if len(self._batch) >= self.config.max_batch_size:
            return True
        return False
