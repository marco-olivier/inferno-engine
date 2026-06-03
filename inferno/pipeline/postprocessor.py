"""Output postprocessing"""
from typing import Dict, List, Any
import numpy as np

class Postprocessor:
    """Output postprocessing utilities"""
    
    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        """Apply softmax"""
        exp = np.exp(logits - np.max(logits))
        return exp / exp.sum()
    
    @staticmethod
    def top_k(probs: np.ndarray, k: int = 5) -> List[tuple]:
        """Get top-k predictions"""
        indices = np.argsort(probs)[::-1][:k]
        return [(int(i), float(probs[i])) for i in indices]
    
    @staticmethod
    def nms(boxes: np.ndarray, scores: np.ndarray, 
            threshold: float = 0.5) -> List[int]:
        """Non-maximum suppression"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        return keep
