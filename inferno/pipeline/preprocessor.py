"""Input preprocessing"""
from typing import Any, Dict
import numpy as np

class Preprocessor:
    """Input preprocessing utilities"""
    
    @staticmethod
    def normalize(image: np.ndarray, mean: list = None, std: list = None) -> np.ndarray:
        """Normalize image"""
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        
        image = image.astype(np.float32) / 255.0
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
        return image
    
    @staticmethod
    def resize(image: np.ndarray, size: tuple) -> np.ndarray:
        """Resize image"""
        from PIL import Image
        img = Image.fromarray(image)
        img = img.resize(size, Image.BILINEAR)
        return np.array(img)
    
    @staticmethod
    def to_chw(image: np.ndarray) -> np.ndarray:
        """Convert HWC to CHW format"""
        return np.transpose(image, (2, 0, 1))
    
    @staticmethod
    def add_batch_dim(data: np.ndarray) -> np.ndarray:
        """Add batch dimension"""
        return np.expand_dims(data, axis=0)
