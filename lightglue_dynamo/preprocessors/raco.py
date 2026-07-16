import numpy as np

from .base import PreprocessorBase


class RaCoPreprocessor(PreprocessorBase):
    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        """Convert OpenCV BGR uint8 images to RGB NCHW float images in [0, 1]."""
        image = image[..., ::-1].astype(np.float32) / np.float32(255)
        axes = [*list(range(image.ndim - 3)), -1, -3, -2]
        return image.transpose(*axes)
