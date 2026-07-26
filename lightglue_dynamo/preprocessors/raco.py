import cv2
import numpy as np

from .base import PreprocessorBase


class RaCoPreprocessor(PreprocessorBase):
    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        """Convert OpenCV BGR uint8 images to RGB NCHW float images in [0, 1]."""
        if image.ndim < 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected (..., H, W, 3) BGR images, got {image.shape}")
        leading_shape = image.shape[:-3]
        height, width = image.shape[-3:-1]
        flattened = image.reshape(-1, height, width, 3)
        blob = cv2.dnn.blobFromImages(
            flattened, scalefactor=1 / 255.0, size=(width, height), swapRB=True, crop=False, ddepth=cv2.CV_32F
        )
        return blob.reshape(*leading_shape, 3, height, width)
