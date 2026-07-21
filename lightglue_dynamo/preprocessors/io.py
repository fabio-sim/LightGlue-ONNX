from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from .base import PreprocessorBase


@dataclass(frozen=True)
class HostPreparedImages:
    images: np.ndarray
    resized_bgr: list[np.ndarray]
    original_shapes: tuple[tuple[int, int], tuple[int, int]]
    read_decode_ms: float
    resize_ms: float
    tensorize_ms: float
    total_ms: float


def prepare_host_images(
    paths: tuple[Path, Path], width: int, height: int, preprocessor: type[PreprocessorBase]
) -> HostPreparedImages:
    """Read, resize, and pack an image pair for a pipeline model."""
    start = time.perf_counter()
    loaded = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    if any(image is None for image in loaded):
        raise FileNotFoundError(f"Failed to read {paths[0]} or {paths[1]}")
    typed = cast(list[np.ndarray], loaded)
    read_end = time.perf_counter()
    original_shapes = cast(
        tuple[tuple[int, int], tuple[int, int]], tuple((image.shape[0], image.shape[1]) for image in typed)
    )
    resized = [cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA) for image in typed]
    resize_end = time.perf_counter()
    images = preprocessor.preprocess(np.stack(resized))
    tensorize_end = time.perf_counter()
    return HostPreparedImages(
        images=np.ascontiguousarray(images),
        resized_bgr=resized,
        original_shapes=original_shapes,
        read_decode_ms=(read_end - start) * 1000,
        resize_ms=(resize_end - read_end) * 1000,
        tensorize_ms=(tensorize_end - resize_end) * 1000,
        total_ms=(tensorize_end - start) * 1000,
    )
