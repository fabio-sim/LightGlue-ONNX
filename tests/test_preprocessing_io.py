from pathlib import Path

import cv2
import numpy as np

from lightglue_dynamo.preprocessors import DISKPreprocessor, RaCoPreprocessor, prepare_host_images


def test_prepare_host_images_reports_stages_and_contiguous_input() -> None:
    root = Path(__file__).resolve().parents[1]
    prepared = prepare_host_images(
        (root / "assets/sacre_coeur1.jpg", root / "assets/sacre_coeur2.jpg"), 64, 96, RaCoPreprocessor
    )
    assert prepared.images.shape == (2, 3, 96, 64)
    assert prepared.images.dtype == np.float32
    assert prepared.images.flags.c_contiguous
    assert prepared.original_shapes == ((731, 1024), (1024, 768))
    assert prepared.read_decode_ms > 0
    assert prepared.resize_ms > 0
    assert prepared.tensorize_ms > 0
    assert prepared.total_ms >= prepared.read_decode_ms + prepared.resize_ms


def test_prepare_host_images_honors_interpolation_and_target_dtype() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = (root / "assets/sacre_coeur1.jpg", root / "assets/sacre_coeur2.jpg")
    prepared = prepare_host_images(paths, 64, 96, DISKPreprocessor, interpolation=cv2.INTER_NEAREST, dtype=np.float16)
    raw = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    expected = np.stack(
        [cv2.resize(image, (64, 96), interpolation=cv2.INTER_NEAREST) for image in raw]  # type: ignore[arg-type]
    )
    expected = np.ascontiguousarray(DISKPreprocessor.preprocess(expected), dtype=np.float16)
    assert prepared.images.dtype == np.float16
    assert prepared.images.flags.c_contiguous
    np.testing.assert_array_equal(prepared.images, expected)
