from pathlib import Path

import numpy as np

from lightglue_dynamo.preprocessors import RaCoPreprocessor, prepare_host_images


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
