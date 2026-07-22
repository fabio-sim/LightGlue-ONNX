import os
from pathlib import Path

import cv2
import numpy as np
import pytest


def test_cuda_raco_preprocessor_rejects_fp16_model_input() -> None:
    from lightglue_dynamo.preprocessors import CudaRaCoPreprocessor

    with pytest.raises(ValueError, match="float32 ONNX input"):
        CudaRaCoPreprocessor(128, 128, dtype="float16")


@pytest.mark.parametrize("size", [128, 1024])
def test_cuda_raco_preprocessor_produces_device_ortvalue(size: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NVIMGCODEC_EXTENSIONS_PATH", raising=False)
    pytest.importorskip("cvcuda")
    from lightglue_dynamo.cli_utils import preload_nvidia_libraries

    preload_nvidia_libraries()
    ort = pytest.importorskip("onnxruntime")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("ONNX Runtime CUDA provider is unavailable")

    from lightglue_dynamo.preprocessors import CudaRaCoPreprocessor, RaCoPreprocessor

    root = Path(__file__).resolve().parents[1]
    paths = (root / "assets/sacre_coeur1.jpg", root / "assets/sacre_coeur2.jpg")
    preprocessor = CudaRaCoPreprocessor(size, size)
    assert "NVIMGCODEC_EXTENSIONS_PATH" not in os.environ
    prepared = preprocessor.prepare(paths)
    try:
        value = prepared.to_ort_value(ort)
        assert value.device_name() == "cuda"
        actual = value.numpy()
        raw = np.stack(
            [cv2.resize(cv2.imread(str(path)), (size, size), interpolation=cv2.INTER_AREA) for path in paths]
        )
        expected = RaCoPreprocessor.preprocess(raw)
        assert actual.shape == expected.shape
        # nvJPEG and OpenCV's JPEG decoder differ slightly in chroma reconstruction.
        difference = np.abs(actual - expected)
        assert float(difference.mean()) < 0.002
        if size == 128:
            assert float(difference.max()) <= 4 / 255
    finally:
        prepared.release()
        preprocessor.close()
