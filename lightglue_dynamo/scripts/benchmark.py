"""Reproducible LightGlue pipeline benchmarks with explicit warmup and synchronization."""

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal

import cv2
import numpy as np
import typer

from lightglue_dynamo.cli_utils import preload_nvidia_libraries
from lightglue_dynamo.preprocessors import DISKPreprocessor, RaCoPreprocessor, SuperPointPreprocessor


def _measure(
    operation: Callable[[], Any], *, warmup: int, runs: int, synchronize: Callable[[], None]
) -> dict[str, float]:
    for _ in range(warmup):
        operation()
    synchronize()
    timings = np.empty(runs, dtype=np.float64)
    for index in range(runs):
        synchronize()
        start = time.perf_counter()
        operation()
        synchronize()
        timings[index] = (time.perf_counter() - start) * 1000
    return {
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "minimum_ms": float(timings.min()),
    }


def _images(left: Path, right: Path, extractor: str, height: int, width: int) -> np.ndarray:
    raw = [cv2.imread(str(path)) for path in (left, right)]
    if any(image is None for image in raw):
        raise typer.BadParameter("Failed to read one or both input images.")
    resized = np.stack([cv2.resize(image, (width, height)) for image in raw if image is not None])
    if extractor == "superpoint":
        result = SuperPointPreprocessor.preprocess(resized)
    elif extractor == "disk":
        result = DISKPreprocessor.preprocess(resized)
    else:
        result = RaCoPreprocessor.preprocess(resized)
    return result.astype(np.float32, copy=False)


def _benchmark_ort(
    backend: str, model_path: Path, images: np.ndarray, *, warmup: int, runs: int
) -> tuple[dict[str, Any], float]:
    # This is harmless for a CPU-only install and also permits benchmarking the
    # CPU EP from an environment that happens to contain the GPU wheel.
    preload_nvidia_libraries(tensorrt=backend == "ort-tensorrt")
    import onnxruntime as ort

    providers = {
        "ort-cpu": ["CPUExecutionProvider"],
        "ort-cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "ort-tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    }[backend]
    start = time.perf_counter()
    session = ort.InferenceSession(str(model_path), providers=providers)
    initialization_ms = (time.perf_counter() - start) * 1000
    results = {
        "full_pipeline": _measure(
            lambda: session.run(None, {"images": images}), warmup=warmup, runs=runs, synchronize=lambda: None
        )
    }
    return results, initialization_ms


def _benchmark_torch(
    backend: str, images: np.ndarray, *, num_keypoints: int, warmup: int, runs: int
) -> tuple[dict[str, Any], float, float | None]:
    import torch

    from lightglue_dynamo.config import Extractor
    from lightglue_dynamo.models import LightGlue, Pipeline, RaCoALIKED

    if not torch.cuda.is_available():
        raise typer.BadParameter("PyTorch CUDA benchmarking requires a CUDA-capable installation and device.")
    device = torch.device("cuda")
    start = time.perf_counter()
    extractor = RaCoALIKED(num_keypoints=num_keypoints).eval().to(device)
    matcher = LightGlue(**Extractor.raco_aliked.lightglue_config).eval().to(device)
    pipeline = Pipeline(extractor, matcher).eval()
    if backend == "torch-compile":
        pipeline = torch.compile(pipeline)
    initialization_ms = (time.perf_counter() - start) * 1000
    tensor = torch.from_numpy(images).to(device)
    synchronize = torch.cuda.synchronize
    compilation_ms: float | None = None
    if backend == "torch-compile":
        synchronize()
        compile_start = time.perf_counter()
        with torch.inference_mode():
            pipeline(tensor)
        synchronize()
        compilation_ms = (time.perf_counter() - compile_start) * 1000

    with torch.inference_mode():
        keypoints, detector_scores, descriptors, ranker_scores = extractor(tensor)
        del detector_scores, ranker_scores
        size = keypoints.new_tensor([tensor.shape[-1], tensor.shape[-2]])
        normalized = (keypoints - size / 2) / (size.max() / 2)
        results = {
            "raco_detector_ranker": _measure(
                lambda: extractor.raco(tensor), warmup=warmup, runs=runs, synchronize=synchronize
            ),
            "aliked_descriptor_sampling": _measure(
                lambda: extractor.aliked(tensor, keypoints), warmup=warmup, runs=runs, synchronize=synchronize
            ),
            "lightglue_plus": _measure(
                lambda: matcher(normalized, descriptors), warmup=warmup, runs=runs, synchronize=synchronize
            ),
            "full_pipeline": _measure(lambda: pipeline(tensor), warmup=warmup, runs=runs, synchronize=synchronize),
        }
    return results, initialization_ms, compilation_ms


def main(
    left_image: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    right_image: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    backend: Annotated[
        Literal["pytorch", "torch-compile", "ort-cpu", "ort-cuda", "ort-tensorrt"], typer.Option()
    ] = "ort-cpu",
    model: Annotated[Path | None, typer.Option(exists=True, dir_okay=False, readable=True)] = None,
    extractor: Annotated[Literal["superpoint", "disk", "raco_aliked"], typer.Option()] = "raco_aliked",
    height: Annotated[int, typer.Option(min=1)] = 1024,
    width: Annotated[int, typer.Option(min=1)] = 1024,
    num_keypoints: Annotated[int, typer.Option(min=1)] = 1024,
    warmup: Annotated[int, typer.Option(min=0)] = 10,
    runs: Annotated[int, typer.Option(min=1)] = 100,
) -> None:
    """Benchmark initialization separately from synchronized steady-state inference."""
    preprocess_start = time.perf_counter()
    images = _images(left_image, right_image, extractor, height, width)
    preprocessing_ms = (time.perf_counter() - preprocess_start) * 1000
    if backend.startswith("ort-"):
        if model is None:
            raise typer.BadParameter("--model is required for ONNX Runtime backends.")
        results, initialization_ms = _benchmark_ort(backend, model, images, warmup=warmup, runs=runs)
    else:
        if extractor != "raco_aliked":
            raise typer.BadParameter("PyTorch component benchmarking currently supports raco_aliked only.")
        results, initialization_ms, compilation_ms = _benchmark_torch(
            backend, images, num_keypoints=num_keypoints, warmup=warmup, runs=runs
        )
    report = {
        "backend": backend,
        "extractor": extractor,
        "shape": list(images.shape),
        "warmup": warmup,
        "runs": runs,
        "preprocessing_ms": preprocessing_ms,
        "initialization_ms": initialization_ms,
        "compilation_ms": compilation_ms if not backend.startswith("ort-") else None,
        "measurements": results,
        "transfer_scope": "NumPy feed and output materialization included for ONNX Runtime; input upload excluded for PyTorch",
    }
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    typer.run(main)
