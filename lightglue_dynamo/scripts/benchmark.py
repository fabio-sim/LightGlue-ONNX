"""Streaming MegaDepth-1500 benchmarks for RaCo-ALIKED-LightGlue+.

The runner writes one JSON object per pair and a separate summary. Results are
resumable so long GPU sweeps can be continued without keeping the dataset in
host memory or repeating completed pairs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, cast

import numpy as np
import typer

from lightglue_dynamo.cli_utils import preload_nvidia_libraries
from lightglue_dynamo.preprocessors import (
    CudaPreparedImages,
    CudaRaCoPreprocessor,
    RaCoPreprocessor,
    prepare_host_images,
)

app = typer.Typer(no_args_is_help=True)

Backend = Literal["pytorch", "torch-compile", "ort-cuda", "ort-tensorrt", "tensorrt"]
Precision = Literal["fp32", "fp16"]
CompileMode = Literal["default", "reduce-overhead", "max-autotune-no-cudagraphs", "max-autotune"]
PreprocessingBackend = Literal["opencv", "cuda"]

# `filter_matches` has a data-dependent output length. Dynamo first specializes
# that length, then compiles generic, zero-length, and one-length variants as it
# encounters them. These canonical pairs cover every such variant observed in
# the requested MegaDepth matrix, keeping lazy compilation out of timed records.
TORCH_COMPILE_STABILIZATION_PAIRS = (1, 239, 235, 96, 135, 330, 163, 250, 394, 596, 286, 518, 95, 246, 844, 140)


@dataclass(frozen=True)
class Pair:
    index: int
    group: str
    group_index: int
    left: Path
    right: Path
    intrinsics0: np.ndarray
    intrinsics1: np.ndarray
    pose0: np.ndarray
    pose1: np.ndarray
    overlap: float


@dataclass
class InferenceResult:
    keypoints: np.ndarray
    matches: np.ndarray
    mscores: np.ndarray
    wall_ms: float
    device_ms: float | None
    h2d_ms: float | None = None
    d2h_ms: float | None = None


@dataclass
class PreparedBenchmarkInput:
    value: np.ndarray | CudaPreparedImages
    original_shapes: tuple[tuple[int, int], tuple[int, int]]
    preprocessing_ms: float
    read_decode_ms: float | None
    resize_ms: float | None
    tensorize_ms: float | None
    submission_ms: float | None

    def release(self) -> None:
        if isinstance(self.value, CudaPreparedImages):
            self.value.release()


class Executor(Protocol):
    initialization_ms: float
    compilation_ms: float | None

    def infer(self, images: np.ndarray | CudaPreparedImages) -> InferenceResult: ...

    def reset_memory(self) -> None: ...

    def peak_memory_mib(self) -> float | None: ...

    def close(self) -> None: ...


class DeviceMemorySampler:
    """Approximate process-external GPU-memory high-water via nvidia-smi."""

    def __init__(self, interval_s: float = 0.02) -> None:
        self.interval_s = interval_s
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _read() -> float | None:
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        values = [float(line.strip()) for line in output.splitlines() if line.strip()]
        return max(values) if values else None

    def start(self) -> None:
        first = self._read()
        if first is not None:
            self.samples.append(first)

        def sample() -> None:
            while not self._stop.wait(self.interval_s):
                value = self._read()
                if value is not None:
                    self.samples.append(value)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[float | None, float | None]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        last = self._read()
        if last is not None:
            self.samples.append(last)
        if not self.samples:
            return None, None
        return self.samples[0], max(self.samples)


def _load_scene(path: Path) -> Mapping[str, Any]:
    scene = np.load(path, allow_pickle=True)
    if isinstance(scene, Mapping):
        return scene
    if getattr(scene, "files", None) == ["arr_0"]:
        return cast(Mapping[str, Any], scene["arr_0"].item())
    return cast(Mapping[str, Any], scene)


def iter_pairs(dataset_root: Path, scene_info_root: Path) -> Iterator[Pair]:
    manifest = scene_info_root / "megadepth_test_1500.txt"
    if not manifest.is_file():
        raise typer.BadParameter(f"Missing canonical pair manifest: {manifest}")
    pair_index = 0
    for line in manifest.read_text().splitlines():
        group = line.strip()
        if not group:
            continue
        scene_path = scene_info_root / group
        if not scene_path.exists() and not group.endswith(".npz"):
            scene_path = scene_info_root / f"{group}.npz"
        scene = _load_scene(scene_path)
        for group_index, pair_info in enumerate(scene["pair_infos"]):
            indices, overlap, _central_match = pair_info
            index0, index1 = (int(value) for value in indices)
            yield Pair(
                index=pair_index,
                group=scene_path.stem,
                group_index=group_index,
                left=dataset_root / str(scene["image_paths"][index0]),
                right=dataset_root / str(scene["image_paths"][index1]),
                intrinsics0=np.asarray(scene["intrinsics"][index0], dtype=np.float64),
                intrinsics1=np.asarray(scene["intrinsics"][index1], dtype=np.float64),
                pose0=np.asarray(scene["poses"][index0], dtype=np.float64),
                pose1=np.asarray(scene["poses"][index1], dtype=np.float64),
                overlap=float(overlap),
            )
            pair_index += 1


def _read_images(pair: Pair, size: int) -> tuple[np.ndarray, tuple[tuple[int, int], tuple[int, int]], float]:
    prepared = prepare_host_images((pair.left, pair.right), size, size, RaCoPreprocessor)
    return prepared.images, prepared.original_shapes, prepared.total_ms


def _scaled_intrinsics(intrinsics: np.ndarray, original_shape: tuple[int, int], size: int) -> np.ndarray:
    height, width = original_shape
    scale = np.diag([size / width, size / height, 1.0])
    return scale @ intrinsics


def _match_metrics(
    pair: Pair,
    original_shapes: tuple[tuple[int, int], tuple[int, int]],
    size: int,
    keypoints: np.ndarray,
    matches: np.ndarray,
    mscores: np.ndarray,
) -> dict[str, Any]:
    count = int(matches.shape[0])
    record: dict[str, Any] = {
        "match_count": count,
        "mscore_mean": float(mscores.mean()) if count else None,
        "mscore_median": float(np.median(mscores)) if count else None,
        "mscore_p10": float(np.percentile(mscores, 10)) if count else None,
        "mscore_p90": float(np.percentile(mscores, 90)) if count else None,
    }
    if count == 0:
        record.update(
            {
                "epipolar_median_px": None,
                "epipolar_p90_px": None,
                "epipolar_precision_1px": None,
                "epipolar_precision_3px": None,
                "epipolar_precision_5px": None,
            }
        )
        return record

    batch_ids = matches[:, 0].astype(np.int64, copy=False)
    if np.any(batch_ids != 0):
        raise ValueError("MegaDepth benchmark expects one image pair per inference")
    points0 = keypoints[0, matches[:, 1].astype(np.int64, copy=False)]
    points1 = keypoints[1, matches[:, 2].astype(np.int64, copy=False)]
    k0 = _scaled_intrinsics(pair.intrinsics0, original_shapes[0], size)
    k1 = _scaled_intrinsics(pair.intrinsics1, original_shapes[1], size)
    relative_pose = pair.pose1 @ np.linalg.inv(pair.pose0)
    translation = relative_pose[:3, 3]
    tx = np.array(
        [
            [0.0, -translation[2], translation[1]],
            [translation[2], 0.0, -translation[0]],
            [-translation[1], translation[0], 0.0],
        ]
    )
    essential = tx @ relative_pose[:3, :3]
    fundamental = np.linalg.inv(k1).T @ essential @ np.linalg.inv(k0)
    ones = np.ones((count, 1), dtype=np.float64)
    homogeneous0 = np.concatenate([points0.astype(np.float64), ones], axis=1)
    homogeneous1 = np.concatenate([points1.astype(np.float64), ones], axis=1)
    fx0 = homogeneous0 @ fundamental.T
    ftx1 = homogeneous1 @ fundamental
    numerator = np.square(np.sum(homogeneous1 * fx0, axis=1))
    denominator = np.square(fx0[:, 0]) + np.square(fx0[:, 1]) + np.square(ftx1[:, 0]) + np.square(ftx1[:, 1])
    sampson_px = np.sqrt(numerator / np.maximum(denominator, np.finfo(np.float64).eps))
    record.update(
        {
            "epipolar_median_px": float(np.median(sampson_px)),
            "epipolar_p90_px": float(np.percentile(sampson_px, 90)),
            "epipolar_precision_1px": float(np.mean(sampson_px <= 1.0)),
            "epipolar_precision_3px": float(np.mean(sampson_px <= 3.0)),
            "epipolar_precision_5px": float(np.mean(sampson_px <= 5.0)),
        }
    )
    return record


class TorchExecutor:
    def __init__(
        self, *, num_keypoints: int, precision: Precision, compile_mode: CompileMode | None, portable: bool, cache: Path
    ) -> None:
        import torch

        from lightglue_dynamo.config import Extractor
        from lightglue_dynamo.models import LightGlue, Pipeline, RaCoALIKED

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable")
        torch.set_float32_matmul_precision("highest")
        self.torch = torch
        self.precision = precision
        start = time.perf_counter()
        extractor = RaCoALIKED(num_keypoints=num_keypoints, portable_deform_conv=portable).eval().cuda()
        matcher = LightGlue(**Extractor.raco_aliked.lightglue_config).eval().cuda()
        pipeline = Pipeline(extractor, matcher).eval()
        if compile_mode is not None:
            inductor_cache = cache / "torchinductor"
            inductor_cache.mkdir(parents=True, exist_ok=True)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache.resolve())
            pipeline = torch.compile(pipeline, mode=None if compile_mode == "default" else compile_mode)
        self.pipeline = pipeline
        torch.cuda.synchronize()
        self.initialization_ms = (time.perf_counter() - start) * 1000
        self.compilation_ms: float | None = None
        self._compile_mode = compile_mode

    def _context(self) -> AbstractContextManager[Any]:
        dtype = self.torch.float16 if self.precision == "fp16" else self.torch.float32
        return self.torch.autocast("cuda", dtype=dtype, enabled=self.precision == "fp16")

    def infer(self, images: np.ndarray) -> InferenceResult:
        torch = self.torch
        torch.cuda.synchronize()
        transfer_start = time.perf_counter()
        tensor = torch.from_numpy(images).cuda()
        torch.cuda.synchronize()
        h2d_ms = (time.perf_counter() - transfer_start) * 1000
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start_event.record()
        with torch.inference_mode(), self._context():
            outputs = self.pipeline(tensor)
        end_event.record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - wall_start) * 1000
        device_ms = float(start_event.elapsed_time(end_event))
        transfer_start = time.perf_counter()
        keypoints, matches, mscores = (output.detach().cpu().numpy() for output in outputs)
        torch.cuda.synchronize()
        d2h_ms = (time.perf_counter() - transfer_start) * 1000
        return InferenceResult(keypoints, matches, mscores, wall_ms, device_ms, h2d_ms, d2h_ms)

    def compile(self, images: np.ndarray) -> None:
        if self._compile_mode is None:
            return
        self.torch.cuda.synchronize()
        start = time.perf_counter()
        self.infer(images)
        self.torch.cuda.synchronize()
        self.compilation_ms = (time.perf_counter() - start) * 1000

    def reset_memory(self) -> None:
        self.torch.cuda.reset_peak_memory_stats()

    def peak_memory_mib(self) -> float:
        return float(self.torch.cuda.max_memory_allocated() / 2**20)

    def close(self) -> None:
        return


class OrtExecutor:
    def __init__(
        self,
        model: Path,
        *,
        backend: Literal["ort-cuda", "ort-tensorrt"],
        precision: Precision,
        cache: Path,
        gpu_mem_limit_mib: int,
    ) -> None:
        preload_nvidia_libraries(tensorrt=backend == "ort-tensorrt")
        import onnxruntime as ort

        cache.mkdir(parents=True, exist_ok=True)
        providers: list[tuple[str, dict[str, Any]]] = []
        if backend == "ort-tensorrt":
            providers.append(
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": str(cache / "engines"),
                        "trt_timing_cache_enable": True,
                        "trt_timing_cache_path": str(cache / "timing"),
                        "trt_fp16_enable": precision == "fp16",
                    },
                )
            )
        cuda_options = {
            "use_tf32": "0",
            "gpu_mem_limit": str(gpu_mem_limit_mib * 2**20),
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_use_max_workspace": "0",
        }
        providers.extend([("CUDAExecutionProvider", cuda_options), ("CPUExecutionProvider", {})])
        session_options = ort.SessionOptions()
        if backend == "ort-cuda" and precision == "fp16":
            # ORT 1.27's level-3 CastFloat16/layout pass aborts on this
            # converted dynamic graph. Level 2 retains all extended fusions
            # and is the strongest optimization level that executes safely.
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        start = time.perf_counter()
        self.session = ort.InferenceSession(str(model), session_options, providers=providers)
        self.ort = ort
        self.initialization_ms = (time.perf_counter() - start) * 1000
        self.compilation_ms = None
        self.input_dtype = np.float16 if self.session.get_inputs()[0].type == "tensor(float16)" else np.float32

    def infer(self, images: np.ndarray | CudaPreparedImages) -> InferenceResult:
        transfer_start = time.perf_counter()
        if isinstance(images, CudaPreparedImages):
            input_value = images.to_ort_value(self.ort)
        else:
            feed = np.ascontiguousarray(images, dtype=self.input_dtype)
            input_value = self.ort.OrtValue.ortvalue_from_numpy(feed, "cuda", 0)
        binding = self.session.io_binding()
        binding.bind_ortvalue_input("images", input_value)
        for output in self.session.get_outputs():
            binding.bind_output(output.name, "cuda", 0)
        binding.synchronize_inputs()
        h2d_ms = (time.perf_counter() - transfer_start) * 1000
        start = time.perf_counter()
        self.session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        wall_ms = (time.perf_counter() - start) * 1000
        transfer_start = time.perf_counter()
        keypoints, matches, mscores = binding.copy_outputs_to_cpu()
        d2h_ms = (time.perf_counter() - transfer_start) * 1000
        return InferenceResult(keypoints, matches, mscores, wall_ms, wall_ms, h2d_ms, d2h_ms)

    def reset_memory(self) -> None:
        return

    def peak_memory_mib(self) -> None:
        return None

    def close(self) -> None:
        return


def _prepare_tensorrt_onnx(source: Path, destination: Path) -> Path:
    """Materialize constant reduction axes as initializers for TensorRT 10.16.

    Dynamo exports some statically known axes as Constant -> Reshape values.
    ONNX Runtime accepts that representation, but the standalone TensorRT ONNX
    parser requires the second input of Reduce operators to be an initializer.
    """
    if destination.exists() and destination.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return destination
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(source)
    producers = {output: node for node in model.graph.node for output in node.output}

    def constant_value(name: str) -> np.ndarray | None:
        node = producers.get(name)
        if node is None:
            return None
        if node.op_type == "Constant":
            value_attribute = next(
                (attribute for attribute in node.attribute if attribute.name.startswith("value")), None
            )
            if value_attribute is None:
                return None
            value = helper.get_attribute_value(value_attribute)
            if isinstance(value, onnx.TensorProto):
                value = numpy_helper.to_array(value)
            return np.asarray(value)
        if node.op_type == "Reshape" and len(node.input) == 2:
            value = constant_value(node.input[0])
            shape = constant_value(node.input[1])
            if value is not None and shape is not None:
                return value.reshape(tuple(int(dimension) for dimension in shape.flat))
        return None

    initializer_names = {initializer.name for initializer in model.graph.initializer}
    replaced_outputs: set[str] = set()
    for node in model.graph.node:
        if not node.op_type.startswith("Reduce") or len(node.input) < 2 or node.input[1] in initializer_names:
            continue
        axes_name = node.input[1]
        axes = constant_value(axes_name)
        if axes is None:
            continue
        model.graph.initializer.append(numpy_helper.from_array(axes.astype(np.int64), name=axes_name))
        initializer_names.add(axes_name)
        replaced_outputs.add(axes_name)
    if replaced_outputs:
        retained = [node for node in model.graph.node if not any(output in replaced_outputs for output in node.output)]
        del model.graph.node[:]
        model.graph.node.extend(retained)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, destination, save_as_external_data=False)
    return destination


class TensorRTExecutor:
    def __init__(self, model: Path, *, precision: Precision, engine_path: Path, size: int) -> None:
        preload_nvidia_libraries(tensorrt=True)
        from polygraphy.backend.common import BytesFromPath
        from polygraphy.backend.trt import (
            CreateConfig,
            EngineFromBytes,
            EngineFromNetwork,
            NetworkFromOnnxPath,
            Profile,
            SaveEngine,
            TrtRunner,
        )

        start = time.perf_counter()
        if engine_path.exists():
            loader: Any = EngineFromBytes(BytesFromPath(str(engine_path)))
        else:
            trt_model = _prepare_tensorrt_onnx(model, engine_path.parent.parent / "tensorrt-ready.onnx")
            profile = Profile().add("images", min=(2, 3, size, size), opt=(2, 3, size, size), max=(2, 3, size, size))
            loader = EngineFromNetwork(
                NetworkFromOnnxPath(str(trt_model)), config=CreateConfig(fp16=precision == "fp16", profiles=[profile])
            )
            engine_path.parent.mkdir(parents=True, exist_ok=True)
            loader = SaveEngine(loader, str(engine_path))
        self.runner = TrtRunner(loader)
        self.runner.activate()
        self.initialization_ms = (time.perf_counter() - start) * 1000
        self.compilation_ms = None

    def infer(self, images: np.ndarray) -> InferenceResult:
        start = time.perf_counter()
        outputs = self.runner.infer(feed_dict={"images": images})
        wall_ms = (time.perf_counter() - start) * 1000
        device_ms: float | None = float(self.runner.last_inference_time() * 1000)
        # CUDA-event timing occasionally reports a wrapped negative duration.
        # Preserve the wall measurement and mark the invalid device sample absent.
        if device_ms <= 0 or device_ms > wall_ms * 1.5:
            device_ms = None
        return InferenceResult(outputs["keypoints"], outputs["matches"], outputs["mscores"], wall_ms, device_ms)

    def reset_memory(self) -> None:
        return

    def peak_memory_mib(self) -> None:
        return None

    def close(self) -> None:
        self.runner.deactivate()


def _percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "minimum": float(array.min()),
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "maximum": float(array.max()),
    }


def _summarize(records: list[dict[str, Any]], configuration: dict[str, Any]) -> dict[str, Any]:
    numeric_fields = [
        "preprocessing_ms",
        "preprocessing_read_decode_ms",
        "preprocessing_resize_ms",
        "preprocessing_tensorize_ms",
        "preprocessing_submission_ms",
        "h2d_ms",
        "inference_wall_ms",
        "device_ms",
        "d2h_ms",
        "executor_total_ms",
        "full_pipeline_ms",
        "pipeline_iteration_ms",
        "match_count",
        "mscore_mean",
        "mscore_median",
        "mscore_p10",
        "mscore_p90",
        "epipolar_median_px",
        "epipolar_p90_px",
        "epipolar_precision_1px",
        "epipolar_precision_3px",
        "epipolar_precision_5px",
    ]
    valid_values = {
        field: [float(record[field]) for record in records if record.get(field) is not None] for field in numeric_fields
    }
    distributions = {field: _percentiles(values) for field, values in valid_values.items()}
    inference = valid_values["inference_wall_ms"]
    window = min(100, len(inference) // 2)
    first_window_median = float(np.median(inference[:window])) if window else None
    last_window_median = float(np.median(inference[-window:])) if window else None
    median = distributions["inference_wall_ms"]["median"]
    p99 = distributions["inference_wall_ms"]["p99"]
    maximum = distributions["inference_wall_ms"]["maximum"]
    zero_match_pairs = sum(record["match_count"] == 0 for record in records)
    return {
        "configuration": configuration,
        "completed_pairs": len(records),
        "latency_and_quality_distributions": distributions,
        "distribution_sample_counts": {field: len(values) for field, values in valid_values.items()},
        "zero_match_pairs": zero_match_pairs,
        "zero_match_rate": zero_match_pairs / len(records) if records else None,
        "latency_stationarity": {
            "window_pairs": window,
            "first_window_median_ms": first_window_median,
            "last_window_median_ms": last_window_median,
            "last_to_first_relative_change": (
                last_window_median / first_window_median - 1
                if first_window_median is not None and first_window_median > 0 and last_window_median is not None
                else None
            ),
            "p99_to_median_ratio": p99 / median if median else None,
            "maximum_to_median_ratio": maximum / median if median else None,
        },
    }


def _completed_records(path: Path, configuration: dict[str, Any]) -> tuple[list[dict[str, Any]], set[int]]:
    if not path.exists():
        return [], set()
    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if records and (not meta_path.exists() or json.loads(meta_path.read_text())["configuration"] != configuration):
        raise typer.BadParameter(f"Existing result configuration does not match: {path}")
    return records, {int(record["pair_index"]) for record in records}


def _create_executor(
    backend: Backend,
    *,
    model: Path | None,
    num_keypoints: int,
    size: int,
    precision: Precision,
    compile_mode: CompileMode,
    portable: bool,
    cache: Path,
    ort_gpu_mem_limit_mib: int,
) -> Executor:
    if backend in {"pytorch", "torch-compile"}:
        return TorchExecutor(
            num_keypoints=num_keypoints,
            precision=precision,
            compile_mode=compile_mode if backend == "torch-compile" else None,
            portable=portable,
            cache=cache,
        )
    if model is None:
        raise typer.BadParameter("--model is required for ONNX Runtime and TensorRT backends")
    if backend in {"ort-cuda", "ort-tensorrt"}:
        return OrtExecutor(
            model, backend=backend, precision=precision, cache=cache, gpu_mem_limit_mib=ort_gpu_mem_limit_mib
        )
    return TensorRTExecutor(model, precision=precision, engine_path=cache / "model.engine", size=size)


@app.command()
def run(
    backend: Annotated[Backend, typer.Option()],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    model: Annotated[Path | None, typer.Option(exists=True, dir_okay=False, readable=True)] = None,
    dataset_root: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path("megadepth_test_1500"),
    scene_info_root: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "megadepth_test_1500/scene_info"
    ),
    size: Annotated[int, typer.Option(min=32)] = 1024,
    num_keypoints: Annotated[int, typer.Option(min=128)] = 1024,
    precision: Annotated[Precision, typer.Option()] = "fp32",
    compile_mode: Annotated[CompileMode, typer.Option()] = "default",
    warmup: Annotated[int, typer.Option(min=0)] = 10,
    max_pairs: Annotated[int, typer.Option(min=0, help="Zero runs the complete manifest.")] = 0,
    portable_deform_conv: Annotated[bool, typer.Option("--portable-deform-conv/--torchvision-deform-conv")] = False,
    cache_root: Annotated[Path, typer.Option(file_okay=False)] = Path("data/benchmark_cache"),
    ort_gpu_mem_limit_mib: Annotated[int, typer.Option(min=1)] = 9216,
    preprocessing: Annotated[
        PreprocessingBackend,
        typer.Option(help="Image preprocessing backend. CUDA requires ORT CUDA/TensorRT and JPEG inputs."),
    ] = "opencv",
    prefetch: Annotated[
        bool,
        typer.Option("--prefetch/--no-prefetch", help="Overlap preparation of the next pair with current inference."),
    ] = False,
) -> None:
    """Benchmark one resumable configuration over the canonical MegaDepth split."""
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    if size % 32:
        raise typer.BadParameter("RaCo-ALIKED input size must be divisible by 32")
    if backend in {"ort-tensorrt", "tensorrt"} and num_keypoints > 3840:
        raise typer.BadParameter("TensorRT TopK supports at most 3840 requested output keypoints")
    if backend == "ort-cuda" and precision == "fp16" and model is not None:
        import onnx

        input_type = onnx.load(model, load_external_data=False).graph.input[0].type.tensor_type.elem_type
        if input_type != onnx.TensorProto.FLOAT16:
            raise typer.BadParameter("ORT CUDA FP16 requires an FP16-converted ONNX model")
    if preprocessing == "cuda" and backend not in {"ort-cuda", "ort-tensorrt"}:
        raise typer.BadParameter("CUDA preprocessing currently supports the ORT CUDA and ORT TensorRT backends")

    configuration = {
        "backend": backend,
        "model": str(model.resolve()) if model is not None else None,
        "size": size,
        "num_keypoints": num_keypoints,
        "precision": precision,
        "compile_mode": compile_mode if backend == "torch-compile" else None,
        "warmup": warmup,
        "portable_deform_conv": portable_deform_conv if backend.startswith("torch") else None,
        "ort_gpu_mem_limit_mib": ort_gpu_mem_limit_mib if backend.startswith("ort-") else None,
        "ort_graph_optimization_level": ("extended" if backend == "ort-cuda" and precision == "fp16" else "all")
        if backend.startswith("ort-")
        else None,
        "preprocessing": preprocessing,
        "prefetch": prefetch,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    records, completed = _completed_records(output, configuration)
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    previous_meta = json.loads(meta_path.read_text()) if records and meta_path.exists() else None
    model_tag = model.stem if model is not None else "pytorch"
    cache = cache_root / model_tag / f"{backend}-{precision}-s{size}-k{num_keypoints}-{compile_mode}"
    pairs = list(iter_pairs(dataset_root, scene_info_root))
    expected_pairs = min(len(pairs), max_pairs) if max_pairs else len(pairs)
    if len(pairs) != 1500:
        raise typer.BadParameter(f"Expected 1500 canonical pairs, found {len(pairs)}")
    warmup_pair = pairs[0]
    memory_baseline_mib = DeviceMemorySampler._read()
    executor = _create_executor(
        backend,
        model=model,
        num_keypoints=num_keypoints,
        size=size,
        precision=precision,
        compile_mode=compile_mode,
        portable=portable_deform_conv,
        cache=cache,
        ort_gpu_mem_limit_mib=ort_gpu_mem_limit_mib,
    )
    cuda_preprocessor: CudaRaCoPreprocessor | None = None
    if preprocessing == "cuda":
        if not isinstance(executor, OrtExecutor):
            raise RuntimeError("CUDA preprocessing requires an ORT executor")
        input_dtype = "float16" if executor.input_dtype == np.float16 else "float32"
        cuda_preprocessor = CudaRaCoPreprocessor(size, size, dtype=input_dtype, slots=2 if prefetch else 1)

    def prepare_pair(pair: Pair) -> PreparedBenchmarkInput:
        if cuda_preprocessor is not None:
            value = cuda_preprocessor.prepare((pair.left, pair.right))
            preprocessing_ms = value.synchronize()
            return PreparedBenchmarkInput(
                value, value.original_shapes, preprocessing_ms, None, None, None, value.submission_ms
            )
        value = prepare_host_images((pair.left, pair.right), size, size, RaCoPreprocessor)
        return PreparedBenchmarkInput(
            value.images,
            value.original_shapes,
            value.total_ms,
            value.read_decode_ms,
            value.resize_ms,
            value.tensorize_ms,
            None,
        )

    warmup_input = prepare_pair(warmup_pair)
    warmup_images = warmup_input.value
    if isinstance(executor, TorchExecutor):
        if not isinstance(warmup_images, np.ndarray):
            raise RuntimeError("Torch compilation requires host preprocessing")
        executor.compile(warmup_images)
    first_inference_start = time.perf_counter()
    executor.infer(warmup_images)
    first_inference_ms = (time.perf_counter() - first_inference_start) * 1000
    compile_stabilization_ms = None
    if isinstance(executor, TorchExecutor) and backend == "torch-compile":
        stabilization_start = time.perf_counter()
        for pair_index in TORCH_COMPILE_STABILIZATION_PAIRS:
            stabilization_input = prepare_pair(pairs[pair_index])
            executor.infer(stabilization_input.value)
            stabilization_input.release()
        compile_stabilization_ms = (time.perf_counter() - stabilization_start) * 1000
    for _ in range(warmup):
        executor.infer(warmup_images)
    sampler = DeviceMemorySampler()
    sampler.start()
    for _ in range(max(warmup, 10)):
        executor.infer(warmup_images)
    _memory_pass_start_mib, peak_device_mib = sampler.stop()
    executor.reset_memory()
    meta = {
        "configuration": configuration,
        "initialization_ms": previous_meta["initialization_ms"] if previous_meta else executor.initialization_ms,
        "compilation_ms": previous_meta["compilation_ms"] if previous_meta else executor.compilation_ms,
        "compile_stabilization_ms": (
            previous_meta.get("compile_stabilization_ms", compile_stabilization_ms)
            if previous_meta
            else compile_stabilization_ms
        ),
        "compile_stabilization_pair_indices": (
            list(TORCH_COMPILE_STABILIZATION_PAIRS) if backend == "torch-compile" else None
        ),
        "first_inference_ms": previous_meta["first_inference_ms"] if previous_meta else first_inference_ms,
        "last_resume_initialization_ms": executor.initialization_ms if previous_meta else None,
        "last_resume_compilation_ms": executor.compilation_ms if previous_meta else None,
        "expected_pairs": expected_pairs,
        "tf32_enabled": False,
        "memory_scope": "PyTorch allocator peak for PyTorch; device-wide nvidia-smi high-water for other executors",
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    warmup_input.release()
    pool: ThreadPoolExecutor | None = None
    try:
        with output.open("a") as stream:
            pending_pairs = [pair for pair in pairs[:expected_pairs] if pair.index not in completed]
            pool = ThreadPoolExecutor(max_workers=1) if prefetch and pending_pairs else None
            future: Future[PreparedBenchmarkInput] | None = (
                pool.submit(prepare_pair, pending_pairs[0]) if pool is not None else None
            )
            for position, pair in enumerate(pending_pairs):
                iteration_start = time.perf_counter()
                prepared = future.result() if future is not None else prepare_pair(pair)
                if pool is not None and position + 1 < len(pending_pairs):
                    future = pool.submit(prepare_pair, pending_pairs[position + 1])
                else:
                    future = None
                result = executor.infer(prepared.value)
                pipeline_iteration_ms = (time.perf_counter() - iteration_start) * 1000
                record = {
                    "pair_index": pair.index,
                    "group": pair.group,
                    "group_index": pair.group_index,
                    "left": str(pair.left),
                    "right": str(pair.right),
                    "overlap": pair.overlap,
                    "preprocessing_ms": prepared.preprocessing_ms,
                    "preprocessing_read_decode_ms": prepared.read_decode_ms,
                    "preprocessing_resize_ms": prepared.resize_ms,
                    "preprocessing_tensorize_ms": prepared.tensorize_ms,
                    "preprocessing_submission_ms": prepared.submission_ms,
                    "h2d_ms": result.h2d_ms,
                    "inference_wall_ms": result.wall_ms,
                    "device_ms": result.device_ms,
                    "d2h_ms": result.d2h_ms,
                    "executor_total_ms": result.wall_ms + (result.h2d_ms or 0.0) + (result.d2h_ms or 0.0),
                    "full_pipeline_ms": prepared.preprocessing_ms
                    + result.wall_ms
                    + (result.h2d_ms or 0.0)
                    + (result.d2h_ms or 0.0),
                    "pipeline_iteration_ms": pipeline_iteration_ms,
                    **_match_metrics(
                        pair, prepared.original_shapes, size, result.keypoints, result.matches, result.mscores
                    ),
                }
                prepared.release()
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                stream.flush()
                records.append(record)
                typer.echo(
                    f"[{len(records):4d}/{expected_pairs}] pair={pair.index:04d} "
                    f"wall={result.wall_ms:.2f} ms matches={record['match_count']}"
                )
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        if cuda_preprocessor is not None:
            cuda_preprocessor.close()
        executor.close()
    meta.update(
        {
            "torch_peak_allocated_mib": executor.peak_memory_mib(),
            "device_memory_baseline_mib": memory_baseline_mib,
            "device_memory_peak_mib": peak_device_mib,
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    summary = _summarize(records, configuration)
    summary.update({key: value for key, value in meta.items() if key != "configuration"})
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    typer.echo(json.dumps(summary, indent=2))
    typer.echo(f"Wrote {output} and {summary_path}")


def _parse_csv_ints(value: str, name: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",") if item]
    except ValueError as exc:
        raise typer.BadParameter(f"{name} must be a comma-separated integer list") from exc
    if not values:
        raise typer.BadParameter(f"{name} must not be empty")
    return values


def _run_logged(command: list[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as stream:
        stream.write(f"\n$ {' '.join(command)}\n")
        stream.flush()
        return subprocess.run(command, text=True, stdout=stream, stderr=subprocess.STDOUT, check=False)


@app.command()
def matrix(
    results_root: Annotated[Path, typer.Option(file_okay=False)] = Path("data/benchmark_results/matrix"),
    models_root: Annotated[Path, typer.Option(file_okay=False)] = Path("data/benchmark_models"),
    cache_root: Annotated[Path, typer.Option(file_okay=False)] = Path("data/benchmark_cache"),
    backends: Annotated[
        str, typer.Option(help="Comma-separated pytorch,torch-compile,ort-cuda,ort-tensorrt,tensorrt list.")
    ] = "torch-compile,ort-cuda,ort-tensorrt",
    precisions: Annotated[str, typer.Option(help="Comma-separated fp32,fp16 list.")] = "fp32,fp16",
    sizes: Annotated[str, typer.Option(help="Comma-separated square input sizes.")] = "512,768,1024,1280",
    keypoints: Annotated[str, typer.Option(help="Comma-separated output keypoint counts.")] = (
        "512,1024,1536,2048,2560,3072,3584,4096"
    ),
    compile_modes: Annotated[str, typer.Option(help="Comma-separated torch.compile modes.")] = "default",
    ort_cuda_fp16: Annotated[
        bool,
        typer.Option(
            "--ort-cuda-fp16/--skip-ort-cuda-fp16",
            help="Run converted FP16 models on ORT CUDA despite the documented ORT 1.27 regression.",
        ),
    ] = False,
    warmup: Annotated[int, typer.Option(min=0)] = 10,
    max_pairs: Annotated[int, typer.Option(min=0)] = 0,
    preprocessing: Annotated[PreprocessingBackend, typer.Option()] = "opencv",
    prefetch: Annotated[bool, typer.Option("--prefetch/--no-prefetch")] = False,
) -> None:
    """Export dynamic models and run the requested benchmark matrix in isolated processes."""
    selected_backends = [item for item in backends.split(",") if item]
    valid_backends = {"pytorch", "torch-compile", "ort-cuda", "ort-tensorrt", "tensorrt"}
    if not selected_backends or any(item not in valid_backends for item in selected_backends):
        raise typer.BadParameter(f"backends must come from {sorted(valid_backends)}")
    if preprocessing == "cuda" and any(backend not in {"ort-cuda", "ort-tensorrt"} for backend in selected_backends):
        raise typer.BadParameter("CUDA preprocessing matrix runs only support ORT CUDA and ORT TensorRT")
    selected_precisions = [item for item in precisions.split(",") if item]
    if not selected_precisions or any(item not in {"fp32", "fp16"} for item in selected_precisions):
        raise typer.BadParameter("precisions must contain fp32 and/or fp16")
    selected_sizes = _parse_csv_ints(sizes, "sizes")
    selected_keypoints = _parse_csv_ints(keypoints, "keypoints")
    selected_compile_modes = [item for item in compile_modes.split(",") if item]
    valid_compile_modes = {"default", "reduce-overhead", "max-autotune-no-cudagraphs", "max-autotune"}
    if not selected_compile_modes or any(item not in valid_compile_modes for item in selected_compile_modes):
        raise typer.BadParameter(f"compile-modes must come from {sorted(valid_compile_modes)}")

    results_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)
    failures_path = results_root / "failures.jsonl"
    needs_onnx = any(backend.startswith("ort-") or backend == "tensorrt" for backend in selected_backends)
    needs_ort_cuda_fp16 = ort_cuda_fp16 and "ort-cuda" in selected_backends and "fp16" in selected_precisions

    for num_keypoints in selected_keypoints:
        model = models_root / f"raco-aliked-lightglue-k{num_keypoints}-dynamic.onnx"
        fp16_model = model.with_suffix(".fp16.onnx")
        if needs_onnx and (not model.exists() or (needs_ort_cuda_fp16 and not fp16_model.exists())):
            export_command = [
                sys.executable,
                "-m",
                "lightglue_dynamo.cli",
                "export",
                "raco_aliked",
                "--output",
                str(model),
                "--batch-size",
                "0",
                "--height",
                "0",
                "--width",
                "0",
                "--num-keypoints",
                str(num_keypoints),
            ]
            if needs_ort_cuda_fp16:
                export_command.append("--fp16")
            typer.echo(f"Exporting dynamic K={num_keypoints} model")
            completed = _run_logged(export_command, results_root / "logs" / f"export-k{num_keypoints}.log")
            if completed.returncode:
                failure = {"stage": "export", "num_keypoints": num_keypoints, "returncode": completed.returncode}
                with failures_path.open("a") as stream:
                    stream.write(json.dumps(failure) + "\n")
                typer.echo(f"Export failed for K={num_keypoints}; recorded in {failures_path}", err=True)
                continue

        for backend in selected_backends:
            modes: list[str | None] = selected_compile_modes if backend == "torch-compile" else [None]
            for precision in selected_precisions:
                if backend == "ort-cuda" and precision == "fp16" and not ort_cuda_fp16:
                    continue
                if backend in {"ort-tensorrt", "tensorrt"} and num_keypoints > 3840:
                    failure = {
                        "stage": "benchmark",
                        "backend": backend,
                        "precision": precision,
                        "num_keypoints": num_keypoints,
                        "status": "unsupported",
                        "reason": "TensorRT TopK limit is 3840",
                    }
                    with failures_path.open("a") as stream:
                        stream.write(json.dumps(failure) + "\n")
                    continue
                selected_model = fp16_model if backend == "ort-cuda" and precision == "fp16" else model
                for size in selected_sizes:
                    for compile_mode in modes:
                        suffix = f"-{compile_mode}" if compile_mode is not None else ""
                        preprocessing_suffix = (
                            ""
                            if preprocessing == "opencv" and not prefetch
                            else f"-prep-{preprocessing}{'-prefetch' if prefetch else ''}"
                        )
                        stem = f"{backend}-{precision}-s{size}-k{num_keypoints}{suffix}{preprocessing_suffix}"
                        output = results_root / f"{stem}.jsonl"
                        summary_path = output.with_suffix(".summary.json")
                        expected_pairs = max_pairs or 1500
                        if (
                            summary_path.exists()
                            and json.loads(summary_path.read_text()).get("completed_pairs") == expected_pairs
                        ):
                            typer.echo(f"Skipping completed {stem}")
                            continue
                        command = [
                            sys.executable,
                            "-m",
                            "lightglue_dynamo.scripts.benchmark",
                            "run",
                            "--backend",
                            backend,
                            "--output",
                            str(output),
                            "--size",
                            str(size),
                            "--num-keypoints",
                            str(num_keypoints),
                            "--precision",
                            precision,
                            "--warmup",
                            str(warmup),
                            "--max-pairs",
                            str(max_pairs),
                            "--cache-root",
                            str(cache_root),
                            "--preprocessing",
                            preprocessing,
                        ]
                        command.append("--prefetch" if prefetch else "--no-prefetch")
                        if backend == "torch-compile" and compile_mode is not None:
                            command.extend(["--compile-mode", compile_mode])
                        if backend.startswith("ort-") or backend == "tensorrt":
                            command.extend(["--model", str(selected_model)])
                        typer.echo(f"Running {stem}")
                        completed = _run_logged(command, results_root / "logs" / f"{stem}.log")
                        if completed.returncode:
                            failure = {
                                "stage": "benchmark",
                                "backend": backend,
                                "precision": precision,
                                "size": size,
                                "num_keypoints": num_keypoints,
                                "compile_mode": compile_mode,
                                "returncode": completed.returncode,
                            }
                            with failures_path.open("a") as stream:
                                stream.write(json.dumps(failure) + "\n")
                            typer.echo(f"Failed {stem}; continuing", err=True)


@app.command("profile-tensorrt-engine")
def profile_tensorrt_engine(
    engine: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option(dir_okay=False)] = Path("data/benchmark_results/profile/tensorrt-layers.json"),
    dataset_root: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path("megadepth_test_1500"),
    scene_info_root: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "megadepth_test_1500/scene_info"
    ),
    size: Annotated[int, typer.Option(min=32)] = 1024,
    pair_index: Annotated[int, typer.Option(min=0, max=1499)] = 0,
    warmup: Annotated[int, typer.Option(min=0)] = 10,
    runs: Annotated[int, typer.Option(min=1)] = 20,
) -> None:
    """Collect per-layer timing from a cached TensorRT or ORT-TRT engine."""
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    preload_nvidia_libraries(tensorrt=True)
    import tensorrt as trt
    from polygraphy.backend.common import BytesFromPath
    from polygraphy.backend.trt import EngineFromBytes, TrtRunner

    class LayerProfiler(trt.IProfiler):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.samples: dict[str, list[float]] = defaultdict(list)

        def report_layer_time(self, layer_name: str, time_ms: float) -> None:
            self.samples[layer_name].append(time_ms)

    manifest = list(iter_pairs(dataset_root, scene_info_root))
    images, _, _ = _read_images(manifest[pair_index], size)
    runner = TrtRunner(EngineFromBytes(BytesFromPath(str(engine))))
    runner.activate()
    try:
        for _ in range(warmup):
            runner.infer({"images": images}, copy_outputs_to_host=False)
        profiler = LayerProfiler()
        runner.context.profiler = profiler
        for _ in range(runs):
            runner.infer({"images": images}, copy_outputs_to_host=False)
        layers = [
            {
                "name": name,
                "mean_ms": float(np.mean(samples)),
                "p95_ms": float(np.percentile(samples, 95)),
                "calls": len(samples),
            }
            for name, samples in profiler.samples.items()
        ]
        layers.sort(key=lambda layer: cast(float, layer["mean_ms"]), reverse=True)
        reformat_layers = [layer for layer in layers if "Reformat" in cast(str, layer["name"])]
        report = {
            "engine": str(engine.resolve()),
            "engine_bytes": engine.stat().st_size,
            "size": size,
            "pair_index": pair_index,
            "warmup": warmup,
            "runs": runs,
            "engine_layers": runner.engine.num_layers,
            "profiling_verbosity": str(runner.engine.profiling_verbosity),
            "reported_layers": len(layers),
            "summed_mean_layer_ms": sum(cast(float, layer["mean_ms"]) for layer in layers),
            "reformat_layer_count": len(reformat_layers),
            "summed_mean_reformat_ms": sum(cast(float, layer["mean_ms"]) for layer in reformat_layers),
            "layers": layers,
        }
    finally:
        runner.deactivate()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    typer.echo(json.dumps({key: value for key, value in report.items() if key != "layers"}, indent=2))
    typer.echo(f"Wrote {output}")


@app.command("profile-pytorch")
def profile_pytorch(
    output: Annotated[Path, typer.Option(dir_okay=False)] = Path("data/benchmark_results/profile/pytorch.json"),
    table_output: Annotated[Path, typer.Option(dir_okay=False)] = Path("data/benchmark_results/profile/pytorch.txt"),
    dataset_root: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path("megadepth_test_1500"),
    scene_info_root: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "megadepth_test_1500/scene_info"
    ),
    size: Annotated[int, typer.Option(min=32)] = 1024,
    num_keypoints: Annotated[int, typer.Option(min=128)] = 1024,
    precision: Annotated[Precision, typer.Option()] = "fp32",
    warmup: Annotated[int, typer.Option(min=0)] = 5,
    pairs: Annotated[int, typer.Option(min=1)] = 10,
) -> None:
    """Capture a component-labelled PyTorch CUDA trace after baseline timing."""
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    import torch

    from lightglue_dynamo.config import Extractor
    from lightglue_dynamo.models import LightGlue, RaCoALIKED

    torch.set_float32_matmul_precision("highest")
    extractor = RaCoALIKED(num_keypoints=num_keypoints).eval().cuda()
    matcher = LightGlue(**Extractor.raco_aliked.lightglue_config).eval().cuda()
    manifest = list(iter_pairs(dataset_root, scene_info_root))
    inputs = [_read_images(pair, size)[0] for pair in manifest[: max(warmup, pairs)]]
    autocast_dtype = torch.float16 if precision == "fp16" else torch.float32

    def operation(images: np.ndarray) -> None:
        tensor = torch.from_numpy(images).cuda()
        with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype, enabled=precision == "fp16"):
            with torch.profiler.record_function("raco_detector_ranker"):
                keypoints, _detection_scores, ranker_scores = extractor.raco(tensor)
            with torch.profiler.record_function("aliked_descriptor_sampling"):
                descriptors = extractor.aliked(tensor, keypoints)
            size_tensor = keypoints.new_tensor([tensor.shape[-1], tensor.shape[-2]])
            normalized = (keypoints - size_tensor / 2) / (size_tensor.max() / 2)
            with torch.profiler.record_function("lightglue_plus"):
                matcher(normalized, descriptors)
            del ranker_scores

    for index in range(warmup):
        operation(inputs[index % len(inputs)])
    torch.cuda.synchronize()
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities, record_shapes=True, profile_memory=True, with_stack=True
    ) as profiler:
        for index in range(pairs):
            operation(inputs[index % len(inputs)])
            torch.cuda.synchronize()
    output.parent.mkdir(parents=True, exist_ok=True)
    table_output.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(output))
    table = profiler.key_averages(group_by_input_shape=True).table(
        sort_by="self_cuda_time_total", row_limit=100, max_name_column_width=100
    )
    table_output.write_text(table + "\n")
    typer.echo(table)
    typer.echo(f"Wrote {output} and {table_output}")


if __name__ == "__main__":
    app()
