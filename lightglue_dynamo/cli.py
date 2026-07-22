from math import ceil, sqrt
from pathlib import Path
from typing import Annotated, Literal, cast

import cv2
import typer

from lightglue_dynamo.cli_utils import check_multiple_of, preload_nvidia_libraries
from lightglue_dynamo.config import Extractor, InferenceDevice

app = typer.Typer()


@app.callback()
def callback() -> None:
    """LightGlue Dynamo CLI."""


@app.command()
def export(
    extractor_type: Annotated[Extractor, typer.Argument()] = Extractor.superpoint,
    output: Annotated[
        Path | None,  # typer does not support Path | None
        typer.Option("-o", "--output", dir_okay=False, writable=True, help="Path to save exported model."),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(
            "-b", "--batch-size", min=0, help="Batch size of exported ONNX model. Set to 0 to mark as dynamic."
        ),
    ] = 0,
    height: Annotated[
        int, typer.Option("-h", "--height", min=0, help="Height of input image. Set to 0 to mark as dynamic.")
    ] = 0,
    width: Annotated[
        int, typer.Option("-w", "--width", min=0, help="Width of input image. Set to 0 to mark as dynamic.")
    ] = 0,
    num_keypoints: Annotated[
        int, typer.Option(min=128, help="Number of keypoints outputted by feature extractor.")
    ] = 1024,
    opset: Annotated[int, typer.Option(min=16, max=20, help="ONNX opset version of exported model.")] = 20,
    portable_deform_conv: Annotated[
        bool,
        typer.Option(
            "--portable-deform-conv/--onnx-deform-conv",
            help="Decompose ALIKED DeformConv for TensorRT and WebGPU portability.",
        ),
    ] = True,
    fp16: Annotated[bool, typer.Option("--fp16", help="Whether to also convert to FP16.")] = False,
) -> None:
    """Export LightGlue to ONNX."""
    import onnx
    import torch
    from onnxruntime.transformers.float16 import convert_float_to_float16
    from onnxscript import opset20 as onnx_op

    from lightglue_dynamo.models import DISK, LightGlue, Pipeline, RaCoALIKED, SuperPoint

    match extractor_type:
        case Extractor.superpoint:
            extractor = SuperPoint(num_keypoints=num_keypoints)
        case Extractor.disk:
            extractor = DISK(num_keypoints=num_keypoints)
        case Extractor.raco_aliked:
            if opset < 20:
                raise typer.BadParameter("raco_aliked export requires ONNX opset 20.")
            extractor = RaCoALIKED(num_keypoints=num_keypoints, portable_deform_conv=portable_deform_conv)
    matcher = LightGlue(**extractor_type.lightglue_config)
    pipeline = Pipeline(extractor, matcher).eval()
    pipeline.fuse_batch_norm()

    if output is None:
        output = Path(f"weights/{extractor_type}_lightglue_pipeline.onnx")

    output_names = ["keypoints", "matches", "mscores"]

    check_multiple_of(batch_size, 2)
    check_multiple_of(height, extractor_type.input_dim_divisor)
    check_multiple_of(width, extractor_type.input_dim_divisor)

    num_candidates = extractor_type.keypoint_candidate_count(num_keypoints)
    if height > 0 and width > 0 and num_candidates > height * width:
        raise typer.BadParameter(
            f"The extractor requires {num_candidates} candidate locations, more than the {height * width} pixels."
        )

    def build_dynamic_shapes() -> tuple[dict[int, object], ...] | None:
        image_shapes: dict[int, object] = {}
        divisor = extractor_type.input_dim_divisor
        square_factor = max(2, ceil(sqrt(num_candidates) / divisor))
        if batch_size == 0:
            pair_count = torch.export.Dim("pair_count", min=1)
            image_shapes[0] = 2 * pair_count
        if height == 0:
            minimum_height_factor = max(2, ceil(num_candidates / width / divisor)) if width else square_factor
            height_factor = torch.export.Dim("height_factor", min=minimum_height_factor)
            image_shapes[2] = divisor * height_factor
        if width == 0:
            minimum_width_factor = max(2, ceil(num_candidates / height / divisor)) if height else square_factor
            width_factor = torch.export.Dim("width_factor", min=minimum_width_factor)
            image_shapes[3] = divisor * width_factor
        return (image_shapes,) if image_shapes else None

    def export_model() -> None:
        def translate_integer_div(self: object, other: object, rounding_mode: str | None = None) -> object:
            if rounding_mode not in {"floor", "trunc"}:
                raise ValueError(f"Unsupported integer division mode: {rounding_mode}")
            # The default ONNX decomposition casts the quotient through float. TensorRT
            # then lowers that cast to FP16, overflowing flattened image indices above
            # 65504. These operands are non-negative integers, so ONNX integer Div is
            # exactly equivalent to both supported rounding modes without a float cast.
            return onnx_op.Div(self, other)

        example_batch = batch_size or 4
        divisor = extractor_type.input_dim_divisor
        dynamic_side = max(2 * divisor, ceil(sqrt(num_candidates) / divisor) * divisor)
        if height == 0 and width > 0:
            example_height = max(2 * divisor, ceil(num_candidates / width / divisor) * divisor)
        else:
            example_height = height or dynamic_side
        if width == 0 and height > 0:
            example_width = max(2 * divisor, ceil(num_candidates / height / divisor) * divisor)
        else:
            example_width = width or dynamic_side
        inputs = (torch.zeros(example_batch, extractor_type.input_channels, example_height, example_width),)
        torch.onnx.export(
            pipeline,
            inputs,
            str(output),
            input_names=["images"],
            output_names=output_names,
            opset_version=opset,
            dynamic_shapes=build_dynamic_shapes(),
            dynamo=True,
            external_data=False,
            optimize=False,
            custom_translation_table={torch.ops.aten.div.Tensor_mode: translate_integer_div},
        )

    export_model()
    onnx.checker.check_model(output)
    typer.echo(f"Successfully exported model to {output}")
    if fp16:
        typer.echo(
            "Converting to FP16. Warning: This FP16 model should NOT be used for TensorRT. TRT provides its own fp16 option."
        )
        from onnxruntime.transformers.onnx_model import OnnxModel

        fp16_model = convert_float_to_float16(onnx.load_model(output))
        # The ORT converter can append precision-boundary Cast nodes after their
        # consumers and leave stale FP16 value_info on blocked FP32 constants.
        # Restore a valid topological order, then rebuild intermediate type/shape
        # annotations from operator semantics before saving the converted graph.
        OnnxModel(fp16_model).topological_sort(is_deterministic=True)
        del fp16_model.graph.value_info[:]
        fp16_model = onnx.shape_inference.infer_shapes(fp16_model, strict_mode=True, data_prop=True)
        onnx.checker.check_model(fp16_model, full_check=True)
        onnx.save_model(fp16_model, output.with_suffix(".fp16.onnx"), save_as_external_data=False)


@app.command()
def infer(
    model_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Path to ONNX model.")],
    left_image_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Path to first image.")
    ],
    right_image_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Path to second image.")
    ],
    extractor_type: Annotated[Extractor, typer.Argument()] = Extractor.superpoint,
    output_path: Annotated[
        Path | None,
        typer.Option("-o", "--output", dir_okay=False, writable=True, help="Path to save output matches figure."),
    ] = None,
    show: Annotated[bool, typer.Option("--show/--no-show", help="Show the match visualization window.")] = False,
    height: Annotated[
        int, typer.Option("-h", "--height", min=1, help="Height of input image at which to perform inference.")
    ] = 1024,
    width: Annotated[
        int, typer.Option("-w", "--width", min=1, help="Width of input image at which to perform inference.")
    ] = 1024,
    device: Annotated[
        InferenceDevice, typer.Option("-d", "--device", help="Device to run inference on.")
    ] = InferenceDevice.cuda,
    fp16: Annotated[bool, typer.Option("--fp16", help="Whether model uses FP16 precision.")] = False,
    profile: Annotated[bool, typer.Option("--profile", help="Whether to profile model execution.")] = False,
    preprocessing: Annotated[
        Literal["opencv", "cuda"],
        typer.Option(help="Image preprocessing backend. CUDA currently supports RaCo JPEG inputs on NVIDIA GPUs."),
    ] = "opencv",
) -> None:
    """Run inference for LightGlue ONNX model."""
    import time

    import numpy as np

    # A GPU wheel can still be used with the CPU EP; preload is harmless for a
    # CPU-only wheel and avoids failing the ORT import in a mixed environment.
    preload_nvidia_libraries(tensorrt=device == InferenceDevice.tensorrt)

    import onnxruntime as ort

    from lightglue_dynamo import viz
    from lightglue_dynamo.preprocessors import (
        CudaPreparedImages,
        CudaRaCoPreprocessor,
        DISKPreprocessor,
        RaCoPreprocessor,
        SuperPointPreprocessor,
        prepare_host_images,
    )

    if preprocessing == "cuda" and (
        extractor_type != Extractor.raco_aliked or device not in {InferenceDevice.cuda, InferenceDevice.tensorrt}
    ):
        raise typer.BadParameter("CUDA preprocessing requires the RaCo-ALIKED extractor and CUDA or TensorRT device")

    host_prepared = None
    images: np.ndarray | CudaPreparedImages
    if preprocessing == "opencv":
        preprocessor = {
            Extractor.superpoint: SuperPointPreprocessor,
            Extractor.disk: DISKPreprocessor,
            Extractor.raco_aliked: RaCoPreprocessor,
        }[extractor_type]
        try:
            host_prepared = prepare_host_images(
                (left_image_path, right_image_path),
                width,
                height,
                preprocessor,
                dtype=np.float16 if fp16 and device != InferenceDevice.tensorrt else np.float32,
            )
        except FileNotFoundError as exc:
            raise typer.BadParameter(str(exc)) from exc
        images = host_prepared.images

    if device in {InferenceDevice.cuda, InferenceDevice.tensorrt}:
        preload = getattr(ort, "preload_dlls", None)
        if callable(preload):
            preload()

    session_options = ort.SessionOptions()  # type: ignore[possibly-missing-attribute]
    session_options.enable_profiling = profile
    if fp16 and device == InferenceDevice.cuda:
        # ORT 1.27's level-3 CastFloat16/layout pass aborts on converted
        # dynamic graphs. Extended optimization is the strongest safe level.
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    # session_options.optimized_model_filepath = "weights/ort_optimized.onnx"

    providers: list[tuple[str, dict[str, object]]] = [("CPUExecutionProvider", {})]
    if device == InferenceDevice.cuda:
        providers.insert(0, ("CUDAExecutionProvider", {}))
    elif device == InferenceDevice.tensorrt:
        providers.insert(0, ("CUDAExecutionProvider", {}))
        providers.insert(
            0,
            (
                "TensorrtExecutionProvider",
                {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "weights/.trtcache_engines",
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": "weights/.trtcache_timings",
                    "trt_fp16_enable": fp16,
                },
            ),
        )
    elif device == InferenceDevice.openvino:
        providers.insert(0, ("OpenVINOExecutionProvider", {}))

    available_providers = set(ort.get_available_providers())  # type: ignore[possibly-missing-attribute]
    selected = [provider for provider in providers if provider[0] in available_providers]
    if not selected:
        typer.echo("Warning: Requested providers unavailable. Falling back to CPUExecutionProvider.")
        selected = [("CPUExecutionProvider", {})]

    try:
        session = ort.InferenceSession(model_path, session_options, selected)
    except Exception as exc:
        if device == InferenceDevice.cuda:
            typer.echo(f"Warning: CUDA provider failed ({exc}). Falling back to CPUExecutionProvider.")
            session = ort.InferenceSession(model_path, session_options, [("CPUExecutionProvider", {})])
        elif device == InferenceDevice.tensorrt:
            typer.echo(f"Warning: TensorRT provider failed ({exc}). Falling back to CUDAExecutionProvider.")
            session = ort.InferenceSession(model_path, session_options, [("CUDAExecutionProvider", {})])
        else:
            raise

    cuda_preprocessor: CudaRaCoPreprocessor | None = None
    cuda_prepared: CudaPreparedImages | None = None
    preprocessing_ms = host_prepared.total_ms if host_prepared is not None else None
    try:
        if preprocessing == "cuda":
            active_providers = set(session.get_providers())
            if not active_providers.intersection({"CUDAExecutionProvider", "TensorrtExecutionProvider"}):
                raise typer.BadParameter("CUDA preprocessing cannot be used after falling back to CPU execution")
            input_dtype = "float16" if session.get_inputs()[0].type == "tensor(float16)" else "float32"
            cuda_preprocessor = CudaRaCoPreprocessor(width, height, dtype=input_dtype)
            cuda_prepared = cuda_preprocessor.prepare((left_image_path, right_image_path))
            preprocessing_ms = cuda_prepared.synchronize()
            images = cuda_prepared

        input_shape = session.get_inputs()[0].shape
        prepared_shape = images.shape
        if len(input_shape) == 4:
            channel_dim = input_shape[1]
            height_dim = input_shape[2]
            width_dim = input_shape[3]
            if isinstance(channel_dim, int) and channel_dim != prepared_shape[1]:
                raise typer.BadParameter(
                    f"Model expects {channel_dim} channels but got {prepared_shape[1]} from preprocessing."
                )
            if isinstance(height_dim, int) and height_dim != height:
                raise typer.BadParameter(f"Model expects height={height_dim} but got {height}.")
            if isinstance(width_dim, int) and width_dim != width:
                raise typer.BadParameter(f"Model expects width={width_dim} but got {width}.")

        last_inference_time: float | None = None
        if isinstance(images, CudaPreparedImages):
            binding = session.io_binding()
            binding.bind_ortvalue_input("images", images.to_ort_value(ort))
            for output in session.get_outputs():
                binding.bind_output(output.name, "cuda", 0)
            for _ in range(100 if profile else 1):
                if profile:
                    start = time.perf_counter()
                session.run_with_iobinding(binding)
                binding.synchronize_outputs()
                outputs = cast(list[np.ndarray], binding.copy_outputs_to_cpu())
                if profile:
                    last_inference_time = time.perf_counter() - start
                keypoints, matches, _mscores = outputs
        else:
            for _ in range(100 if profile else 1):
                if profile:
                    start = time.perf_counter()
                outputs = cast(list[np.ndarray], session.run(None, {"images": images}))
                if profile:
                    last_inference_time = time.perf_counter() - start
                keypoints, matches, _mscores = outputs

        match_count = int(matches.shape[0])
        typer.echo(f"Matches: {match_count}")
        if profile and last_inference_time is not None:
            typer.echo(f"Preprocessing Time: {preprocessing_ms / 1000:.6f} s")
            typer.echo(f"Inference Time: {last_inference_time:.6f} s")

        if output_path is not None or show:
            if host_prepared is not None:
                raw_images = host_prepared.resized_bgr
            elif cuda_prepared is not None:
                raw_images = [
                    cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                    for image in cuda_prepared.original_bgr_images()
                ]
            else:
                raise RuntimeError("Missing visualization images")
            viz.plot_images(raw_images)
            viz.plot_matches(keypoints[0][matches[..., 1]], keypoints[1][matches[..., 2]], color="lime", lw=0.2)
            if output_path is not None:
                viz.save_plot(output_path)
            if show:
                viz.plt.show()
    finally:
        if cuda_prepared is not None:
            cuda_prepared.release()
        if cuda_preprocessor is not None:
            cuda_preprocessor.close()


@app.command()
def trtexec(
    model_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Path to ONNX model or built TensorRT engine."),
    ],
    left_image_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Path to first image.")
    ],
    right_image_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Path to second image.")
    ],
    extractor_type: Annotated[Extractor, typer.Argument()] = Extractor.superpoint,
    output_path: Annotated[
        Path | None,
        typer.Option("-o", "--output", dir_okay=False, writable=True, help="Path to save output matches figure."),
    ] = None,
    show: Annotated[bool, typer.Option("--show/--no-show", help="Show the match visualization window.")] = False,
    height: Annotated[
        int, typer.Option("-h", "--height", min=1, help="Height of input image at which to perform inference.")
    ] = 1024,
    width: Annotated[
        int, typer.Option("-w", "--width", min=1, help="Width of input image at which to perform inference.")
    ] = 1024,
    strongly_typed: Annotated[
        bool,
        typer.Option(
            "--strongly-typed/--no-strongly-typed",
            help="Enable TensorRT strongly typed network (recommended for FP8 Q/DQ models).",
        ),
    ] = False,
    fp16: Annotated[bool, typer.Option("--fp16", help="Whether model uses FP16 precision.")] = False,
    precision_constraints: Annotated[
        str, typer.Option("--precision-constraints", help="Precision constraints for TensorRT (none, prefer, obey).")
    ] = "none",
    profile: Annotated[bool, typer.Option("--profile", help="Whether to profile model execution.")] = False,
) -> None:
    """Run pure TensorRT inference for LightGlue model using Polygraphy (requires TensorRT to be installed)."""
    import site

    preload_nvidia_libraries(tensorrt=True)

    import numpy as np
    from polygraphy.backend.common import BytesFromPath
    from polygraphy.backend.trt import (
        CreateConfig,
        EngineFromBytes,
        EngineFromNetwork,
        NetworkFromOnnxPath,
        SaveEngine,
        TrtRunner,
    )

    from lightglue_dynamo import viz
    from lightglue_dynamo.preprocessors import (
        DISKPreprocessor,
        RaCoPreprocessor,
        SuperPointPreprocessor,
        prepare_host_images,
    )

    preprocessor = {
        Extractor.superpoint: SuperPointPreprocessor,
        Extractor.disk: DISKPreprocessor,
        Extractor.raco_aliked: RaCoPreprocessor,
    }[extractor_type]
    try:
        prepared = prepare_host_images(
            (left_image_path, right_image_path), width, height, preprocessor, dtype=np.float32
        )
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    raw_images = prepared.resized_bgr
    images = prepared.images

    if strongly_typed and precision_constraints.lower() != "none":
        raise typer.BadParameter("precision-constraints must be 'none' when --strongly-typed is set.")

    precision_constraints_value = precision_constraints.lower()
    if precision_constraints_value not in {"none", "prefer", "obey"}:
        raise typer.BadParameter("precision-constraints must be one of: none, prefer, obey.")
    if precision_constraints_value == "none":
        precision_constraints_value = None

    # Build TensorRT engine
    if model_path.suffix == ".engine":
        build_engine = EngineFromBytes(BytesFromPath(str(model_path)))
    else:  # .onnx
        build_engine = EngineFromNetwork(
            NetworkFromOnnxPath(str(model_path), strongly_typed=strongly_typed),
            config=CreateConfig(fp16=fp16, precision_constraints=precision_constraints_value),
        )
        build_engine = SaveEngine(build_engine, str(model_path.with_suffix(".engine")))

    def _print_cuda_runtime_hint() -> None:
        site_paths = [path for path in [*site.getsitepackages(), site.getusersitepackages()] if path]
        candidates: list[Path] = []
        for path in site_paths:
            base = Path(path)
            trt_libs = base / "tensorrt_libs"
            cuda_runtime = base / "nvidia" / "cu13" / "lib"
            legacy_cuda_runtime = base / "nvidia" / "cuda_runtime" / "lib"
            if trt_libs.exists():
                candidates.append(trt_libs)
            if cuda_runtime.exists():
                candidates.append(cuda_runtime)
            if legacy_cuda_runtime.exists():
                candidates.append(legacy_cuda_runtime)
        if candidates:
            joined = ":".join(str(path) for path in candidates)
            typer.echo("Hint: add TensorRT + CUDA runtime libs to LD_LIBRARY_PATH, e.g.:")
            typer.echo(f'export LD_LIBRARY_PATH="{joined}:${{LD_LIBRARY_PATH:-}}"')
        else:
            typer.echo(
                "Hint: ensure TensorRT and CUDA runtime libraries (libnvinfer.so, libcudart.so) are on LD_LIBRARY_PATH."
            )

    try:
        with TrtRunner(build_engine) as runner:
            warmup_runs = 10 if profile else 0
            if warmup_runs:
                for _ in range(warmup_runs):
                    outputs = runner.infer(feed_dict={"images": images})
                    keypoints, matches, mscores = outputs["keypoints"], outputs["matches"], outputs["mscores"]

            measured_runs = 100 if profile else 1
            inference_times: list[float] = []
            for _ in range(measured_runs):
                outputs = runner.infer(feed_dict={"images": images})
                keypoints, matches, mscores = outputs["keypoints"], outputs["matches"], outputs["mscores"]  # noqa: F841
                if profile:
                    inference_times.append(runner.last_inference_time())

            match_count = int(matches.shape[0])
            typer.echo(f"Matches: {match_count}")
            if profile:
                median_time = float(np.median(np.asarray(inference_times, dtype=np.float64)))
                typer.echo(f"Inference Time (median over 100 runs, 10 warmup): {median_time:.6f} s")
    except OSError as exc:
        if "libcudart" in str(exc):
            _print_cuda_runtime_hint()
        raise

    if output_path is not None or show:
        viz.plot_images(raw_images)
        viz.plot_matches(keypoints[0][matches[..., 1]], keypoints[1][matches[..., 2]], color="lime", lw=0.2)
        if output_path is not None:
            viz.save_plot(output_path)
        if show:
            viz.plt.show()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
