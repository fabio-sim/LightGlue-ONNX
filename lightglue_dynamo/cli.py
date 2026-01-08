from pathlib import Path
from typing import Annotated, cast

import cv2
import typer

from lightglue_dynamo.cli_utils import check_multiple_of
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
    fuse_multi_head_attention: Annotated[
        bool,
        typer.Option(
            "--fuse-multi-head-attention",
            help="Fuse multi-head attention subgraph into one optimized operation. (ONNX Runtime-only).",
        ),
    ] = False,
    dynamo_export: Annotated[
        bool,
        typer.Option(
            "--dynamo-export/--legacy-export", help="Use the TorchDynamo ONNX exporter. Legacy export uses TorchScript."
        ),
    ] = False,
    opset: Annotated[int, typer.Option(min=16, max=20, help="ONNX opset version of exported model.")] = 18,
    fp16: Annotated[bool, typer.Option("--fp16", help="Whether to also convert to FP16.")] = False,
) -> None:
    """Export LightGlue to ONNX."""
    import onnx
    import torch
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
    from onnxruntime.transformers.float16 import convert_float_to_float16

    from lightglue_dynamo.models import DISK, LightGlue, Pipeline, SuperPoint
    from lightglue_dynamo.ops import use_fused_multi_head_attention

    match extractor_type:
        case Extractor.superpoint:
            extractor = SuperPoint(num_keypoints=num_keypoints)
        case Extractor.disk:
            extractor = DISK(num_keypoints=num_keypoints)
    matcher = LightGlue(**extractor_type.lightglue_config)
    pipeline = Pipeline(extractor, matcher).eval()

    if output is None:
        output = Path(f"weights/{extractor_type}_lightglue_pipeline.onnx")

    check_multiple_of(batch_size, 2)
    check_multiple_of(height, extractor_type.input_dim_divisor)
    check_multiple_of(width, extractor_type.input_dim_divisor)

    if height > 0 and width > 0 and num_keypoints > height * width:
        raise typer.BadParameter("num_keypoints cannot be greater than height * width.")

    if fuse_multi_head_attention:
        typer.echo(
            "Warning: Multi-head attention nodes will be fused. Exported model will only work with ONNX Runtime CPU & CUDA execution providers."
        )
        torch_version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:3])
        if torch_version < (2, 4):
            raise typer.Abort("Fused multi-head attention requires PyTorch 2.4 or later.")
        use_fused_multi_head_attention()
        if dynamo_export:
            typer.echo(
                "Warning: Fused multi-head attention is not supported by the Dynamo exporter. Using legacy export."
            )

    def build_dynamic_config(
        use_dynamo: bool,
    ) -> tuple[dict[str, dict[int, str]] | None, tuple[dict[int, str], ...] | None]:
        dynamic_axes: dict[str, dict[int, str]] | None = None
        dynamic_shapes: tuple[dict[int, str], ...] | None = None
        if use_dynamo:
            image_shapes: dict[int, str] = {}
            if batch_size == 0:
                image_shapes[0] = "batch_size"
            if height == 0:
                image_shapes[2] = "height"
            if width == 0:
                image_shapes[3] = "width"
            if image_shapes:
                dynamic_shapes = (image_shapes,)
        else:
            dynamic_axes = {"matches": {0: "num_matches"}, "mscores": {0: "num_matches"}}
            dynamic_axes["images"] = {}
            dynamic_axes["keypoints"] = {}
            if batch_size == 0:
                dynamic_axes["images"][0] = "batch_size"
                dynamic_axes["keypoints"][0] = "batch_size"
            if height == 0:
                dynamic_axes["images"][2] = "height"
            if width == 0:
                dynamic_axes["images"][3] = "width"
        return dynamic_axes, dynamic_shapes

    def export_model(use_dynamo: bool) -> None:
        dynamic_axes, dynamic_shapes = build_dynamic_config(use_dynamo)
        inputs = (torch.zeros(batch_size or 2, extractor_type.input_channels, height or 256, width or 256),)
        torch.onnx.export(
            pipeline,
            inputs,
            str(output),
            input_names=["images"],
            output_names=["keypoints", "matches", "mscores"],
            opset_version=opset,
            dynamic_axes=dynamic_axes,
            dynamic_shapes=dynamic_shapes,
            dynamo=use_dynamo,
        )

    use_dynamo = dynamo_export and not fuse_multi_head_attention
    try:
        export_model(use_dynamo)
        onnx.checker.check_model(output)
    except Exception as exc:
        if not use_dynamo:
            raise
        typer.echo(
            f"Warning: Dynamo exporter failed ({exc}). Falling back to legacy exporter. "
            "Use --legacy-export to skip Dynamo."
        )
        export_model(False)
        onnx.checker.check_model(output)
    try:
        inferred = SymbolicShapeInference.infer_shapes(onnx.load_model(output), auto_merge=True)  # type: ignore
        onnx.save_model(inferred, output)
    except Exception as exc:
        typer.echo(f"Warning: Symbolic shape inference failed ({exc}). Falling back to onnx.shape_inference.")
        try:
            inferred = onnx.shape_inference.infer_shapes(onnx.load_model(output))
            onnx.save_model(inferred, output)
        except Exception as fallback_exc:
            typer.echo(f"Warning: onnx.shape_inference failed ({fallback_exc}). Skipping.")
    typer.echo(f"Successfully exported model to {output}")
    if fp16:
        typer.echo(
            "Converting to FP16. Warning: This FP16 model should NOT be used for TensorRT. TRT provides its own fp16 option."
        )
        onnx.save_model(convert_float_to_float16(onnx.load_model(output)), output.with_suffix(".fp16.onnx"))


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
        typer.Option(
            "-o",
            "--output",
            dir_okay=False,
            writable=True,
            help="Path to save output matches figure. If not given, show visualization.",
        ),
    ] = None,
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
) -> None:
    """Run inference for LightGlue ONNX model."""
    import numpy as np
    import onnxruntime as ort

    from lightglue_dynamo import viz
    from lightglue_dynamo.preprocessors import DISKPreprocessor, SuperPointPreprocessor

    def _load_and_resize(path: Path) -> np.ndarray:
        image = cv2.imread(str(path))
        if image is None:
            raise typer.BadParameter(f"Failed to read image: {path}")
        return cv2.resize(cast(np.ndarray, image), (width, height))

    raw_images = [_load_and_resize(left_image_path), _load_and_resize(right_image_path)]
    images = np.stack(raw_images)
    match extractor_type:
        case Extractor.superpoint:
            images = SuperPointPreprocessor.preprocess(images)
        case Extractor.disk:
            images = DISKPreprocessor.preprocess(images)
    images = images.astype(np.float16 if fp16 and device != InferenceDevice.tensorrt else np.float32)

    if device in {InferenceDevice.cuda, InferenceDevice.tensorrt}:
        preload = getattr(ort, "preload_dlls", None)
        if callable(preload):
            preload()

    session_options = ort.SessionOptions()  # type: ignore[possibly-missing-attribute]
    session_options.enable_profiling = profile
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

    input_shape = session.get_inputs()[0].shape
    if len(input_shape) == 4:
        channel_dim = input_shape[1]
        height_dim = input_shape[2]
        width_dim = input_shape[3]
        if isinstance(channel_dim, int) and channel_dim != images.shape[1]:
            raise typer.BadParameter(
                f"Model expects {channel_dim} channels but got {images.shape[1]} from preprocessing."
            )
        if isinstance(height_dim, int) and height_dim != height:
            raise typer.BadParameter(f"Model expects height={height_dim} but got {height}.")
        if isinstance(width_dim, int) and width_dim != width:
            raise typer.BadParameter(f"Model expects width={width_dim} but got {width}.")

    for _ in range(100 if profile else 1):
        outputs = cast(list[np.ndarray], session.run(None, {"images": images}))
        keypoints, matches, _mscores = outputs[0], outputs[1], outputs[2]

    viz.plot_images(raw_images)
    viz.plot_matches(keypoints[0][matches[..., 1]], keypoints[1][matches[..., 2]], color="lime", lw=0.2)
    if output_path is None:
        viz.plt.show()
    else:
        viz.save_plot(output_path)


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
        typer.Option(
            "-o",
            "--output",
            dir_okay=False,
            writable=True,
            help="Path to save output matches figure. If not given, show visualization.",
        ),
    ] = None,
    height: Annotated[
        int, typer.Option("-h", "--height", min=1, help="Height of input image at which to perform inference.")
    ] = 1024,
    width: Annotated[
        int, typer.Option("-w", "--width", min=1, help="Width of input image at which to perform inference.")
    ] = 1024,
    fp16: Annotated[bool, typer.Option("--fp16", help="Whether model uses FP16 precision.")] = False,
    profile: Annotated[bool, typer.Option("--profile", help="Whether to profile model execution.")] = False,
) -> None:
    """Run pure TensorRT inference for LightGlue model using Polygraphy (requires TensorRT to be installed)."""
    import site

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
    from lightglue_dynamo.preprocessors import DISKPreprocessor, SuperPointPreprocessor

    def _load_and_resize(path: Path) -> np.ndarray:
        image = cv2.imread(str(path))
        if image is None:
            raise typer.BadParameter(f"Failed to read image: {path}")
        return cv2.resize(cast(np.ndarray, image), (width, height))

    raw_images = [_load_and_resize(left_image_path), _load_and_resize(right_image_path)]
    images = np.stack(raw_images)
    match extractor_type:
        case Extractor.superpoint:
            images = SuperPointPreprocessor.preprocess(images)
        case Extractor.disk:
            images = DISKPreprocessor.preprocess(images)
    images = images.astype(np.float32)

    # Build TensorRT engine
    if model_path.suffix == ".engine":
        build_engine = EngineFromBytes(BytesFromPath(str(model_path)))
    else:  # .onnx
        build_engine = EngineFromNetwork(NetworkFromOnnxPath(str(model_path)), config=CreateConfig(fp16=fp16))
        build_engine = SaveEngine(build_engine, str(model_path.with_suffix(".engine")))

    def _print_cuda_runtime_hint() -> None:
        site_paths = [path for path in [*site.getsitepackages(), site.getusersitepackages()] if path]
        candidates: list[Path] = []
        for path in site_paths:
            base = Path(path)
            trt_libs = base / "tensorrt_libs"
            cuda_runtime = base / "nvidia" / "cuda_runtime" / "lib"
            if trt_libs.exists():
                candidates.append(trt_libs)
            if cuda_runtime.exists():
                candidates.append(cuda_runtime)
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
            for _ in range(10 if profile else 1):  # Warm-up if profiling
                outputs = runner.infer(feed_dict={"images": images})
                keypoints, matches, mscores = outputs["keypoints"], outputs["matches"], outputs["mscores"]  # noqa: F841

            if profile:
                typer.echo(f"Inference Time: {runner.last_inference_time():.3f} s")
    except OSError as exc:
        if "libcudart" in str(exc):
            _print_cuda_runtime_hint()
        raise

    viz.plot_images(raw_images)
    viz.plot_matches(keypoints[0][matches[..., 1]], keypoints[1][matches[..., 2]], color="lime", lw=0.2)
    if output_path is None:
        viz.plt.show()
    else:
        viz.save_plot(output_path)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
