"""Calibrate and explicitly quantize only RaCo's ranker convolutions to FP8.

This intentionally avoids loading the full matching pipeline during calibration.
Images are streamed through the small ranker on CUDA, and only one scalar
activation maximum per convolution is retained. The emitted standard ONNX Q/DQ
graph is consumed by TensorRT's explicit-quantization path.
"""

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Annotated, Literal

import cv2
import ml_dtypes
import numpy as np
import onnx
import torch
import typer
from onnx import helper, numpy_helper
from torch import nn

from lightglue_dynamo.models.raco import RaCo

app = typer.Typer()

_FP8_MAX = 448.0
_DEFAULT_IMAGES = (
    Path("assets/sacre_coeur1.jpg"),
    Path("assets/sacre_coeur2.jpg"),
    Path("assets/DSC_0410.JPG"),
    Path("assets/DSC_0411.JPG"),
)


class _Ranker(nn.Module):
    """Keep the exported initializer names aligned with the full pipeline."""

    def __init__(self, ranker_head: nn.Module) -> None:
        super().__init__()
        self.ranker_head = ranker_head

    def forward(self, normalized_images: torch.Tensor) -> torch.Tensor:
        return self.ranker_head(normalized_images)


def _build_ranker(device: torch.device) -> _Ranker:
    raco = RaCo().eval()
    raco.fuse_batch_norm()
    return _Ranker(raco.ranker_head).eval().to(device)


def _load_image(path: Path, height: int, width: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / np.float32(255)


def _batches(items: list[Path], batch_size: int) -> Iterable[list[Path]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _calibrate_activation_maxima(
    ranker: _Ranker, image_paths: list[Path], height: int, width: int, batch_size: int
) -> tuple[dict[str, float], float]:
    device = next(ranker.parameters()).device
    if device.type != "cuda":
        raise ValueError("Ranker FP8 calibration requires CUDA to bound host-memory use.")

    maxima: dict[str, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def capture(name: str) -> Callable[[nn.Module, tuple[torch.Tensor, ...]], None]:
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            value = inputs[0].detach().abs().amax()
            maxima[name] = value if name not in maxima else torch.maximum(maxima[name], value)

        return hook

    for name, module in ranker.named_modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_pre_hook(capture(name)))

    mean = torch.tensor((0.485, 0.456, 0.406), device=device).reshape(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).reshape(1, 3, 1, 1)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        with torch.inference_mode():
            for paths in _batches(image_paths, batch_size):
                host = np.stack([_load_image(path, height, width) for path in paths])
                images = torch.from_numpy(host).to(device=device, non_blocking=False)
                ranker((images - mean) / std)
                del images, host
        torch.cuda.synchronize(device)
        peak_mib = torch.cuda.max_memory_allocated(device) / 1024**2
        return {name: float(value.cpu()) for name, value in maxima.items()}, peak_mib
    finally:
        for handle in handles:
            handle.remove()


def _ranker_module_name(weight_name: str) -> str | None:
    marker = "ranker_head."
    offset = weight_name.find(marker)
    if offset < 0:
        return None
    module_name = weight_name[offset:]
    return module_name.removesuffix(".weight")


def _initializer_map(model: onnx.ModelProto) -> dict[str, onnx.TensorProto]:
    return {initializer.name: initializer for initializer in model.graph.initializer}


def _unique_name(existing: set[str], desired: str) -> str:
    name = desired
    suffix = 1
    while name in existing:
        name = f"{desired}_{suffix}"
        suffix += 1
    existing.add(name)
    return name


def _fp8_zero_initializer(name: str, shape: tuple[int, ...]) -> onnx.TensorProto:
    values = np.zeros(shape, dtype=ml_dtypes.float8_e4m3fn)
    return numpy_helper.from_array(values, name=name)


def _scale(values: np.ndarray, dtype: np.dtype[np.floating]) -> np.ndarray:
    minimum = np.nextafter(dtype.type(0), dtype.type(1), dtype=dtype)
    return np.maximum(values.astype(dtype, copy=False) / dtype.type(_FP8_MAX), minimum)


def _insert_ranker_qdq(
    model: onnx.ModelProto, activation_maxima: dict[str, float], scope: Literal["all", "inner"]
) -> tuple[onnx.ModelProto, list[dict[str, object]]]:
    if not any(opset.domain == "" and opset.version >= 19 for opset in model.opset_import):
        raise ValueError("FP8 Q/DQ requires the default ONNX opset to be at least 19.")

    initializers = _initializer_map(model)
    names = {
        value.name
        for value in ([*model.graph.input, *model.graph.output, *model.graph.value_info, *model.graph.initializer])
    }
    names.update(output for node in model.graph.node for output in node.output)
    rewritten_nodes: list[onnx.NodeProto] = []
    added_initializers: list[onnx.TensorProto] = []
    metadata: list[dict[str, object]] = []

    for node in model.graph.node:
        if node.op_type != "Conv" or len(node.input) < 2:
            rewritten_nodes.append(node)
            continue
        weight_name = node.input[1]
        module_name = _ranker_module_name(weight_name)
        if module_name is None:
            rewritten_nodes.append(node)
            continue
        if weight_name not in initializers:
            raise ValueError(f"Ranker weight is not an initializer: {weight_name}")
        stored_weights = numpy_helper.to_array(initializers[weight_name])
        is_inner_spatial = stored_weights.shape[:2] == (12, 12) and stored_weights.shape[2:] == (3, 3)
        if scope == "inner" and not is_inner_spatial:
            rewritten_nodes.append(node)
            continue
        if module_name not in activation_maxima:
            raise ValueError(f"Missing calibration maximum for {module_name} ({node.name or weight_name}).")

        stem = node.name or weight_name.replace(".", "_")
        activation_scale_name = _unique_name(names, f"{stem}_input_scale")
        activation_zero_name = _unique_name(names, f"{stem}_input_zero_point")
        activation_quantized = _unique_name(names, f"{stem}_input_fp8")
        activation_dequantized = _unique_name(names, f"{stem}_input_dequantized")
        weight_scale_name = _unique_name(names, f"{stem}_weight_scale")
        weight_zero_name = _unique_name(names, f"{stem}_weight_zero_point")
        weight_quantized = _unique_name(names, f"{stem}_weight_fp8")
        weight_dequantized = _unique_name(names, f"{stem}_weight_dequantized")

        if stored_weights.dtype not in {np.dtype(np.float16), np.dtype(np.float32)}:
            raise ValueError(f"Expected FP16 or FP32 ranker weights, found {stored_weights.dtype} for {weight_name}.")
        scale_dtype = stored_weights.dtype
        activation_scale = _scale(np.asarray(activation_maxima[module_name]), scale_dtype)
        weights = stored_weights.astype(np.float32, copy=False)
        reduce_axes = tuple(range(1, weights.ndim))
        weight_scale = _scale(np.max(np.abs(weights), axis=reduce_axes), scale_dtype)

        added_initializers.extend(
            (
                numpy_helper.from_array(activation_scale, name=activation_scale_name),
                _fp8_zero_initializer(activation_zero_name, ()),
                numpy_helper.from_array(weight_scale, name=weight_scale_name),
                _fp8_zero_initializer(weight_zero_name, weight_scale.shape),
            )
        )
        rewritten_nodes.extend(
            (
                helper.make_node(
                    "QuantizeLinear",
                    (node.input[0], activation_scale_name, activation_zero_name),
                    (activation_quantized,),
                    name=f"{stem}_input_QuantizeLinear",
                    saturate=1,
                ),
                helper.make_node(
                    "DequantizeLinear",
                    (activation_quantized, activation_scale_name, activation_zero_name),
                    (activation_dequantized,),
                    name=f"{stem}_input_DequantizeLinear",
                ),
                helper.make_node(
                    "QuantizeLinear",
                    (weight_name, weight_scale_name, weight_zero_name),
                    (weight_quantized,),
                    name=f"{stem}_weight_QuantizeLinear",
                    axis=0,
                    saturate=1,
                ),
                helper.make_node(
                    "DequantizeLinear",
                    (weight_quantized, weight_scale_name, weight_zero_name),
                    (weight_dequantized,),
                    name=f"{stem}_weight_DequantizeLinear",
                    axis=0,
                ),
            )
        )
        node.input[0] = activation_dequantized
        node.input[1] = weight_dequantized
        rewritten_nodes.append(node)
        metadata.append(
            {
                "node": node.name,
                "module": module_name,
                "activation_max": activation_maxima[module_name],
                "activation_scale": float(activation_scale),
                "weight_scale_min": float(weight_scale.min()),
                "weight_scale_max": float(weight_scale.max()),
            }
        )

    if not metadata:
        raise ValueError("No ranker Conv nodes were found. Expected initializer names containing 'ranker_head.'.")
    if scope == "all" and len(metadata) != len(activation_maxima):
        missing = sorted(set(activation_maxima) - {str(item["module"]) for item in metadata})
        raise ValueError(
            f"Rewrote {len(metadata)} Conv nodes but calibrated {len(activation_maxima)}: missing {missing}"
        )

    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    model.graph.initializer.extend(added_initializers)
    onnx.checker.check_model(model, full_check=True)
    return model, metadata


@app.command()
def export(
    output: Annotated[Path, typer.Option("-o", "--output", dir_okay=False, writable=True)],
    batch_size: Annotated[int, typer.Option("-b", "--batch-size", min=1)] = 2,
    height: Annotated[int, typer.Option("-h", "--height", min=32)] = 1280,
    width: Annotated[int, typer.Option("-w", "--width", min=32)] = 1280,
    opset: Annotated[int, typer.Option(min=19, max=20)] = 20,
    fp16: Annotated[
        bool, typer.Option("--fp16/--fp32", help="Export explicit FP16 types for strongly typed TensorRT.")
    ] = True,
) -> None:
    """Export the fused RaCo ranker alone for low-risk TensorRT experiments."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to keep the standalone export's activation memory off the host.")
    output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    ranker = _build_ranker(device)
    dtype = torch.float16 if fp16 else torch.float32
    ranker.to(dtype=dtype)
    example = torch.zeros(batch_size, 3, height, width, device=device, dtype=dtype)
    torch.onnx.export(
        ranker,
        (example,),
        output,
        input_names=("normalized_images",),
        output_names=("ranker_map",),
        opset_version=opset,
        dynamo=True,
        external_data=False,
        optimize=True,
    )
    model = onnx.load_model(output, load_external_data=False)
    conv_count = sum(node.op_type == "Conv" for node in model.graph.node)
    if conv_count != 28:
        raise RuntimeError(f"Expected 28 fused ranker Conv nodes, found {conv_count}.")
    typer.echo(f"Exported {conv_count}-Conv ranker to {output}")


@app.command()
def quantize(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Ranker or pipeline ONNX; explicit FP16 is recommended for strongly typed TensorRT.",
        ),
    ],
    output: Annotated[Path, typer.Option("-o", "--output", dir_okay=False, writable=True)],
    images: Annotated[
        list[Path] | None,
        typer.Option("--images", exists=True, dir_okay=False, readable=True, help="Streamed calibration images."),
    ] = None,
    height: Annotated[int, typer.Option("-h", "--height", min=32)] = 1280,
    width: Annotated[int, typer.Option("-w", "--width", min=32)] = 1280,
    batch_size: Annotated[int, typer.Option("-b", "--batch-size", min=1, max=8)] = 1,
    scope: Annotated[
        Literal["all", "inner"],
        typer.Option(help="Quantize all ranker convolutions, or only the inner 12x12 3x3 convolutions."),
    ] = "all",
) -> None:
    """Insert explicit FP8 E4M3 Q/DQ around only the RaCo ranker Conv inputs and weights."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for bounded-memory ranker calibration.")
    image_paths = list(images or _DEFAULT_IMAGES)
    if not image_paths:
        raise typer.BadParameter("Provide at least one calibration image.")

    ranker = _build_ranker(torch.device("cuda"))
    maxima, peak_mib = _calibrate_activation_maxima(ranker, image_paths, height, width, batch_size)
    del ranker
    torch.cuda.empty_cache()

    model = onnx.load_model(input_path, load_external_data=True)
    model, nodes = _insert_ranker_qdq(model, maxima, scope)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, output, save_as_external_data=False)
    report = {
        "input": str(input_path),
        "output": str(output),
        "height": height,
        "width": width,
        "calibration_images": [str(path) for path in image_paths],
        "calibration_batch_size": batch_size,
        "calibration_peak_cuda_mib": peak_mib,
        "format": "FLOAT8E4M3FN",
        "activation_granularity": "per-tensor",
        "weight_granularity": "per-output-channel",
        "scope": scope,
        "quantized_convolutions": nodes,
    }
    report_path = output.with_suffix(".calibration.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    typer.echo(
        f"Quantized {len(nodes)} ranker convolutions to explicit FP8 Q/DQ at {output}; "
        f"CUDA calibration peak {peak_mib:.1f} MiB; report {report_path}"
    )


if __name__ == "__main__":
    app()
