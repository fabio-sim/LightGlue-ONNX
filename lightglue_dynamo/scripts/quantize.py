# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "nvidia-modelopt[all]==0.40.0",
#     "numpy==2.2.6",
#     "opencv-python==4.12.0.88",
#     "tensorrt==10.9.0.34",
#     "typer==0.21.0",
# ]
# ///

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal

import cv2
import modelopt.onnx.quantization as moq
import numpy as np
import onnx
import typer


def _parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() in {"none", "null"}:
        return None
    items = [item.strip() for item in stripped.split(",") if item.strip()]
    return items or None


def _load_and_resize(path: Path, width: int, height: int) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.resize(image, (width, height))


def _preprocess_superpoint(images: np.ndarray) -> np.ndarray:
    images = images[..., ::-1] / 255 * [0.299, 0.587, 0.114]
    images = images.sum(axis=-1, keepdims=True)
    axes = [*list(range(images.ndim - 3)), -1, -3, -2]
    return images.transpose(*axes)


def _preprocess_disk(images: np.ndarray) -> np.ndarray:
    images = images / 255
    axes = [*list(range(images.ndim - 3)), -1, -3, -2]
    return images.transpose(*axes)


def _build_calibration_data(
    extractor: str, image_paths: list[Path], height: int, width: int, max_pairs: int
) -> np.ndarray:
    if len(image_paths) % 2 != 0:
        raise ValueError("Provide an even number of images so they can be paired.")

    if max_pairs > 0:
        image_paths = image_paths[: max_pairs * 2]

    if not image_paths:
        raise ValueError("No calibration images provided after applying --max-pairs.")

    channels = 1 if extractor == "superpoint" else 3
    calibration_data = np.empty((len(image_paths), channels, height, width), dtype=np.float32)
    for idx in range(0, len(image_paths), 2):
        left = _load_and_resize(image_paths[idx], width, height)
        right = _load_and_resize(image_paths[idx + 1], width, height)
        raw = np.stack([left, right])
        processed = _preprocess_superpoint(raw) if extractor == "superpoint" else _preprocess_disk(raw)
        if processed.shape[1:] != calibration_data.shape[1:]:
            raise ValueError(
                "Processed calibration data shape does not match expected input shape: "
                f"{processed.shape} vs {calibration_data.shape}"
            )
        calibration_data[idx : idx + 2] = processed.astype(np.float32, copy=False)
    return calibration_data


def _infer_input_spec(model_path: Path) -> tuple[str, list[int | None]]:
    model = onnx.load_model(model_path)
    if not model.graph.input:
        raise ValueError("Model has no inputs.")
    input_value = model.graph.input[0]
    if not input_value.type.HasField("tensor_type"):
        raise ValueError(f"Input {input_value.name} is not a tensor.")
    shape = input_value.type.tensor_type.shape
    dims: list[int | None] = []
    for dim in shape.dim:
        if dim.HasField("dim_value") and dim.dim_value > 0:
            dims.append(int(dim.dim_value))
        else:
            dims.append(None)
    return input_value.name, dims


def _require_static_shape(input_name: str, dims: Sequence[int | None]) -> None:
    if len(dims) != 4:
        raise ValueError(f"Expected 4D input for {input_name}, got {len(dims)}D: {dims}")
    if any(dim is None for dim in dims):
        raise ValueError(
            f"Input {input_name} has dynamic dimensions {dims}. Export a static ONNX model first "
            "by setting batch size/height/width to non-zero values."
        )


def main(
    input_path: Annotated[
        Path,
        typer.Option("--input", exists=True, dir_okay=False, readable=True, help="Path to ONNX model to quantize."),
    ],
    output_path: Annotated[
        Path, typer.Option("--output", dir_okay=False, writable=True, help="Path to save quantized ONNX model.")
    ],
    extractor: Annotated[
        Literal["superpoint", "disk"], typer.Option("--extractor", help="Extractor type used to select preprocessing.")
    ],
    height: Annotated[int, typer.Option("--height", min=1, help="Resize height for calibration images.")] = 1024,
    width: Annotated[int, typer.Option("--width", min=1, help="Resize width for calibration images.")] = 1024,
    images: Annotated[
        list[Path] | None,
        typer.Option("--images", exists=True, dir_okay=False, readable=True, help="Image paths (paired in order)."),
    ] = None,
    max_pairs: Annotated[
        int,
        typer.Option(
            "--max-pairs", min=0, help="Optional cap on the number of image pairs used for calibration (0 = all)."
        ),
    ] = 0,
    quantize_mode: Annotated[
        Literal["fp8", "int8", "int4"], typer.Option("--quantize-mode", help="Quantization mode for ModelOpt.")
    ] = "fp8",
    calibration_method: Annotated[
        Literal["entropy", "max"], typer.Option("--calibration-method", help="Calibration method for int8/fp8 modes.")
    ] = "entropy",
    calibration_eps: Annotated[
        str,
        typer.Option("--calibration-eps", help="Comma-separated calibration EP priority list (e.g. trt,cuda:0,cpu)."),
    ] = "trt,cuda:0,cpu",
    calibrate_per_node: Annotated[
        bool,
        typer.Option("--calibrate-per-node", help="Calibrate activations per node to reduce memory usage (slower)."),
    ] = False,
    op_types_to_quantize: Annotated[
        str | None,
        typer.Option("--op-types-to-quantize", help="Comma-separated op types to quantize (e.g. MatMul,Conv)."),
    ] = None,
    op_types_to_exclude: Annotated[
        str | None, typer.Option("--op-types-to-exclude", help="Comma-separated op types to exclude from quantization.")
    ] = None,
    op_types_to_exclude_fp16: Annotated[
        str | None,
        typer.Option("--op-types-to-exclude-fp16", help="Comma-separated op types to keep in FP32 when fp16."),
    ] = None,
    nodes_to_quantize: Annotated[
        str | None,
        typer.Option(
            "--nodes-to-quantize", help="Comma-separated node names/patterns to quantize (regex support varies)."
        ),
    ] = None,
    nodes_to_exclude: Annotated[
        str | None,
        typer.Option("--nodes-to-exclude", help="Comma-separated node names/patterns to exclude from quantization."),
    ] = None,
    high_precision_dtype: Annotated[
        Literal["fp16", "fp32"] | None,
        typer.Option(
            "--high-precision-dtype",
            help="High-precision dtype for non-quantized ops. Defaults to fp32 for fp8, fp16 otherwise.",
        ),
    ] = None,
    mha_accumulation_dtype: Annotated[
        Literal["fp16", "fp32"],
        typer.Option("--mha-accumulation-dtype", help="Accumulation dtype for MHA if ModelOpt fuses attention."),
    ] = "fp16",
    disable_mha_qdq: Annotated[
        bool,
        typer.Option("--disable-mha-qdq", help="Disable Q/DQ insertion around multi-head attention (if detected)."),
    ] = False,
    direct_io_types: Annotated[
        bool, typer.Option("--direct-io-types", help="Allow ModelOpt to lower I/O types when possible.")
    ] = False,
    dq_only: Annotated[bool, typer.Option("--dq-only/--no-dq-only", help="Use DQ-only output graph.")] = False,
    simplify: Annotated[
        bool, typer.Option("--simplify/--no-simplify", help="Run ONNX simplifier on the quantized model.")
    ] = False,
) -> None:
    input_name, input_shape = _infer_input_spec(input_path)
    _require_static_shape(input_name, input_shape)
    if images is None:
        default_image_paths = [
            "assets/sacre_coeur1.jpg",
            "assets/sacre_coeur2.jpg",
            "assets/DSC_0410.JPG",
            "assets/DSC_0411.JPG",
        ]
        images = [Path(p) for p in default_image_paths]

    calibration_data = _build_calibration_data(extractor, images, height, width, max_pairs)

    calibration_eps_list = [ep.strip() for ep in calibration_eps.split(",") if ep.strip()]

    op_types_to_quantize_list = _parse_csv_list(op_types_to_quantize)
    op_types_to_exclude_list = _parse_csv_list(op_types_to_exclude)
    op_types_to_exclude_fp16_list = _parse_csv_list(op_types_to_exclude_fp16)
    nodes_to_quantize_list = _parse_csv_list(nodes_to_quantize)
    nodes_to_exclude_list = _parse_csv_list(nodes_to_exclude)

    resolved_high_precision = high_precision_dtype or ("fp32" if quantize_mode == "fp8" else "fp16")

    moq.quantize(
        onnx_path=str(input_path),
        quantize_mode=quantize_mode,
        calibration_data=calibration_data,
        calibration_method=calibration_method,
        calibration_eps=calibration_eps_list,
        op_types_to_quantize=op_types_to_quantize_list,
        op_types_to_exclude=op_types_to_exclude_list,
        op_types_to_exclude_fp16=op_types_to_exclude_fp16_list,
        nodes_to_quantize=nodes_to_quantize_list,
        nodes_to_exclude=nodes_to_exclude_list,
        high_precision_dtype=resolved_high_precision,
        mha_accumulation_dtype=mha_accumulation_dtype,
        disable_mha_qdq=disable_mha_qdq,
        dq_only=dq_only,
        simplify=simplify,
        calibrate_per_node=calibrate_per_node,
        direct_io_types=direct_io_types,
        output_path=str(output_path),
    )

    typer.echo(f"Saved quantized model to {output_path}")


if __name__ == "__main__":
    typer.run(main)
