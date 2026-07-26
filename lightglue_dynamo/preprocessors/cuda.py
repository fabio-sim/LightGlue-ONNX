from __future__ import annotations

import os
import site
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from lightglue_dynamo.cli_utils import preload_nvidia_libraries

CudaPreprocessDType = Literal["float16", "float32"]

_EXTENSION_DIRECTORY: tempfile.TemporaryDirectory[str] | None = None


class _OrtValueFactory(Protocol):
    def from_dlpack(self, data: object, /) -> object: ...


class OrtModule(Protocol):
    OrtValue: _OrtValueFactory


@contextmanager
def _jpeg_only_nvimgcodec_extensions() -> Iterator[None]:
    """Temporarily select installed JPEG plugins without changing the caller's environment."""
    global _EXTENSION_DIRECTORY
    if "NVIMGCODEC_EXTENSIONS_PATH" in os.environ:
        yield
        return
    if _EXTENSION_DIRECTORY is None:
        roots = [Path(path) for path in [*site.getsitepackages(), site.getusersitepackages()] if path]
        extension_root = next(
            (
                root / "nvidia/nvimgcodec/extensions"
                for root in roots
                if (root / "nvidia/nvimgcodec/extensions").is_dir()
            ),
            None,
        )
        if extension_root is None:
            yield
            return
        _EXTENSION_DIRECTORY = tempfile.TemporaryDirectory(prefix="lightglue-nvimgcodec-")
        destination = Path(_EXTENSION_DIRECTORY.name)
        for pattern in ("libnvjpeg_ext.so*", "libjpeg_turbo_ext.so*"):
            for source in extension_root.glob(pattern):
                target = destination / source.name
                if not target.exists():
                    target.symlink_to(source)
    destination = Path(_EXTENSION_DIRECTORY.name)
    os.environ["NVIMGCODEC_EXTENSIONS_PATH"] = str(destination)
    try:
        yield
    finally:
        os.environ.pop("NVIMGCODEC_EXTENSIONS_PATH", None)


@dataclass
class _CudaPreprocessSlot:
    stream: Any
    decoder: Any
    resized: list[Any]
    stacked: Any
    scaled: Any
    output: Any
    decoded: list[Any] | None = None
    busy: bool = False


@dataclass
class CudaPreparedImages:
    """GPU-resident RaCo input and the resources that own its storage."""

    slot: _CudaPreprocessSlot
    original_shapes: tuple[tuple[int, int], tuple[int, int]]
    submitted_at: float
    submission_ms: float
    _ready_ms: float | None = None
    _external_buffer: object | None = field(default=None, init=False, repr=False)
    _ort_value: object | None = field(default=None, init=False, repr=False)

    @property
    def shape(self) -> tuple[int, ...]:
        return cast(tuple[int, ...], tuple(self.slot.output.shape))

    def synchronize(self) -> float:
        if self._ready_ms is None:
            self.slot.stream.sync()
            self._ready_ms = (time.perf_counter() - self.submitted_at) * 1000
        return self._ready_ms

    def to_ort_value(self, ort: OrtModule) -> object:
        self.synchronize()
        if self._ort_value is None:
            self._external_buffer = self.slot.output.cuda()
            self._ort_value = ort.OrtValue.from_dlpack(self._external_buffer)
        return self._ort_value

    def original_bgr_images(self) -> list[Any]:
        """Copy decoded RGB images to host only when visualization is requested."""
        import numpy as np

        self.synchronize()
        if self.slot.decoded is None:
            raise RuntimeError("Decoded image storage was released")
        return [np.asarray(image.cpu())[..., ::-1].copy() for image in self.slot.decoded]

    def release(self) -> None:
        self.synchronize()
        self._ort_value = None
        self._external_buffer = None
        self.slot.busy = False


class CudaRaCoPreprocessor:
    """Decode JPEG pairs and prepare RaCo inputs without a host image tensor."""

    def __init__(self, width: int, height: int, *, dtype: CudaPreprocessDType = "float32", slots: int = 1) -> None:
        if width < 1 or height < 1:
            raise ValueError("width and height must be positive")
        if slots < 1:
            raise ValueError("slots must be positive")
        if dtype != "float32":
            raise ValueError(
                "CUDA preprocessing currently requires a float32 ONNX input. TensorRT FP16 is supported because "
                "TensorRT keeps the model boundary in float32 and selects FP16 internally."
            )
        preload_nvidia_libraries(image_codec=True)
        with _jpeg_only_nvimgcodec_extensions():
            try:
                import cvcuda
                from nvidia import nvimgcodec
            except ImportError as exc:
                raise RuntimeError(
                    "CUDA preprocessing requires the gpu-preprocess dependency group: "
                    "uv sync --no-default-groups --group cuda --group gpu-preprocess"
                ) from exc

            self.cvcuda = cvcuda
            self.nvimgcodec = nvimgcodec
            self.width = width
            self.height = height
            self.dtype = dtype
            output_type = cvcuda.Type.F32
            backends = [nvimgcodec.Backend(nvimgcodec.BackendKind.HYBRID_CPU_GPU)]
            self.slots: list[_CudaPreprocessSlot] = []
            for _ in range(slots):
                stream = cvcuda.Stream()
                resized = [cvcuda.Tensor((height, width, 3), cvcuda.Type.U8, "HWC") for _ in range(2)]
                self.slots.append(
                    _CudaPreprocessSlot(
                        stream=stream,
                        decoder=nvimgcodec.Decoder(backends=backends, options=":num_cuda_streams=1"),
                        resized=resized,
                        stacked=cvcuda.Tensor((2, height, width, 3), cvcuda.Type.U8, "NHWC"),
                        scaled=cvcuda.Tensor((2, height, width, 3), output_type, "NHWC"),
                        output=cvcuda.Tensor((2, 3, height, width), output_type, "NCHW"),
                    )
                )

    def prepare(self, paths: tuple[Path, Path]) -> CudaPreparedImages:
        if any(path.suffix.lower() not in {".jpg", ".jpeg"} for path in paths):
            raise ValueError("CUDA preprocessing currently supports JPEG inputs; use --preprocessing opencv otherwise")
        slot = next((candidate for candidate in self.slots if not candidate.busy), None)
        if slot is None:
            raise RuntimeError("No free CUDA preprocessing slot")
        slot.busy = True
        start = time.perf_counter()
        try:
            decoded = slot.decoder.read(
                [str(path) for path in paths], images=slot.decoded, cuda_stream=slot.stream.handle
            )
            if any(image is None for image in decoded):
                raise RuntimeError(f"nvImageCodec failed to decode {paths[0]} or {paths[1]}")
            slot.decoded = cast(list[Any], decoded)
            original_shapes = cast(
                tuple[tuple[int, int], tuple[int, int]],
                tuple((int(image.shape[0]), int(image.shape[1])) for image in slot.decoded),
            )
            tensors = [self.cvcuda.as_tensor(image, "HWC") for image in slot.decoded]
            for destination, source, image in zip(slot.resized, tensors, slot.decoded, strict=True):
                # CV-CUDA AREA is intended for minification and can leave invalid
                # regions when either axis is magnified. OpenCV also treats AREA
                # as a linear-style interpolation for enlargement.
                interpolation = (
                    self.cvcuda.Interp.AREA
                    if self.width <= int(image.shape[1]) and self.height <= int(image.shape[0])
                    else self.cvcuda.Interp.LINEAR
                )
                self.cvcuda.resize_into(destination, source, interpolation, stream=slot.stream)
            self.cvcuda.stack_into(slot.stacked, slot.resized, stream=slot.stream)
            self.cvcuda.convertto_into(slot.scaled, slot.stacked, scale=1 / 255.0, stream=slot.stream)
            self.cvcuda.reformat_into(slot.output, slot.scaled, stream=slot.stream)
        except Exception:
            slot.busy = False
            raise
        submission_ms = (time.perf_counter() - start) * 1000
        return CudaPreparedImages(slot, original_shapes, start, submission_ms)

    def close(self) -> None:
        for slot in self.slots:
            slot.stream.sync()
            slot.busy = False
