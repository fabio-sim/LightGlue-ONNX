import ctypes
import os
import site
from pathlib import Path

import typer

_PRELOADED_LIBRARIES: list[ctypes.CDLL] = []


def check_multiple_of(value: int, k: int) -> None:
    if value % k != 0:
        raise typer.BadParameter(f"Value must be a multiple of {k}.")


def preload_nvidia_libraries(*, tensorrt: bool = False) -> None:
    """Preload NVIDIA wheels whose shared-library directories are not on the loader path."""
    roots = [Path(path) for path in [*site.getsitepackages(), site.getusersitepackages()] if path]
    library_names = [
        ("libcudart.so.13", "libcudart.so.12"),
        ("libnvJitLink.so.13", "libnvJitLink.so.12"),
        ("libcublasLt.so.13", "libcublasLt.so.12"),
        ("libcublas.so.13", "libcublas.so.12"),
        ("libnvrtc.so.13", "libnvrtc.so.12"),
        ("libcurand.so.10",),
        ("libcufft.so.12", "libcufft.so.11"),
        ("libcudnn_graph.so.9",),
        ("libcudnn_ops.so.9",),
        ("libcudnn_heuristic.so.9",),
        ("libcudnn_engines_precompiled.so.9",),
        ("libcudnn_engines_runtime_compiled.so.9",),
        ("libcudnn_cnn.so.9",),
        ("libcudnn_adv.so.9",),
        ("libcudnn.so.9",),
    ]
    if tensorrt:
        library_names.extend([("libnvinfer.so.10",), ("libnvinfer_plugin.so.10",), ("libnvonnxparser.so.10",)])

    loaded_directories: set[Path] = set()
    for alternatives in library_names:
        candidates = (
            candidate
            for name in alternatives
            for root in roots
            for pattern in (f"nvidia/**/{name}", f"tensorrt_libs/{name}")
            for candidate in root.glob(pattern)
        )
        path = next(candidates, None)
        if path is not None:
            _PRELOADED_LIBRARIES.append(ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL))
            loaded_directories.add(path.parent)

    if loaded_directories:
        current = os.environ.get("LD_LIBRARY_PATH", "")
        entries = [*(str(path) for path in sorted(loaded_directories)), *(part for part in current.split(":") if part)]
        os.environ["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(entries))
