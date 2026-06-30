"""JAX runtime helpers."""

from __future__ import annotations

import os
import sys
import ctypes
from pathlib import Path


def bootstrap_cuda_wheel_paths(*, reexec: bool = True) -> None:
    """Re-exec with NVIDIA wheel library paths so JAX can find CUDA libs."""
    if os.environ.get("NTHMC_JAX_CUDA_BOOTSTRAPPED") == "1":
        return
    roots = [Path(sys.prefix), Path(sys.executable).resolve().parents[1]]
    candidates = []
    for root in roots:
        candidates.extend(root.glob("lib/python*/site-packages/nvidia/*/lib"))
    candidates = sorted(set(candidates))
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    paths = [str(path) for path in candidates if path.is_dir()]
    if not paths:
        os.environ["NTHMC_JAX_CUDA_BOOTSTRAPPED"] = "1"
        return
    merged = ":".join(paths + ([existing] if existing else []))
    if existing == merged:
        os.environ["NTHMC_JAX_CUDA_BOOTSTRAPPED"] = "1"
        return
    os.environ["LD_LIBRARY_PATH"] = merged
    os.environ["NTHMC_JAX_CUDA_BOOTSTRAPPED"] = "1"
    if reexec:
        os.execv(sys.executable, [sys.executable, *sys.argv])


# def set_platform(device: str) -> str:
#     """Resolve auto/gpu/cuda/cpu and set JAX platform before import-heavy work."""
#     normalized = "gpu" if device in {"cuda", "gpu"} else device
#     if normalized == "auto":
#         return "auto"
#     if normalized not in {"cpu", "gpu"}:
#         raise ValueError(f"Unsupported device: {device!r}")
#     os.environ["JAX_PLATFORM_NAME"] = normalized
#     os.environ["JAX_PLATFORMS"] = normalized
#     return normalized

def set_platform(device: str) -> str:
    """Resolve auto/gpu/cuda/cpu and set JAX platform before import-heavy work."""
    normalized = "cuda" if device in {"cuda", "gpu"} else device
    if normalized == "auto":
        return "auto"
    if normalized not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device!r}")
    os.environ["JAX_PLATFORM_NAME"] = normalized
    os.environ["JAX_PLATFORMS"] = normalized
    return normalized


def preconfigure_platform_from_argv(argv: list[str] | None = None) -> str:
    """Set JAX platform from a simple --device CLI argument before importing JAX."""
    argv = list(sys.argv if argv is None else argv)
    device = "auto"
    for index, arg in enumerate(argv):
        if arg == "--device" and index + 1 < len(argv):
            device = argv[index + 1]
            break
        if arg.startswith("--device="):
            device = arg.split("=", 1)[1]
            break
    # if device == "cuda":
    #     device = "gpu"
    if device != "auto":
        set_platform(device)
    return device


def preload_cuda_wheel_libraries() -> None:
    """Best-effort notebook-safe preload of NVIDIA wheel libraries."""
    root = Path(sys.prefix)
    names = [
        "cuda_runtime/lib/libcudart.so.12",
        "cublas/lib/libcublas.so.12",
        "cublas/lib/libcublasLt.so.12",
        "cusparse/lib/libcusparse.so.12",
        "cusolver/lib/libcusolver.so.11",
        "cufft/lib/libcufft.so.11",
        "cuda_nvrtc/lib/libnvrtc.so.12",
        "nvjitlink/lib/libnvJitLink.so.12",
    ]
    nvidia_root = next(iter(root.glob("lib/python*/site-packages/nvidia")), None)
    if nvidia_root is None:
        return
    for name in names:
        path = nvidia_root / name
        if path.exists():
            try:
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
