"""Make pip-installed CUDA libraries (cuBLAS, cuDNN) visible on Windows.

ctranslate2 / faster-whisper load cublas64_12.dll and cudnn*.dll at runtime.
The ``nvidia-cublas-cu12`` / ``nvidia-cudnn-cu12`` wheels put those DLLs in
``site-packages/nvidia/<lib>/bin`` which is not on the DLL search path, so
call :func:`register_cuda_dlls` BEFORE importing ctranslate2 or faster_whisper.
"""
from __future__ import annotations

import os
import site
import sys
from pathlib import Path

_NVIDIA_LIBS = ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc")
_registered: list[str] = []
_dll_handles = []  # keep add_dll_directory handles alive for the process


def _site_dirs() -> list[Path]:
    dirs = []
    try:
        dirs.extend(site.getsitepackages())
    except AttributeError:  # some embedded / virtualenv builds
        pass
    try:
        dirs.append(site.getusersitepackages())
    except AttributeError:
        pass
    dirs.extend(p for p in sys.path if p.endswith("site-packages"))
    seen, out = set(), []
    for d in dirs:
        p = Path(d)
        if p.is_dir() and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def find_cuda_dll_dirs() -> list[Path]:
    """Return existing nvidia/<lib>/bin directories inside site-packages."""
    found = []
    for sp in _site_dirs():
        for lib in _NVIDIA_LIBS:
            bin_dir = sp / "nvidia" / lib / "bin"
            if bin_dir.is_dir() and bin_dir not in found:
                found.append(bin_dir)
    return found


def register_cuda_dlls(verbose: bool = False) -> list[str]:
    """Register CUDA DLL directories via os.add_dll_directory and PATH.

    Idempotent; a no-op on non-Windows platforms. Returns the registered dirs.
    """
    if os.name != "nt":
        return []
    if _registered:
        return list(_registered)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    for bin_dir in find_cuda_dll_dirs():
        s = str(bin_dir)
        try:
            _dll_handles.append(os.add_dll_directory(s))
        except (OSError, AttributeError):
            pass
        if s not in path_parts:
            path_parts.insert(0, s)
        _registered.append(s)
        if verbose:
            print(f"[cuda_setup] registered {s}")
    os.environ["PATH"] = os.pathsep.join(path_parts)
    return list(_registered)
