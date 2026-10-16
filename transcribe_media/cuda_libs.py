"""Make pip-installed NVIDIA runtime libraries (cuBLAS, cuDNN) loadable.

CTranslate2 opens ``libcublas`` / ``cudnn`` at run time. The ``nvidia-*-cu12``
wheels put them under ``site-packages/nvidia/*/``, which neither Windows nor
Linux searches by default, so GPU inference otherwise fails with "Library
... is not found". Call :func:`configure_cuda_environment` before importing
``ctranslate2`` or ``torch``.
"""

from __future__ import annotations

import ctypes
import glob
import importlib.util
import os
import sys

# Load order matters: cuBLASLt before cuBLAS, then cuDNN.
_LINUX_LIBS = ("libcublasLt.so.*", "libcublas.so.*", "libcudnn.so.*")


def nvidia_lib_dirs(subdir: str) -> list[str]:
    """Existing ``site-packages/nvidia/*/<subdir>`` directories, de-duplicated."""
    try:
        spec = importlib.util.find_spec("nvidia")
        roots = list(spec.submodule_search_locations) if spec and spec.submodule_search_locations else []
    except (ImportError, AttributeError, ValueError):
        roots = []
    seen: set[str] = set()
    dirs: list[str] = []
    for root in roots:
        for path in sorted(glob.glob(os.path.join(root, "*", subdir))):
            key = os.path.normcase(os.path.abspath(path))
            if os.path.isdir(path) and key not in seen:
                seen.add(key)
                dirs.append(path)
    return dirs


def _register_windows_dirs() -> list[str]:
    dirs = nvidia_lib_dirs("bin")
    for path in dirs:
        try:
            os.add_dll_directory(path)
        except OSError:
            pass
    # Newer CTranslate2 loads cuBLAS with a plain LoadLibrary, which ignores
    # add_dll_directory, so PATH is extended as well.
    current = os.environ.get("PATH", "")
    have = {os.path.normcase(p) for p in current.split(os.pathsep) if p}
    missing = [d for d in dirs if os.path.normcase(d) not in have]
    if missing:
        os.environ["PATH"] = os.pathsep.join([*missing, current] if current else missing)
    return dirs


def _preload_linux_libs() -> list[str]:
    """dlopen the libraries globally so CTranslate2's later dlopen finds them."""
    failures: list[str] = []
    for pattern in _LINUX_LIBS:
        for lib_dir in nvidia_lib_dirs("lib"):
            for path in sorted(glob.glob(os.path.join(lib_dir, pattern))):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                except OSError as exc:
                    failures.append(f"{path}: {exc}")
    return failures


def configure_cuda_environment() -> list[str]:
    """Prepare the process for CUDA inference; returns any library load failures.

    Also makes CUDA device indices match ``nvidia-smi`` (PCI bus order), so the
    GPU chosen from hardware detection is the one actually used.
    """
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        _register_windows_dirs()
        return []
    if sys.platform.startswith("linux"):
        return _preload_linux_libs()
    return []
