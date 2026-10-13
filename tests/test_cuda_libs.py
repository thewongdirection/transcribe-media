import ctypes
import os
import sys
from types import SimpleNamespace

import pytest

from transcribe_media import cuda_libs


@pytest.fixture
def site(tmp_path, monkeypatch):
    """A fake site-packages/nvidia tree, discoverable via importlib."""
    root = tmp_path / "nvidia"
    for rel in (
        "cublas/lib/libcublasLt.so.12",
        "cublas/lib/libcublas.so.12",
        "cudnn/lib/libcudnn.so.9",
        "cublas/bin/cublas64_12.dll",
    ):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    monkeypatch.setattr(
        cuda_libs.importlib.util, "find_spec",
        lambda name: SimpleNamespace(submodule_search_locations=[str(root)]),
    )
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "placeholder")  # restored on teardown
    monkeypatch.delenv("CUDA_DEVICE_ORDER")
    return root


def test_no_nvidia_package_means_no_dirs(monkeypatch):
    monkeypatch.setattr(cuda_libs.importlib.util, "find_spec", lambda name: None)
    assert cuda_libs.nvidia_lib_dirs("lib") == []


def test_linux_preloads_in_dependency_order_with_global_mode(site, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    loaded = []
    monkeypatch.setattr(ctypes, "CDLL", lambda path, mode: loaded.append((os.path.basename(path), mode)))
    failures = cuda_libs.configure_cuda_environment()
    assert failures == []
    assert [n for n, _ in loaded] == ["libcublasLt.so.12", "libcublas.so.12", "libcudnn.so.9"]
    assert all(mode == ctypes.RTLD_GLOBAL for _, mode in loaded)


def test_linux_collects_load_failures_instead_of_raising(site, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")

    def boom(path, mode):
        raise OSError("bad ELF")

    monkeypatch.setattr(ctypes, "CDLL", boom)
    failures = cuda_libs.configure_cuda_environment()
    assert len(failures) == 3 and all("bad ELF" in f for f in failures)


def test_device_order_defaults_to_pci_bus_but_respects_user_choice(site, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    cuda_libs.configure_cuda_environment()
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    cuda_libs.configure_cuda_environment()
    assert os.environ["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"


def test_windows_registers_dll_dirs_and_extends_path_once(site, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    added = []
    monkeypatch.setattr(os, "add_dll_directory", added.append, raising=False)
    monkeypatch.setenv("PATH", "C:\\existing")
    cuda_libs.configure_cuda_environment()
    cuda_libs.configure_cuda_environment()
    bin_dir = str(site / "cublas" / "bin")
    assert added.count(bin_dir) == 2
    assert os.environ["PATH"].split(os.pathsep).count(bin_dir) == 1
    assert os.environ["PATH"].endswith("C:\\existing")
