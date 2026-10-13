import subprocess
from pathlib import Path

import pytest

from transcribe_media import hardware
from transcribe_media.hardware import (
    GpuInfo,
    detect_nvidia_gpus,
    free_disk_gb,
    parse_nvidia_smi,
    total_ram_gb,
)


def test_parse_rtx5080_row():
    gpus = parse_nvidia_smi("0, NVIDIA GeForce RTX 5080, 16303, 12.0, 576.80\n")
    assert gpus == (GpuInfo(0, "NVIDIA GeForce RTX 5080", 16303, (12, 0), "576.80"),)
    assert gpus[0].sm_arch == "sm_120"
    assert not gpus[0].int8_supported


def test_parse_non_blackwell_supports_int8():
    (gpu,) = parse_nvidia_smi("0, NVIDIA GeForce RTX 3070 Ti, 8192, 8.6, 552.22")
    assert gpu.compute_capability == (8, 6)
    assert gpu.int8_supported


def test_parse_name_containing_comma_and_multiple_gpus():
    text = "0, Weird GPU, Rev 2, 8000, 8.9, 550.1\n1, NVIDIA RTX A4000, 16376, 8.6, 550.1\n"
    gpus = parse_nvidia_smi(text)
    assert [g.name for g in gpus] == ["Weird GPU, Rev 2", "NVIDIA RTX A4000"]
    assert [g.index for g in gpus] == [0, 1]


def test_parse_legacy_output_infers_blackwell_from_name():
    (gpu,) = parse_nvidia_smi("0, NVIDIA GeForce RTX 5090, 32607, 570.00", has_compute_cap=False)
    assert gpu.compute_capability == (12, 0)
    assert not gpu.int8_supported


def test_parse_legacy_unknown_architecture_stays_unknown_and_allows_int8():
    (gpu,) = parse_nvidia_smi("0, NVIDIA GeForce GTX 1080, 8192, 470.1", has_compute_cap=False)
    assert gpu.compute_capability is None
    assert gpu.int8_supported


def test_parse_skips_garbage_and_na_driver():
    text = "garbage line\n\n0, NVIDIA GeForce RTX 4090, 24564, 8.9, [N/A]\nx, y\n"
    (gpu,) = parse_nvidia_smi(text)
    assert gpu.driver is None


def test_detect_returns_empty_without_nvidia_smi():
    def runner(cmd):
        raise FileNotFoundError("nvidia-smi")

    assert detect_nvidia_gpus(runner) == ()


def test_detect_falls_back_to_legacy_query_when_compute_cap_unsupported():
    calls = []

    def runner(cmd):
        calls.append(cmd[1])
        if "compute_cap" in cmd[1]:
            raise subprocess.CalledProcessError(2, cmd)
        return "0, NVIDIA GeForce RTX 3060, 12288, 535.1\n"

    (gpu,) = detect_nvidia_gpus(runner)
    assert gpu.vram_mb == 12288
    assert len(calls) == 2


def test_best_gpu_is_largest_vram():
    hw = hardware.Hardware(
        "Linux", "x86_64", 8, 32.0, 100.0,
        (GpuInfo(0, "small", 4000), GpuInfo(1, "big", 24000)),
    )
    assert hw.best_gpu.name == "big"


def test_apple_silicon_flag():
    hw = hardware.Hardware("Darwin", "arm64", 8, 16.0, 100.0)
    assert hw.apple_silicon and hw.best_gpu is None


def test_total_ram_macos_uses_sysctl():
    assert total_ram_gb("Darwin", runner=lambda cmd: "17179869184\n") == 16.0


def test_total_ram_unknown_system_is_zero():
    assert total_ram_gb("Plan9") == 0.0


def test_free_disk_uses_nearest_existing_parent(tmp_path: Path):
    assert free_disk_gb(tmp_path / "does" / "not" / "exist") > 0


# --- CUDA_VISIBLE_DEVICES: nvidia-smi ignores it, CUDA does not ---------------------- #

def gpus(count):
    return tuple(GpuInfo(i, f"GPU {i}", 8000 + i, (8, 9), "550") for i in range(count))


def test_no_restriction_leaves_every_gpu_visible():
    assert hardware.apply_visible_devices(gpus(2), None) == gpus(2)


@pytest.mark.parametrize("visible", ["", "-1", "  "])
def test_an_empty_or_negative_setting_hides_every_gpu(visible):
    assert hardware.apply_visible_devices(gpus(2), visible) == ()


def test_visible_gpus_are_renumbered_the_way_cuda_numbers_them():
    (gpu,) = hardware.apply_visible_devices(gpus(3), "2")
    assert (gpu.name, gpu.index) == ("GPU 2", 0)
    first, second = hardware.apply_visible_devices(gpus(3), "2,0")
    assert [(g.name, g.index) for g in (first, second)] == [("GPU 2", 0), ("GPU 0", 1)]


def test_devices_that_do_not_exist_are_ignored():
    assert hardware.apply_visible_devices(gpus(2), "5") == ()


@pytest.mark.parametrize("visible", ["GPU-8f3a2c1e", "MIG-abc/1/0"])
def test_forms_that_cannot_be_mapped_leave_the_list_alone(visible):
    assert hardware.apply_visible_devices(gpus(2), visible) == gpus(2)


def test_detection_honours_the_environment(monkeypatch, tmp_path):
    def runner(cmd):
        return "0, NVIDIA GeForce RTX 5080, 16303, 576.80, 12.0\n"

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert hardware.detect_hardware(tmp_path, runner=runner).gpus == ()
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert len(hardware.detect_hardware(tmp_path, runner=runner).gpus) == 1
