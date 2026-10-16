"""Accuracy-first selection of the Whisper model, device and precision.

Speed is deliberately a low priority: the largest model that fits is always
chosen, and quantization is used only when memory forces it. The function is
pure so every tier can be unit-tested without hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import InputError
from .hardware import GpuInfo, Hardware

DEFAULT_MODEL = "large-v3"
INT8_FAMILY = ("int8", "int8_float16", "int8_float32", "int8_bfloat16")
# CTranslate2 offers these on a GPU only; the CPU has float32, int16, int8 and int8_float32.
GPU_ONLY_COMPUTE_TYPES = ("float16", "int8_float16", "bfloat16", "int8_bfloat16")

# VRAM needed (MB) with headroom, from faster-whisper's published large-v2
# figures (large-v3 is the same size): ~4.5 GB fp16, ~2.9 GB int8.
_LARGE_FP16_MB = 6000
_LARGE_INT8_MB = 3500
_MEDIUM_MB = 2500
# System RAM thresholds (GB) for CPU inference.
_CPU_LARGE_GB = 8
_CPU_MEDIUM_GB = 4


@dataclass(frozen=True)
class AsrProfile:
    model: str
    device: str  # "cuda" | "cpu"
    compute_type: str
    device_index: int = 0
    notes: tuple[str, ...] = ()


def _gpu_choice(gpu: GpuInfo) -> tuple[str, str, tuple[str, ...]] | None:
    """(model, compute_type, notes) for ``gpu``, or None if it is too small."""
    vram = gpu.vram_mb
    if vram >= _LARGE_FP16_MB:
        return DEFAULT_MODEL, "float16", ()
    if vram >= _LARGE_INT8_MB and gpu.int8_supported:
        return DEFAULT_MODEL, "int8_float16", (
            f"{gpu.name} has {vram} MB VRAM: using INT8 to fit {DEFAULT_MODEL}.",
        )
    if vram >= _MEDIUM_MB:
        compute = "int8_float16" if gpu.int8_supported else "float16"
        return "medium", compute, (
            f"{gpu.name} has only {vram} MB VRAM: using 'medium', which is less "
            f"accurate than {DEFAULT_MODEL}.",
        )
    return None


def _cpu_choice(hw: Hardware) -> tuple[str, str, tuple[str, ...]]:
    if hw.ram_gb >= _CPU_LARGE_GB:
        model = DEFAULT_MODEL
    elif hw.ram_gb >= _CPU_MEDIUM_GB:
        model = "medium"
    else:
        model = "small"
    notes = ["Running on the CPU: transcription will be slow, but accuracy is unaffected."]
    if model != DEFAULT_MODEL:
        notes.append(
            f"Only {hw.ram_gb:.1f} GB RAM: using '{model}', which is less accurate "
            f"than {DEFAULT_MODEL}."
        )
    if hw.apple_silicon:
        notes.append("CTranslate2 has no Apple GPU backend, so Apple Silicon runs on the CPU.")
    return model, "int8", tuple(notes)


def select_asr_profile(
    hw: Hardware,
    *,
    model: str | None = None,
    device: str = "auto",
    compute_type: str = "auto",
) -> AsrProfile:
    """Choose the ASR configuration for ``hw``, honouring explicit overrides."""
    if device not in ("auto", "cuda", "cpu"):
        raise InputError(f"unknown device {device!r} (choose auto, cuda or cpu)")

    gpu = hw.best_gpu if device != "cpu" else None
    choice = _gpu_choice(gpu) if gpu else None
    if device == "cuda" and gpu is None:
        raise InputError("--device cuda was requested but no NVIDIA GPU was detected")
    if device == "cuda" and choice is None and gpu is not None:
        raise InputError(f"--device cuda was requested but {gpu.name} ({gpu.vram_mb} MB) is too small for any model")

    if choice and gpu:
        auto_model, auto_compute, notes = choice
        chosen = AsrProfile(auto_model, "cuda", auto_compute, gpu.index, notes)
    else:
        auto_model, auto_compute, notes = _cpu_choice(hw)
        if gpu and device == "auto":
            notes = (f"{gpu.name} ({gpu.vram_mb} MB) is too small for GPU inference.", *notes)
        chosen = AsrProfile(auto_model, "cpu", auto_compute, 0, notes)

    final_model = model or chosen.model
    final_compute = chosen.compute_type if compute_type == "auto" else compute_type
    if chosen.device == "cuda" and gpu and not gpu.int8_supported and final_compute in INT8_FAMILY:
        raise InputError(
            f"--compute-type {final_compute} is not supported on {gpu.name} "
            f"({gpu.sm_arch}); CTranslate2 disables INT8 on Blackwell. Use float16."
        )
    return AsrProfile(final_model, chosen.device, final_compute, chosen.device_index, chosen.notes)
