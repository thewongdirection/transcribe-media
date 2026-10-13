"""Hardware detection: NVIDIA GPUs, RAM, CPU threads and free disk.

Standard library only, so ``scripts/setup.py`` can use it before any package is
installed. Parsing is separated from I/O so it can be tested without a GPU.
"""

from __future__ import annotations

import ctypes
import dataclasses
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

Runner = Callable[[Sequence[str]], str]

_NVIDIA_SMI_FIELDS = "index,name,memory.total,compute_cap,driver_version"
_NVIDIA_SMI_FIELDS_LEGACY = "index,name,memory.total,driver_version"
# Older drivers cannot report compute_cap; RTX 50-series / Blackwell workstation
# cards are sm_120, which matters because CTranslate2 disables INT8 there.
_BLACKWELL_NAME = re.compile(r"RTX\s*50\d{2}|RTX\s*PRO\s*\d+\s*Blackwell|Blackwell", re.I)
BLACKWELL_MAJOR = 12


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    vram_mb: int
    compute_capability: tuple[int, int] | None = None
    driver: str | None = None

    @property
    def int8_supported(self) -> bool:
        """CTranslate2 disables INT8 on sm_120 and newer (Blackwell consumer)."""
        cc = self.compute_capability
        return not (cc is not None and cc[0] >= BLACKWELL_MAJOR)

    @property
    def sm_arch(self) -> str | None:
        cc = self.compute_capability
        return f"sm_{cc[0]}{cc[1]}" if cc else None


@dataclass(frozen=True)
class Hardware:
    system: str
    machine: str
    cpu_threads: int
    ram_gb: float
    free_disk_gb: float
    gpus: tuple[GpuInfo, ...] = ()

    @property
    def apple_silicon(self) -> bool:
        return self.system == "Darwin" and self.machine == "arm64"

    @property
    def best_gpu(self) -> GpuInfo | None:
        return max(self.gpus, key=lambda g: g.vram_mb) if self.gpus else None


def _run(cmd: Sequence[str]) -> str:
    result = subprocess.run(
        list(cmd), capture_output=True, text=True, errors="replace", timeout=20, check=True
    )
    return result.stdout


def _parse_compute_cap(value: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"(\d+)\.(\d+)", value.strip())
    return (int(match.group(1)), int(match.group(2))) if match else None


def parse_nvidia_smi(csv_text: str, *, has_compute_cap: bool = True) -> tuple[GpuInfo, ...]:
    """Parse ``nvidia-smi --format=csv,noheader,nounits`` output.

    The GPU name is the only free-text field, so rows are split from both ends
    rather than on every comma.
    """
    gpus = []
    for line in csv_text.splitlines():
        line = line.strip()
        if not line:
            continue
        tail_fields = 3 if has_compute_cap else 2
        try:
            index_str, rest = line.split(",", 1)
            parts = [p.strip() for p in rest.rsplit(",", tail_fields)]
            name, vram = parts[0], parts[1]
            if has_compute_cap:
                cc, driver = _parse_compute_cap(parts[2]), parts[3]
            else:
                cc, driver = None, parts[2]
            vram_mb = int(float(vram))
            index = int(index_str)
        except (ValueError, IndexError):
            continue
        if cc is None and _BLACKWELL_NAME.search(name):
            cc = (BLACKWELL_MAJOR, 0)
        gpus.append(GpuInfo(index, name, vram_mb, cc, driver if driver != "[N/A]" else None))
    return tuple(gpus)


def detect_nvidia_gpus(runner: Runner = _run) -> tuple[GpuInfo, ...]:
    """Query ``nvidia-smi``; returns ``()`` when there is no NVIDIA driver."""
    for fields, has_cc in ((_NVIDIA_SMI_FIELDS, True), (_NVIDIA_SMI_FIELDS_LEGACY, False)):
        try:
            out = runner(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
        except (OSError, subprocess.SubprocessError):
            continue
        gpus = parse_nvidia_smi(out, has_compute_cap=has_cc)
        if gpus:
            return gpus
    return ()


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def total_ram_gb(system: str | None = None, runner: Runner = _run) -> float:
    """Total physical memory in GiB; 0.0 when it cannot be determined."""
    system = system or platform.system()
    try:
        if system == "Linux":
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024**2
        elif system == "Darwin":
            return int(runner(["sysctl", "-n", "hw.memsize"]).strip()) / 1024**3
        elif system == "Windows":
            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
            return status.ullTotalPhys / 1024**3
    except (OSError, ValueError, subprocess.SubprocessError, AttributeError):
        pass
    return 0.0


def free_disk_gb(path: Path) -> float:
    """Free space (GiB) on the filesystem holding ``path`` (nearest existing parent)."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1024**3


def apply_visible_devices(gpus: tuple[GpuInfo, ...], visible: str | None) -> tuple[GpuInfo, ...]:
    """Honour CUDA_VISIBLE_DEVICES, which nvidia-smi ignores.

    Device indices seen by CUDA are positions in the visible list, so the GPUs are
    renumbered. Forms that cannot be mapped (UUIDs, MIG names) leave the list alone.
    """
    if visible is None:
        return gpus
    visible = visible.strip()
    if visible in ("", "-1"):
        return ()
    try:
        wanted = [int(part) for part in visible.split(",")]
    except ValueError:
        return gpus
    by_index = {gpu.index: gpu for gpu in gpus}
    return tuple(dataclasses.replace(by_index[i], index=position)
                 for position, i in enumerate(wanted) if i in by_index)


def detect_hardware(cache_dir: Path, runner: Runner = _run) -> Hardware:
    """Probe the machine. ``cache_dir`` is where models will be stored."""
    return Hardware(
        system=platform.system(),
        machine=platform.machine().lower(),
        cpu_threads=os.cpu_count() or 1,
        ram_gb=total_ram_gb(runner=runner),
        free_disk_gb=free_disk_gb(cache_dir),
        gpus=apply_visible_devices(detect_nvidia_gpus(runner), os.environ.get("CUDA_VISIBLE_DEVICES")),
    )
