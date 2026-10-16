"""Environment diagnostics for ``--check``: what works, what is missing, how to fix it."""

from __future__ import annotations

import importlib.metadata
import platform
import shutil
import sys
from dataclasses import dataclass
from typing import Callable

from . import audio, diarize, models
from .cuda_libs import configure_cuda_environment
from .environment import ORDINARY_MACHINE, Environment
from .hardware import Hardware
from .profiles import AsrProfile

OK, WARN, FAIL = "ok", "warn", "fail"
MIN_PYTHON = (3, 10)
# First CTranslate2 release with CUDA 12.8 support, which Blackwell GPUs need.
MIN_CTRANSLATE2_BLACKWELL = (4, 6, 3)


@dataclass
class Check:
    name: str
    status: str
    detail: str
    remedy: str = ""


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _as_tuple(version: str) -> tuple[int, ...]:
    parts = []
    for piece in version.split(".")[:3]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def check_python() -> Check:
    found = ".".join(map(str, sys.version_info[:3]))
    if sys.version_info[:2] >= MIN_PYTHON:
        return Check("Python", OK, found)
    return Check("Python", FAIL, f"{found} is too old", "Python 3.10+ is required; run scripts/setup.py")


def check_ffmpeg() -> Check:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if not missing:
        return Check("FFmpeg", OK, f"found at {shutil.which('ffmpeg')}")
    try:
        audio.require_ffmpeg()
    except Exception as exc:
        return Check("FFmpeg", FAIL, f"{' and '.join(missing)} not on PATH", str(exc))
    return Check("FFmpeg", FAIL, "missing")  # pragma: no cover


def check_hardware(hw: Hardware, profile: AsrProfile) -> list[Check]:
    gpu = hw.best_gpu
    if gpu:
        arch = f", {gpu.sm_arch}" if gpu.sm_arch else ""
        gpu_text = f"{gpu.name}, {gpu.vram_mb / 1024:.0f} GB VRAM{arch}, driver {gpu.driver or 'unknown'}"
    else:
        gpu_text = "no NVIDIA GPU detected"
    machine = f"{hw.ram_gb:.0f} GB RAM, {hw.cpu_threads} CPU threads, {hw.free_disk_gb:.0f} GB free for models"
    chosen = f"{profile.model} / {profile.compute_type} on {profile.device}"
    checks = [Check("Hardware", OK, f"{gpu_text}; {machine}"), Check("Selected model", OK, chosen)]
    checks += [Check("Note", WARN, note) for note in profile.notes]
    return checks


def check_asr_packages(hw: Hardware) -> list[Check]:
    fw, ct2 = _version("faster-whisper"), _version("ctranslate2")
    if not fw or not ct2:
        return [Check("faster-whisper", FAIL, "not installed", "run scripts/setup.py")]
    checks = [Check("faster-whisper", OK, f"{fw} (ctranslate2 {ct2})")]
    gpu = hw.best_gpu
    if gpu and not gpu.int8_supported and _as_tuple(ct2) < MIN_CTRANSLATE2_BLACKWELL:
        checks.append(Check(
            "CTranslate2 for Blackwell", FAIL, f"{ct2} predates CUDA 12.8 support",
            "install ctranslate2 >= 4.6.3 (re-run scripts/setup.py)"))
    return checks


def check_cuda(hw: Hardware, device_count: Callable[[], int]) -> Check | None:
    if not hw.gpus:
        return None
    failures = configure_cuda_environment()
    count = device_count()
    if count > 0 and not failures:
        return Check("CUDA runtime", OK, f"{count} CUDA device(s) usable by CTranslate2")
    detail = f"CTranslate2 sees {count} CUDA device(s)" + (f"; {len(failures)} library load error(s)" if failures else "")
    return Check("CUDA runtime", FAIL, detail, "re-run scripts/setup.py to install CUDA 12.8 cuBLAS/cuDNN; use --device cpu meanwhile")


def check_model_cache(profile: AsrProfile) -> Check:
    spec = models.WHISPER_MODELS.get(profile.model)
    if spec is None:
        return Check("Model files", OK, f"custom model at {profile.model}")
    if models.cached_path(spec):
        return Check("Model files", OK, f"{profile.model} is cached")
    return Check("Model files", WARN, f"{profile.model} not downloaded yet (~{spec.size_gb:.1f} GB, fetched on first run)")


def check_ytdlp() -> list[Check]:
    version = _version("yt-dlp")
    if not version:
        return [Check("yt-dlp", FAIL, "not installed", "run scripts/setup.py (needed for URLs)")]
    runtime = next((name for name in ("deno", "node", "bun", "qjs") if shutil.which(name)), None)
    checks = [Check("yt-dlp", OK, version)]
    if runtime:
        checks.append(Check("JavaScript runtime", OK, f"{runtime} (used by yt-dlp for YouTube)"))
    else:
        checks.append(Check(
            "JavaScript runtime", WARN, "none found",
            "YouTube audio downloads may fail; install Deno (https://deno.com) or Node.js 22+"))
    return checks


def check_diarization(offline: bool) -> Check:
    setup = diarize.prepare_diarization(offline=offline, allow_download=True, dry_run=True)
    if setup.available:
        return Check("Speaker labels", OK, "ready (model cached; no token needed)")
    if setup.will_download:
        return Check("Speaker labels", OK, "ready (model will be downloaded on first use)")
    return Check("Speaker labels", WARN, f"will be skipped: {setup.skip_reason}")


def check_environment(env: Environment) -> Check | None:
    if not env.is_cloud:
        return None
    detail = f"running in {env.name}"
    if env.ephemeral:
        detail += "; the environment and downloaded models are rebuilt each session, so the first run is slower"
    return Check("Environment", OK, detail)


def check_cloud(hw: Hardware, problem: str | None) -> Check:
    """Whether a rented cloud GPU can be used (optional; uploading audio is always opt-in)."""
    if problem is None:
        return Check("Cloud GPU", OK, "Modal credentials found; --backend auto or modal can use a rented GPU "
                                      "(this uploads the audio)")
    # Only worth a warning when there is no local GPU to fall back on.
    status = OK if hw.gpus else WARN
    return Check("Cloud GPU (optional)", status, f"not set up: {problem}",
                 "for much faster transcription without a local GPU, see 'Cloud GPU' in the README")


def run_checks(
    hw: Hardware,
    profile: AsrProfile,
    *,
    offline: bool,
    device_count: Callable[[], int],
    environment: Environment = ORDINARY_MACHINE,
    cloud_problem: str | None = None,
) -> list[Check]:
    checks = [Check("Platform", OK, f"{platform.system()} {platform.machine()}"), check_python(), check_ffmpeg()]
    env_check = check_environment(environment)
    if env_check:
        checks.append(env_check)
    checks += check_hardware(hw, profile)
    checks += check_asr_packages(hw)
    cuda = check_cuda(hw, device_count)
    if cuda:
        checks.append(cuda)
    checks.append(check_model_cache(profile))
    checks += check_ytdlp()
    checks.append(check_diarization(offline))
    checks.append(check_cloud(hw, cloud_problem))
    return checks


def format_report(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks)
    lines = ["transcribe-media environment check", "=" * 34]
    for c in checks:
        lines.append(f"[{c.status.upper():4}] {c.name.ljust(width)}  {c.detail}")
        if c.remedy and c.status != OK:
            lines.append(f"       -> {c.remedy}")
    failed = any(c.status == FAIL for c in checks)
    lines.append("")
    lines.append("Result: " + ("problems found (see FAIL lines)." if failed else "ready to transcribe."))
    return "\n".join(lines) + "\n"


def has_failures(checks: list[Check]) -> bool:
    return any(c.status == FAIL for c in checks)
