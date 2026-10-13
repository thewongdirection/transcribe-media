"""Where a transcription job runs: on this machine, or on a rented cloud GPU.

Nothing leaves the machine unless the user asks for it: ``local`` is the
default, and ``modal`` / ``auto`` are explicit opt-ins to upload the audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..console import Log
from ..environment import ORDINARY_MACHINE, Environment
from ..errors import TranscribeError
from ..job import JobOptions, JobResult

BACKENDS = ("local", "auto", "modal")
DEFAULT_MIN_CLOUD_MINUTES = 10.0


class Backend(Protocol):
    name: str

    def run(self, wav: Path, options: JobOptions, log: Log) -> JobResult: ...


@dataclass(frozen=True)
class BackendChoice:
    name: str  # "local" or "modal"
    note: str | None = None  # shown to the user: why, or a suggestion


def choose_backend(
    requested: str,
    *,
    local_gpu: bool,
    duration_s: float | None,
    cloud_problem: str | None,
    min_cloud_minutes: float = DEFAULT_MIN_CLOUD_MINUTES,
    environment: Environment = ORDINARY_MACHINE,
) -> BackendChoice:
    """Decide where to run.

    ``cloud_problem`` is None when a cloud GPU is usable, otherwise why not.
    ``local`` never uploads (it may suggest the cloud); ``modal`` demands it;
    ``auto`` uses a local GPU if there is one, a cloud GPU for long audio
    otherwise, and falls back to the CPU rather than failing.
    """
    long_enough = duration_s is None or duration_s >= min_cloud_minutes * 60
    where = f" ({environment.name})" if environment.name else ""

    if requested == "modal":
        if cloud_problem:
            raise TranscribeError(f"--backend modal is not available: {cloud_problem}")
        return BackendChoice("modal")
    if requested == "auto":
        if local_gpu:
            return BackendChoice("local")
        if not long_enough:
            return BackendChoice("local")
        if cloud_problem is None:
            return BackendChoice("modal", f"No GPU on this machine{where}: using a cloud GPU.")
        return BackendChoice(
            "local", f"No GPU on this machine{where} and no cloud GPU is configured ({cloud_problem}); using the CPU.")
    if requested == "local":
        if not local_gpu and long_enough and cloud_problem is None:
            return BackendChoice(
                "local", f"No GPU on this machine{where}, so this runs on the CPU. "
                "--backend auto would use a cloud GPU instead (it uploads the audio).")
        return BackendChoice("local")
    raise TranscribeError(f"unknown backend {requested!r} (choose from {', '.join(BACKENDS)})")
