"""Run the job on this machine."""

from __future__ import annotations

from pathlib import Path

from ..console import Log
from ..job import JobOptions, JobResult, run_job


class LocalBackend:
    name = "local"

    def run(self, wav: Path, options: JobOptions, log: Log) -> JobResult:
        return run_job(wav, options, log, backend=self.name)
