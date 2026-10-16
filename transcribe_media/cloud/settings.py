"""Constants shared by the client and the container; importable without Modal."""

from __future__ import annotations

import re

APP_NAME = "transcribe-media"
VOLUME_NAME = "transcribe-media-data"
MOUNT = "/data"
JOBS_DIR = "jobs"
# Tried in order. Cards that run large-v3 in float16 with plenty of headroom. The
# Blackwell data-center parts (B200/B300) are left out: Modal documents that B300
# needs CUDA 13.1+, while the image here targets CUDA 12.8.
DEFAULT_GPUS = ["L4", "A10", "L40S"]
ALLOWED_GPUS = ("T4", "L4", "A10", "L40S", "A100", "A100-40GB", "A100-80GB", "H100")
# Hard ceiling for any single job; the client enforces the user's own, lower cap.
MAX_JOB_SECONDS = 6 * 3600
# Job directories older than this are removed on the next run (a crash may leave one behind).
STALE_JOB_SECONDS = 24 * 3600
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"

_JOB_ID = re.compile(r"[0-9a-f]{32}")


def valid_job_id(job_id: object) -> bool:
    """Job ids are random UUID hex strings; anything else must never reach a path."""
    return isinstance(job_id, str) and bool(_JOB_ID.fullmatch(job_id))
