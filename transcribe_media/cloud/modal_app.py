"""Modal definitions for the cloud GPU backend.

Imported only when that backend is used, and by Modal inside the container,
which finds ``run_remote`` here by name. Everything the container needs is in
the image: this package's source, FFmpeg, faster-whisper, CUDA 12.8 libraries
and the speaker-label stack.
"""

from __future__ import annotations

from pathlib import Path

import modal

from .settings import (
    APP_NAME,
    DEFAULT_GPUS,
    JOBS_DIR,
    MAX_JOB_SECONDS,
    MOUNT,
    TORCH_INDEX,
    VOLUME_NAME,
    valid_job_id,
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .env({
        "HF_HOME": f"{MOUNT}/hf",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYANNOTE_METRICS_ENABLED": "0",  # pyannote's usage telemetry is on by default
        "TRANSCRIBE_MEDIA_REMOTE": "1",  # messages here must not suggest local fixes
        "PYTHONUNBUFFERED": "1",
    })
    # PyTorch alone from its own index; everything else from PyPI (never mixed).
    .uv_pip_install("torch>=2.8,<2.12", "torchaudio>=2.8,<2.12", index_url=TORCH_INDEX)
    .uv_pip_install(
        "faster-whisper>=1.2.1,<2",
        "huggingface-hub>=0.28,<3",
        "nvidia-cublas-cu12>=12.8,<13",
        "nvidia-cudnn-cu12>=9,<10",
        "pyannote.audio>=4.0.7,<5",
    )
    .add_local_python_source("transcribe_media")
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu=DEFAULT_GPUS,
    timeout=MAX_JOB_SECONDS,
    volumes={MOUNT: volume},
    max_containers=1,  # one container per run (each run is its own ephemeral app, so separate runs can overlap)
    retries=0,  # a failed job must not silently bill twice
)
def run_remote(job_id: str) -> dict:
    """Run the uploaded job ``job_id`` on the rented GPU."""
    from transcribe_media.cloud.remote_job import execute

    if not valid_job_id(job_id):
        raise ValueError("invalid job id")
    status = execute(Path(MOUNT) / JOBS_DIR / job_id)
    volume.commit()  # persist results and the model cache for the next run
    return status
