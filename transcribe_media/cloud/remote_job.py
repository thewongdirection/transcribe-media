"""The part of a cloud job that runs on the rented machine.

Plain Python with no cloud-provider imports, so it can be tested locally. The
provider-specific wrapper (``modal_app``) calls :func:`execute` inside the
container with the directory the client uploaded to.

Layout of a job directory:

* ``audio.flac``, ``options.json``  - written by the client
* ``result.json`` or ``error.json`` - written here
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from .. import audio, job
from ..console import Console, Log, RecordingLog, clean_text
from ..errors import InputError, TranscribeError, TranscribeMediaError

AUDIO_FILE = "audio.flac"
OPTIONS_FILE = "options.json"
RESULT_FILE = "result.json"
ERROR_FILE = "error.json"


def _fail(job_dir: Path, exit_code: int, message: str) -> dict[str, Any]:
    (job_dir / ERROR_FILE).write_text(json.dumps({"message": message, "exit_code": exit_code}), encoding="utf-8")
    return {"ok": False}


def execute(job_dir: Path, log: Log | None = None) -> dict[str, Any]:
    """Run the job in ``job_dir``; returns a small status dict, details go in files."""
    recorder = RecordingLog(log or Console())
    try:
        try:
            options = job.JobOptions.from_dict(json.loads((job_dir / OPTIONS_FILE).read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:  # includes malformed JSON and rejected option values
            raise InputError(f"invalid job options: {clean_text(exc)}") from exc
        with tempfile.TemporaryDirectory(prefix="transcribe-media-") as tmp:
            wav = Path(tmp) / "audio.wav"
            audio.decode_to_wav(job_dir / AUDIO_FILE, wav)
            (job_dir / AUDIO_FILE).unlink(missing_ok=True)  # decoded: do not keep the upload on the provider's disk
            result = job.run_job(wav, options, recorder, backend="modal")
        result.warnings = [clean_text(w, 300) for w in recorder.warnings]
        (job_dir / RESULT_FILE).write_text(json.dumps(result.to_dict(), ensure_ascii=False), encoding="utf-8")
        return {"ok": True}
    except TranscribeMediaError as exc:
        return _fail(job_dir, exc.exit_code, clean_text(exc, 600))
    except Exception as exc:  # whatever happens, report it in a file instead of a bare crash
        return _fail(job_dir, TranscribeError.exit_code,
                     f"unexpected error in the cloud job: {type(exc).__name__}: {clean_text(exc, 600)}")
    finally:
        (job_dir / AUDIO_FILE).unlink(missing_ok=True)
