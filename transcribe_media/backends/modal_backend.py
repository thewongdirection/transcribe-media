"""Run the job on a rented GPU through Modal.

What is sent: the decoded 16 kHz audio (lossless FLAC) and the transcription
options (which include any ``--prompt`` text), to a private volume in the user's
own Modal account. The job deletes the audio as soon as it is decoded, and the
client deletes the whole job directory afterwards. The user's local Hugging Face
token is never sent; speaker labels in the cloud use a Modal Secret the user
created themselves (``--cloud-hf-secret``).

Cost control has two layers, because a client-side timer dies with the client:
the client cancels the call after ``max_minutes`` (and on Ctrl-C, SIGTERM or
SIGHUP), and the function itself carries a server-side timeout of the same
length plus a small margin.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import signal
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

from .. import audio
from ..cloud import settings
from ..cloud.remote_job import AUDIO_FILE, ERROR_FILE, OPTIONS_FILE, RESULT_FILE
from ..console import Log, clean_text
from ..errors import InputError, TranscribeError, TranscribeMediaError, from_exit_code
from ..job import JobOptions, JobResult

AUTH_HELP = (
    "run `modal token new`, or set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in the environment "
    "(in a cloud session add them in the environment's settings; never paste them into a chat)"
)
INSTALL_HELP = "the Modal client is not installed (run `python scripts/setup.py --cloud`)"
DEFAULT_MAX_MINUTES = 120
# The function's own timeout is the user's cap plus this, so the client normally cancels first.
SERVER_TIMEOUT_MARGIN_SECONDS = 120


def cloud_problem(modal: ModuleType | None = None) -> str | None:
    """Why the cloud GPU backend cannot be used here, or None if it can."""
    try:
        modal = modal or importlib.import_module("modal")
    except ImportError:
        return INSTALL_HELP
    config = modal.config.config
    if not (config.get("token_id") and config.get("token_secret")):
        return f"no Modal credentials: {AUTH_HELP}"
    return None


def load_cloud_app() -> Any:
    try:
        return importlib.import_module("transcribe_media.cloud.modal_app")
    except ImportError as exc:
        raise TranscribeError(f"{INSTALL_HELP} ({clean_text(exc)})") from exc


@contextlib.contextmanager
def _stop_on_termination() -> Iterator[None]:
    """Turn SIGTERM/SIGHUP into KeyboardInterrupt so a killed process still cancels its paid GPU.

    Only the main thread may install handlers; elsewhere this does nothing.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def interrupt(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    signals = [getattr(signal, name) for name in ("SIGTERM", "SIGHUP") if hasattr(signal, name)]
    previous = {sig: signal.signal(sig, interrupt) for sig in signals}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def validate_settings(gpu: str | None, max_minutes: int) -> None:
    """Reject unusable cloud settings (shared with the CLI, which checks them before any work starts)."""
    if gpu is not None and gpu not in settings.ALLOWED_GPUS:
        raise InputError(f"unsupported --cloud-gpu {gpu!r} (choose from {', '.join(settings.ALLOWED_GPUS)})")
    if not 1 <= max_minutes <= settings.MAX_JOB_SECONDS // 60:
        raise InputError(f"--cloud-max-minutes must be between 1 and {settings.MAX_JOB_SECONDS // 60}")


class ModalBackend:
    name = "modal"

    def __init__(
        self,
        *,
        gpu: str | None = None,
        max_minutes: int = DEFAULT_MAX_MINUTES,
        hf_secret: str | None = None,
        quiet: bool = False,
        cloud: Any = None,
    ):
        validate_settings(gpu, max_minutes)
        self.gpu, self.max_minutes, self.hf_secret, self.quiet = gpu, max_minutes, hf_secret, quiet
        self._cloud = cloud

    def run(self, wav: Path, options: JobOptions, log: Log) -> JobResult:
        cloud = self._cloud or load_cloud_app()
        job_id = uuid.uuid4().hex
        self._purge_stale(cloud)
        try:
            with tempfile.TemporaryDirectory(prefix="transcribe-media-", ignore_cleanup_errors=True) as tmp:
                flac = Path(tmp) / AUDIO_FILE
                audio.encode_flac(wav, flac)
                log.notice(
                    f"Uploading the audio (~{flac.stat().st_size / 1e6:.0f} MB) and the transcription options to "
                    "your Modal account to transcribe it on a cloud GPU. The audio is deleted when the job ends."
                )
                self._upload(cloud, job_id, flac, options)
            status = self._call(cloud, job_id, log)
            result = self._download(cloud, job_id, status)
        except TranscribeMediaError:
            raise
        except cloud.modal.exception.AuthError as exc:
            raise TranscribeError(f"Modal rejected the credentials: {AUTH_HELP}") from exc
        except Exception as exc:  # any provider failure: network, build, quota, container crash
            raise TranscribeError(f"the cloud GPU job failed: {clean_text(exc, 400)}") from exc
        finally:
            self._cleanup(cloud, job_id, log)
        for warning in result.warnings:
            log.warn(f"[cloud] {clean_text(warning, 300)}")
        return result

    # -- steps -------------------------------------------------------------- #

    def _job_path(self, job_id: str, name: str = "") -> str:
        return f"/{settings.JOBS_DIR}/{job_id}" + (f"/{name}" if name else "")

    def _upload(self, cloud: Any, job_id: str, flac: Path, options: JobOptions) -> None:
        payload = json.dumps(options.to_dict()).encode("utf-8")
        with cloud.volume.batch_upload() as batch:
            batch.put_file(str(flac), self._job_path(job_id, AUDIO_FILE))
            batch.put_file(io.BytesIO(payload), self._job_path(job_id, OPTIONS_FILE))

    def _overrides(self, cloud: Any) -> dict[str, Any]:
        overrides: dict[str, Any] = {"timeout": self.max_minutes * 60 + SERVER_TIMEOUT_MARGIN_SECONDS}
        if self.gpu:
            overrides["gpu"] = self.gpu
        if self.hf_secret:
            overrides["secrets"] = [cloud.modal.Secret.from_name(self.hf_secret, required_keys=["HF_TOKEN"])]
        return overrides

    def _call(self, cloud: Any, job_id: str, log: Log) -> dict[str, Any]:
        modal = cloud.modal
        log.info("Starting a cloud GPU (the first run builds the environment and downloads the model, "
                 "so it takes a few extra minutes) ...")
        # Modal's progress display writes to stdout; keep stdout clean for the transcript.
        with contextlib.redirect_stdout(sys.stderr), \
                (contextlib.nullcontext() if self.quiet else modal.enable_output()), \
                cloud.app.run(), _stop_on_termination():
            call = cloud.run_remote.with_options(**self._overrides(cloud)).spawn(job_id)
            try:
                return call.get(timeout=self.max_minutes * 60)
            except (modal.exception.TimeoutError, TimeoutError) as exc:
                self._stop(call, log)
                raise TranscribeError(
                    f"the cloud job ran longer than --cloud-max-minutes={self.max_minutes} and was cancelled"
                ) from exc
            except BaseException:  # includes Ctrl-C and termination signals: never leave a paid GPU running
                self._stop(call, log)
                raise

    @staticmethod
    def _stop(call: Any, log: Log) -> None:
        try:
            call.cancel(terminate_containers=True)
        except Exception as exc:
            log.warn(
                f"could not cancel the cloud job ({clean_text(exc, 200)}); it may still be running and billing. "
                f"Stop the '{settings.APP_NAME}' app in the Modal dashboard (https://modal.com/apps)."
            )

    def _download(self, cloud: Any, job_id: str, status: dict[str, Any]) -> JobResult:
        ok = bool(status.get("ok"))
        raw = b"".join(cloud.volume.read_file(self._job_path(job_id, RESULT_FILE if ok else ERROR_FILE)))
        data = json.loads(raw)
        if not ok:
            raise from_exit_code(int(data.get("exit_code", 1)), clean_text(data.get("message", "the cloud job failed"), 600))
        return JobResult.from_dict(data)

    def _cleanup(self, cloud: Any, job_id: str, log: Log) -> None:
        try:
            cloud.volume.remove_file(self._job_path(job_id), recursive=True)
        except FileNotFoundError:
            pass  # already gone, which is the goal
        except Exception as exc:
            log.warn(
                f"could not delete job {job_id} from the Modal volume '{settings.VOLUME_NAME}' "
                f"({clean_text(exc, 200)}); remove it with "
                f"`modal volume rm {settings.VOLUME_NAME} {self._job_path(job_id)} -r`"
            )

    def _purge_stale(self, cloud: Any) -> None:
        """Remove job directories a crashed earlier run may have left behind."""
        try:
            cutoff = time.time() - settings.STALE_JOB_SECONDS
            for entry in cloud.volume.listdir(f"/{settings.JOBS_DIR}"):
                # An unknown (zero) modification time must never count as old: it could be a live job.
                if entry.mtime and entry.mtime < cutoff:
                    # Listings omit the leading slash that uploads and deletes use.
                    cloud.volume.remove_file("/" + entry.path.lstrip("/"), recursive=True)
        except Exception:  # housekeeping only: the directory may not exist yet
            pass
