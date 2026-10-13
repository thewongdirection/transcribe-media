"""Error types. Each carries the process exit code, which is a stable contract."""

from __future__ import annotations


class TranscribeMediaError(RuntimeError):
    """Base class; ``exit_code`` is what the CLI exits with."""

    exit_code = 1


class InputError(TranscribeMediaError):
    exit_code = 2


class FfmpegMissingError(TranscribeMediaError):
    exit_code = 3


class AudioError(TranscribeMediaError):
    exit_code = 5


class TranscribeError(TranscribeMediaError):
    exit_code = 6


class DiarizeError(TranscribeMediaError):
    exit_code = 7


class WriteError(TranscribeMediaError):
    exit_code = 9


class NoAudioStreamError(TranscribeMediaError):
    exit_code = 10


class OverwriteRefusedError(TranscribeMediaError):
    exit_code = 11


class BadRecordingError(TranscribeMediaError):
    exit_code = 12


class DownloadError(TranscribeMediaError):
    exit_code = 13


class ModelError(TranscribeMediaError):
    exit_code = 14


def from_exit_code(code: int, message: str) -> TranscribeMediaError:
    """Recreate an error raised on another machine, keeping its exit code."""
    for cls in TranscribeMediaError.__subclasses__():
        if cls.exit_code == code:
            return cls(message)
    return TranscribeMediaError(message)
