"""FFmpeg helpers: stream check, duration, and decoding to Whisper's input format."""

from __future__ import annotations

import json
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .console import clean_text
from .errors import AudioError, FfmpegMissingError, NoAudioStreamError

if TYPE_CHECKING:
    import numpy as np

TARGET_SR = 16000
# Only plain files and pipes: a crafted playlist must not make FFmpeg fetch URLs.
_PROTOCOLS = "file,pipe"
# Containers that are really playlists or live descriptions: they point at *other* media (possibly
# other local files) instead of holding audio, so they are never decoded.
_INDIRECT_FORMATS = ("hls", "applehttp", "concat", "ffconcat", "dash", "sdp", "rtsp", "rtp", "lavfi", "image2")
_LOUDNORM = "loudnorm=I=-16:TP=-1.5:LRA=11"
_INSTALL_HINT = (
    "Install FFmpeg (it provides ffmpeg and ffprobe): Windows `winget install ffmpeg`, "
    "Debian/Ubuntu `sudo apt install ffmpeg`, macOS `brew install ffmpeg`."
)


@dataclass(frozen=True)
class MediaInfo:
    duration: float | None  # seconds, as declared by the container
    format_name: str


def require_ffmpeg() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise FfmpegMissingError(f"{' and '.join(missing)} not found on PATH. {_INSTALL_HINT}")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace", stdin=subprocess.DEVNULL)


def _probe_json(path: Path, entries: str, *extra: str) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-protocol_whitelist", _PROTOCOLS,
        *extra, "-show_entries", entries, "-of", "json", "-i", str(path),
    ]
    result = _run(cmd)
    if result.returncode != 0:
        detail = clean_text(result.stderr.strip() or "unreadable", 300)
        raise AudioError(f"cannot read {clean_text(path.name, 100)} as media: {detail}")
    try:
        return json.loads(result.stdout or "{}")
    except ValueError as exc:
        raise AudioError(f"ffprobe returned unreadable output for {clean_text(path.name, 100)}") from exc


def _as_seconds(value: object) -> float | None:
    try:
        seconds = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 else None


def ensure_audio_stream(path: Path) -> MediaInfo:
    """Raise unless ``path`` is readable audio-bearing media; returns what the container declares."""
    data = _probe_json(path, "stream=codec_name:format=format_name,duration", "-select_streams", "a:0")
    format_name = str((data.get("format") or {}).get("format_name", ""))
    if any(name in _INDIRECT_FORMATS for name in format_name.split(",")):
        raise AudioError(
            f"{clean_text(path.name, 100)} is a playlist or stream description ({format_name}), not a media file"
        )
    if not data.get("streams"):
        raise NoAudioStreamError(f"{clean_text(path.name, 100)} has no audio stream")
    return MediaInfo(_as_seconds((data.get("format") or {}).get("duration")), format_name)


def probe_duration(path: Path) -> float | None:
    try:
        return _as_seconds((_probe_json(path, "format=duration").get("format") or {}).get("duration"))
    except AudioError:
        return None


def load_samples(path: Path) -> np.ndarray:
    """Read a WAV made by :func:`decode_to_wav` as float32 samples in [-1, 1].

    Models receive this array directly instead of a file path, so decoding
    never depends on faster-whisper's PyAV call (whose keyword arguments broke
    with recent PyAV releases).
    """
    import numpy as np

    try:
        with wave.open(str(path), "rb") as wav:
            layout = (wav.getsampwidth(), wav.getnchannels(), wav.getframerate())
            raw = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError, OSError) as exc:
        raise AudioError(f"cannot read decoded audio {path.name}: {exc}") from exc
    if layout != (2, 1, TARGET_SR):
        raise AudioError(f"expected 16-bit mono {TARGET_SR} Hz WAV, got (width, channels, rate)={layout}")
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def decode_to_wav(src: Path, dst: Path, *, normalize: bool = False, max_seconds: float | None = None) -> None:
    """Decode any media to 16 kHz mono 16-bit PCM WAV, dropping video.

    ``max_seconds`` hard-caps the output length, so a small file that decodes to
    an enormous one (hours of silence) cannot fill the disk or memory.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-protocol_whitelist", _PROTOCOLS, "-i", str(src), "-vn",
    ]
    if normalize:
        cmd += ["-af", _LOUDNORM]
    if max_seconds is not None:
        cmd += ["-t", f"{max_seconds:.3f}"]
    cmd += ["-ac", "1", "-ar", str(TARGET_SR), "-c:a", "pcm_s16le", str(dst)]
    result = _run(cmd)
    if result.returncode != 0:
        raise AudioError(f"ffmpeg could not decode {clean_text(src.name, 100)}: {clean_text(result.stderr, 300)}")


def encode_flac(wav: Path, dst: Path) -> None:
    """Losslessly compress ``wav`` (about half the size) for upload."""
    result = _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                   "-protocol_whitelist", _PROTOCOLS, "-i", str(wav), "-c:a", "flac", str(dst)])
    if result.returncode != 0:
        raise AudioError(f"ffmpeg could not compress {wav.name}: {clean_text(result.stderr, 300)}")
