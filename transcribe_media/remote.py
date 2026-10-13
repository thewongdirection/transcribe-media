"""URL access through yt-dlp's Python API.

No shell, no plugins, no user config, and every connection is checked by the
network guard so redirects and embedded media cannot reach private addresses.
Browser cookies are never used unless the caller names a browser, which is an
explicit opt-in: cookies are credentials.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

from .console import clean_text
from .errors import DownloadError, InputError
from .netguard import network_guard

Factory = Callable[[dict], Any]  # yt_dlp.YoutubeDL-compatible constructor

# Plain names only: yt-dlp also accepts "browser:profile/path" forms we do not want.
BROWSERS = ("brave", "chrome", "chromium", "edge", "firefox", "opera", "safari", "vivaldi", "whale")

# Only protocols that yt-dlp downloads with its own (guarded) HTTP client. Others (ftp, rtmp, rtsp,
# mms ...) are handed to ffmpeg, whose sockets the network guard cannot see.
_SAFE_PROTOCOLS = "^(https?|m3u8_native|http_dash_segments)$"
AUDIO_FORMAT = f"bestaudio[protocol~='{_SAFE_PROTOCOLS}']/best[protocol~='{_SAFE_PROTOCOLS}']"
DOWNLOAD_DEADLINE_SECONDS = 3600
# yt-dlp also hands some downloads to ffmpeg (HLS with a feature it cannot handle itself, anything flagged
# live, ...), and ffmpeg's own sockets are invisible to the network guard. A path below a regular file can
# never exist, so pointing yt-dlp at one leaves it without ffmpeg: it then refuses those downloads.
NO_FFMPEG = os.path.join(__file__, "ffmpeg")

_LOGIN_SIGNS = ("sign in to confirm", "not a bot", "login_required", "http error 403")
_LOGIN_HINT = (
    "\n  -> YouTube is refusing anonymous access (common from cloud servers, VPNs and busy networks). "
    "Retry from a normal home connection, or pass --cookies-from-browser <browser> to reuse "
    "the session of a browser you are signed in to."
)


class _SilentLogger:
    """yt-dlp prints errors itself even when quiet; we raise our own instead."""

    def debug(self, msg: str) -> None: ...
    def info(self, msg: str) -> None: ...
    def warning(self, msg: str) -> None: ...
    def error(self, msg: str) -> None: ...


class _DownloadLimits:
    """A yt-dlp progress hook that aborts a download that gets too big or takes too long.

    yt-dlp's own ``max_filesize`` only works when the server states the size up front.
    """

    def __init__(self, max_bytes: int, deadline_seconds: float, clock: Callable[[], float] = time.monotonic):
        self.max_bytes, self._clock = max_bytes, clock
        self._deadline = clock() + deadline_seconds
        self._minutes = max(1, round(deadline_seconds / 60))

    def __call__(self, status: dict[str, Any]) -> None:
        if self._clock() > self._deadline:
            raise DownloadError(f"the download took longer than {self._minutes} minutes and was stopped")
        if (status.get("downloaded_bytes") or 0) > self.max_bytes:
            raise DownloadError(f"the download passed the size limit of {self.max_bytes // (1024 * 1024)} MB "
                                "(--max-download-mb) and was stopped")


def _refuse_live(info: dict, *, incomplete: bool = False) -> str | None:
    """yt-dlp ``match_filter``: a server may call a video live only on the second request."""
    return "live streams are not supported" if info.get("is_live") else None


def _factory(factory: Factory | None) -> Factory:
    if factory is not None:
        return factory
    # Plugins are Python code that yt-dlp would load from the current directory and user folders.
    os.environ["YTDLP_NO_PLUGINS"] = "1"
    from yt_dlp import YoutubeDL

    return YoutubeDL


def _options(cookies_from_browser: str | None = None, **extra: Any) -> dict:
    options: dict = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "logger": _SilentLogger(),
        "noplaylist": True,
        # Never resolve the entries of a feed or playlist (request amplification); we refuse them anyway.
        "extract_flat": "in_playlist",
        "playlistend": 1,
        "restrictfilenames": True,
        "socket_timeout": 30,
        "retries": 5,
        "ffmpeg_location": NO_FFMPEG,
        "match_filter": _refuse_live,
        # YouTube needs a JavaScript runtime to solve its player challenges;
        # enabling both lets yt-dlp use whichever is installed.
        "js_runtimes": {"deno": {}, "node": {}},
    }
    if cookies_from_browser:
        if cookies_from_browser not in BROWSERS:
            raise InputError(f"unsupported browser {cookies_from_browser!r} (choose from {', '.join(BROWSERS)})")
        options["cookiesfrombrowser"] = (cookies_from_browser,)
    return {**options, **extra}


def _explain(exc: Exception) -> str:
    text = clean_text(exc, 400)
    return text + _LOGIN_HINT if any(sign in text.lower() for sign in _LOGIN_SIGNS) else text


def extract_info(
    url: str,
    *,
    cookies_from_browser: str | None = None,
    allow_private_hosts: bool = False,
    factory: Factory | None = None,
) -> dict:
    """Metadata for a single, non-live video; refuses playlists and feeds."""
    options = _options(cookies_from_browser, skip_download=True)
    try:
        with network_guard(allow_private=allow_private_hosts), _factory(factory)(options) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # yt-dlp raises many unrelated types
        raise DownloadError(f"could not read {url}: {_explain(exc)}") from exc
    if not info:
        raise DownloadError(f"no media found at {url}")
    if info.get("_type") == "playlist":
        raise InputError("playlist and feed URLs are not supported; pass the URL of a single video")
    if info.get("is_live"):
        raise InputError("live streams are not supported; wait until the recording is available")
    return info


def download_audio(
    url: str,
    dest_dir: Path,
    *,
    max_filesize_mb: int,
    cookies_from_browser: str | None = None,
    allow_private_hosts: bool = False,
    deadline_seconds: float = DOWNLOAD_DEADLINE_SECONDS,
    factory: Factory | None = None,
) -> Path:
    """Download the best audio stream of ``url`` into ``dest_dir``."""
    limits = _DownloadLimits(max_filesize_mb * 1024 * 1024, deadline_seconds)
    options = _options(
        cookies_from_browser,
        format=AUDIO_FORMAT,
        outtmpl=str(dest_dir / "%(id)s.%(ext)s"),
        max_filesize=max_filesize_mb * 1024 * 1024,
        progress_hooks=[limits],
    )
    try:
        with network_guard(allow_private=allow_private_hosts), _factory(factory)(options) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as exc:
        raise DownloadError(f"download failed for {url}: {_explain(exc)}") from exc
    if (info or {}).get("is_live"):  # the first look said it was not live; see _refuse_live
        raise InputError("live streams are not supported; wait until the recording is available")
    downloads = (info or {}).get("requested_downloads") or []
    path = Path(downloads[0]["filepath"]) if downloads and downloads[0].get("filepath") else None
    if path is None or not path.is_file():
        raise DownloadError(
            f"no audio was downloaded from {url} (it may exceed --max-download-mb={max_filesize_mb})"
        )
    return path


def download_caption(
    url: str,
    language: str,
    auto: bool,
    dest_dir: Path,
    *,
    cookies_from_browser: str | None = None,
    allow_private_hosts: bool = False,
    factory: Factory | None = None,
) -> Path:
    """Download one caption track as json3 (word-timed for auto captions)."""
    options = _options(
        cookies_from_browser,
        skip_download=True,
        writesubtitles=not auto,
        writeautomaticsub=auto,
        subtitleslangs=[language],
        subtitlesformat="json3",
        outtmpl=str(dest_dir / "%(id)s.%(ext)s"),
    )
    try:
        with network_guard(allow_private=allow_private_hosts), _factory(factory)(options) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as exc:
        raise DownloadError(f"could not fetch captions for {url}: {_explain(exc)}") from exc
    track = ((info or {}).get("requested_subtitles") or {}).get(language) or {}
    path = Path(track["filepath"]) if track.get("filepath") else None
    if path is None or not path.is_file():
        raise DownloadError(f"YouTube did not return a '{language}' caption track")
    return path
