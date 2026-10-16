"""User-facing messages. Everything goes to stderr so stdout stays clean for results."""

from __future__ import annotations

import sys
import time
import unicodedata
from typing import Callable, Protocol


class Log(Protocol):
    def info(self, message: str) -> None: ...
    def warn(self, message: str) -> None: ...
    def notice(self, message: str) -> None: ...


class Console:
    """``info`` can be silenced; ``warn`` and ``notice`` never are.

    ``notice`` carries things the user should always see even in quiet mode,
    such as the fact that audio is being uploaded to a cloud provider.
    """

    def __init__(self, quiet: bool = False):
        self.quiet = quiet

    def info(self, message: str) -> None:
        if not self.quiet:
            print(message, file=sys.stderr, flush=True)

    def warn(self, message: str) -> None:
        print(f"warning: {message}", file=sys.stderr, flush=True)

    def notice(self, message: str) -> None:
        print(message, file=sys.stderr, flush=True)


class RecordingLog:
    """Wraps a log and remembers warnings, so a remote run can send them back."""

    def __init__(self, inner: Log):
        self._inner = inner
        self.warnings: list[str] = []

    def info(self, message: str) -> None:
        self._inner.info(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        self._inner.warn(message)

    def notice(self, message: str) -> None:
        self._inner.notice(message)


_KEEP_FORMAT_CHARS = {"\u200c", "\u200d"}  # joiners that real scripts (Persian, Indic) need


def clean_text(value: object, limit: int = 200) -> str:
    """Make text from outside (titles, error messages, cloud warnings) safe to print.

    Control characters (including escape sequences and newlines) and invisible
    formatting marks are removed, whitespace is collapsed to one line, and the
    length is capped. Such text can say anything, so it must never be able to
    forge a line of this tool's own output.
    """
    text = "".join(
        " " if unicodedata.category(ch) == "Cc"
        else "" if unicodedata.category(ch) == "Cf" and ch not in _KEEP_FORMAT_CHARS
        else ch
        for ch in str(value)
    )
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def clock(seconds: float) -> str:
    """``m:ss`` or ``h:mm:ss``."""
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class ProgressReporter:
    """Prints a line every 5 % or 30 s, whichever comes first."""

    def __init__(self, log: Log, now: Callable[[], float] = time.monotonic):
        self._log, self._now = log, now
        self._last_pct, self._last_time = -5.0, float("-inf")

    def __call__(self, done: float, total: float) -> None:
        if total <= 0:
            return
        pct, moment = min(done / total * 100, 100.0), self._now()
        if pct - self._last_pct >= 5 or moment - self._last_time >= 30:
            self._log.info(f"  transcribing {pct:3.0f}%  ({clock(done)} / {clock(total)})")
            self._last_pct, self._last_time = pct, moment
