"""Resolve the user's input (local file or URL) into a local media file."""

from __future__ import annotations

import ipaddress
import re
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from . import remote
from .console import clean_text
from .errors import InputError
from .netguard import is_public

_URL = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
_MAX_STEM_CHARS = 80
# File systems cap a name at 255 bytes, and CJK characters take three each. The stem gets
# a budget that leaves room for ".transcript.json", ".youtube-manual-<lang>.txt" and ".tmp".
_MAX_STEM_BYTES = 120
_BLOCKED_NAMES = {"localhost", "metadata", "metadata.google.internal"}
_BLOCKED_SUFFIXES = (".localhost", ".internal", ".local")


def is_network_path(text: str) -> bool:
    """Whether ``text`` is a Windows UNC path (``\\\\host\\share\\file``), in any of its spellings."""
    path = text.replace("/", "\\")  # Windows accepts either slash
    if path.upper().startswith(("\\\\?\\UNC\\", "\\\\.\\UNC\\")):
        return True
    return path.startswith("\\\\") and not path.startswith(("\\\\?\\", "\\\\.\\"))


def refuse_network_path(text: str, what: str, *, windows: bool | None = None) -> None:
    """Opening a UNC path makes Windows contact that host and offer it the user's credentials."""
    if (sys.platform == "win32" if windows is None else windows) and is_network_path(text):
        raise InputError(f"{what} {clean_text(text, 100)} is a network path (\\\\host\\share); "
                         "copy the file to a local folder first")


@dataclass
class MediaSource:
    path: Path
    stem: str  # filesystem-safe name for outputs
    title: str
    url: str | None = None
    info: dict[str, Any] | None = None  # yt-dlp metadata for URLs


def is_url(text: str) -> bool:
    return bool(_URL.match(text))


def safe_stem(text: str, fallback: str = "transcript", max_bytes: int = _MAX_STEM_BYTES) -> str:
    """A filename stem without path separators, control characters or reserved names."""
    stem = re.sub(r"[^\w]+", "-", text, flags=re.UNICODE).strip("-_.")[:_MAX_STEM_CHARS]
    stem = stem.encode("utf-8")[:max_bytes].decode("utf-8", "ignore").strip("-_.")
    if not stem:
        return fallback
    return f"_{stem}" if stem.lower() in _RESERVED else stem


Resolver = Callable[..., Any]


def _literal_address(name: str) -> str | None:
    """The IP address a host name spells out, or None for an ordinary name.

    Covers the legacy forms that resolvers also accept: 2130706433, 0x7f.1, 0177.0.0.1.
    """
    candidate = name.split("%")[0]
    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        pass
    try:
        return socket.inet_ntoa(socket.inet_aton(candidate))
    except (OSError, UnicodeError):
        return None


def _refuse_non_public(host: str, detail: str = "") -> InputError:
    return InputError(
        f"{host} is not a public host{detail}; pass --allow-private-hosts if this is intended"
    )


def validate_url(url: str, *, allow_private_hosts: bool = False, resolver: Resolver = socket.getaddrinfo) -> str:
    """Accept only http(s) URLs without embedded credentials, aimed at public hosts.

    A URL is untrusted input (it may come from a web page the model read), so
    loopback, link-local and private addresses are refused unless explicitly
    allowed. This is the first line of defence only: redirects and embedded
    media are checked again at connect time by :mod:`netguard`. A host that
    cannot be resolved here is let through, since it may be reachable only via a
    proxy; the download fails on its own if it is not.
    """
    try:
        parts = urlsplit(url)
        host, _ = parts.hostname, parts.port  # .port raises ValueError for a malformed port
    except (ValueError, UnicodeError) as exc:
        raise InputError(f"malformed URL: {clean_text(exc)}") from exc
    if parts.scheme not in ("http", "https"):
        raise InputError(f"only http and https URLs are supported, got {parts.scheme or 'none'!r}")
    if not host:
        raise InputError("the URL has no host")
    if parts.username or parts.password:
        raise InputError("URLs containing credentials are not accepted")
    if allow_private_hosts:
        return url

    name = host.rstrip(".").lower()
    if name in _BLOCKED_NAMES or name.endswith(_BLOCKED_SUFFIXES):
        raise _refuse_non_public(name)
    literal = _literal_address(name)
    if literal is not None:  # an IP literal needs no DNS to be judged
        if not is_public(literal):
            raise _refuse_non_public(name, f" (it is {literal})" if literal != name else "")
        return url
    try:
        addresses = {info[4][0] for info in resolver(name, None, proto=socket.IPPROTO_TCP)}
    except (socket.gaierror, UnicodeError):
        return url
    for address in addresses:
        if not is_public(address):
            raise _refuse_non_public(name, f" (it resolves to {address})")
    return url


def resolve_input(
    arg: str,
    *,
    workdir: Path,
    allow_private_hosts: bool = False,
    max_download_mb: int = 2048,
    cookies_from_browser: str | None = None,
    log: Callable[[str], None] = lambda message: None,
    factory: remote.Factory | None = None,
) -> MediaSource:
    """Turn a path or URL into a :class:`MediaSource` backed by a local file."""
    if not is_url(arg):
        refuse_network_path(arg, "the input")
        path = Path(arg).expanduser()
        if not path.is_file():
            raise InputError(f"input file not found: {arg}")
        return MediaSource(path=path, stem=safe_stem(path.stem), title=clean_text(path.name, 150))

    url = validate_url(arg, allow_private_hosts=allow_private_hosts)
    log("Reading video metadata ...")
    info = remote.extract_info(
        url, cookies_from_browser=cookies_from_browser, allow_private_hosts=allow_private_hosts, factory=factory)
    # Titles come from the remote site: they are data, never to be trusted as text we print.
    title = clean_text(info.get("title") or info.get("id") or "video", 150)
    log(f'Downloading audio: "{title}"')
    path = remote.download_audio(
        url, workdir, max_filesize_mb=max_download_mb, cookies_from_browser=cookies_from_browser,
        allow_private_hosts=allow_private_hosts, factory=factory,
    )
    ident = clean_text(info.get("id") or "", 40)
    title_part = safe_stem(title, max_bytes=90)
    stem = title_part if not ident or ident.lower() in title.lower() else f"{title_part}-{safe_stem(ident, max_bytes=30)}"
    return MediaSource(path=path, stem=stem, title=title, url=url, info=info)
