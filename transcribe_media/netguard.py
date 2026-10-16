"""Connect-time network policy for fetching untrusted URLs.

Checking a URL's host once is not enough: redirects, embedded media, feeds and
DNS rebinding all send the downloader to other addresses later. This guard
refuses the connection itself, using a Python audit hook that sees every socket
connect made by this process while a guard is active.

Limits, stated plainly: it covers connections made by this process (yt-dlp's own
HTTP client), not by programs it spawns, and when an HTTP proxy is configured
the connection goes to the proxy, whose own policy then decides what is
reachable.
"""

from __future__ import annotations

import contextlib
import ipaddress
import os
import socket
import sys
import threading
from typing import Iterator
from urllib.parse import urlsplit

IP = ipaddress.IPv4Address | ipaddress.IPv6Address
_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
_DEFAULT_PROXY_PORTS = {"http": 80, "https": 443, "socks4": 1080, "socks5": 1080, "socks5h": 1080}


class BlockedAddressError(OSError):
    """A connection to a non-public address was refused."""


def _embedded(ip: IP) -> list[IP]:
    """The addresses an IPv6 form really points at (mapped, NAT64, 6to4, Teredo)."""
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped:
            return [ip.ipv4_mapped]
        if ip.sixtofour:
            return [ip.sixtofour]
        if ip.teredo:
            return list(ip.teredo)
        if ip in _NAT64:
            return [ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)]
    return []


def is_public(address: str) -> bool:
    """True only for a globally routable address; anything unparseable is not."""
    try:
        ip = ipaddress.ip_address(address.split("%")[0].rstrip("."))
    except ValueError:
        return False
    return ip.is_global and all(inner.is_global for inner in _embedded(ip))


_lock = threading.Lock()
_installed = False
_depth = 0
_exempt: set[tuple[str, int]] = set()


def _normalised(host: str) -> str:
    """The canonical text of an IP address (so '::1' and '0:0:0:0:0:0:0:1' compare equal)."""
    try:
        return str(ipaddress.ip_address(host.split("%")[0]))
    except ValueError:
        return host


def _proxy_endpoints() -> set[tuple[str, int]]:
    """(address, port) of each configured proxy: connecting to exactly those is expected.

    The port matters: a proxy on 127.0.0.1:3128 must not make every other service on loopback reachable.
    """
    endpoints: set[tuple[str, int]] = set()
    for variable in _PROXY_VARIABLES:
        for key in (variable, variable.lower()):
            value = os.environ.get(key, "").strip()
            if not value:
                continue
            parts = urlsplit(value if "://" in value else f"http://{value}")  # "host:port" is accepted too
            try:
                host, port = parts.hostname, parts.port or _DEFAULT_PROXY_PORTS.get(parts.scheme.lower())
            except ValueError:  # an unusable port number
                continue
            if not host or port is None:
                continue
            try:
                addresses = [str(ipaddress.ip_address(host))]
            except ValueError:  # a host name: exempt whatever it resolves to, on that port only
                try:
                    addresses = [_normalised(info[4][0]) for info in socket.getaddrinfo(host, port)]
                except OSError:
                    continue
            endpoints.update((address, port) for address in addresses)
    return endpoints


def _audit(event: str, args: tuple) -> None:
    if event != "socket.connect" or not _depth or len(args) < 2:
        return
    address = args[1]
    if not isinstance(address, tuple) or not address or not isinstance(address[0], str):
        return  # not an IP connection (for example a Unix socket)
    host = address[0]
    port = address[1] if len(address) > 1 and isinstance(address[1], int) else None
    try:
        ipaddress.ip_address(host.split("%")[0])
        candidates = [host]
    except ValueError:  # a hostname: connect() resolves it, so check what it resolves to
        try:
            candidates = [info[4][0] for info in socket.getaddrinfo(host, None)]
        except OSError:
            return  # it will fail to resolve on its own
    for candidate in candidates:
        if (_normalised(candidate), port) not in _exempt and not is_public(candidate):
            raise BlockedAddressError(
                f"refused to connect to non-public address {candidate} "
                "(pass --allow-private-hosts if this is intended)"
            )


@contextlib.contextmanager
def network_guard(*, allow_private: bool = False) -> Iterator[None]:
    """Refuse connections to non-public addresses for the duration of the block."""
    if allow_private:
        yield
        return
    global _installed, _depth, _exempt
    with _lock:
        if not _installed:
            sys.addaudithook(_audit)  # audit hooks cannot be removed; ours is inert when _depth is 0
            _installed = True
        _exempt = _proxy_endpoints()
        _depth += 1
    try:
        yield
    finally:
        with _lock:
            _depth -= 1
