import http.server
import shutil
import socket
import threading

import pytest

from transcribe_media import netguard, remote
from transcribe_media.errors import TranscribeMediaError
from transcribe_media.netguard import BlockedAddressError, is_public, network_guard


@pytest.mark.parametrize("address, public", [
    ("8.8.8.8", True), ("2606:4700:4700::1111", True),
    ("127.0.0.1", False), ("10.0.0.1", False), ("192.168.1.1", False), ("172.16.0.1", False),
    ("169.254.169.254", False), ("0.0.0.0", False), ("100.64.0.1", False), ("::1", False),
    ("fe80::1%eth0", False), ("fc00::1", False),
    # Forms that smuggle an IPv4 address inside an IPv6 one
    ("::ffff:127.0.0.1", False), ("::ffff:10.0.0.1", False), ("::ffff:8.8.8.8", True),
    ("64:ff9b::a9fe:a9fe", False), ("2002:7f00:1::", False),
    ("127.0.0.1.", False), ("not an address", False), ("", False),
])
def test_address_policy(address, public):
    assert is_public(address) is public


@pytest.fixture
def listener():
    """A TCP server on loopback that counts connections."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(5)
    server.settimeout(0.2)
    accepted = []
    stop = threading.Event()

    def accept():
        while not stop.is_set():
            try:
                conn, _ = server.accept()
                accepted.append(1)
                conn.close()
            except OSError:
                continue

    threading.Thread(target=accept, daemon=True).start()
    yield server.getsockname()[1], accepted
    stop.set()
    server.close()


def clear_proxies(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)


def test_guard_refuses_loopback_only_while_active(listener, monkeypatch):
    clear_proxies(monkeypatch)
    port, accepted = listener
    with network_guard():
        with pytest.raises(BlockedAddressError, match="127.0.0.1"):
            socket.create_connection(("127.0.0.1", port), timeout=2)
    assert accepted == []
    socket.create_connection(("127.0.0.1", port), timeout=2).close()  # inert again outside the guard


def test_guard_also_checks_what_a_host_name_resolves_to(listener, monkeypatch):
    clear_proxies(monkeypatch)
    port, accepted = listener
    with network_guard(), socket.socket() as sock, pytest.raises(BlockedAddressError):
        sock.connect(("localhost", port))  # connect() resolves the name itself
    assert accepted == []


def test_allow_private_switches_the_guard_off(listener, monkeypatch):
    clear_proxies(monkeypatch)
    port, _ = listener
    with network_guard(allow_private=True):
        socket.create_connection(("127.0.0.1", port), timeout=2).close()


def test_a_configured_proxy_is_exempt_because_connecting_to_it_is_expected(listener, monkeypatch):
    clear_proxies(monkeypatch)
    port, _ = listener
    monkeypatch.setenv("HTTPS_PROXY", f"http://127.0.0.1:{port}")
    with network_guard():
        socket.create_connection(("127.0.0.1", port), timeout=2).close()


def test_nested_guards_stay_active_until_the_outermost_exits(listener, monkeypatch):
    clear_proxies(monkeypatch)
    port, _ = listener
    with network_guard():
        with network_guard():
            pass
        with pytest.raises(BlockedAddressError):
            socket.create_connection(("127.0.0.1", port), timeout=2)
    socket.create_connection(("127.0.0.1", port), timeout=2).close()


def test_the_guard_is_released_even_when_the_body_fails(listener, monkeypatch):
    clear_proxies(monkeypatch)
    port, _ = listener
    with pytest.raises(RuntimeError), network_guard():
        raise RuntimeError("boom")
    socket.create_connection(("127.0.0.1", port), timeout=2).close()


def test_every_yt_dlp_call_runs_inside_the_guard(monkeypatch, tmp_path):
    entered = []

    class Recorder:
        def __init__(self, allow_private=False):
            self.allow = allow_private

        def __enter__(self):
            entered.append(self.allow)

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(remote, "network_guard", lambda allow_private=False: Recorder(allow_private))

    class FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            return {"id": "a", "requested_downloads": [], "requested_subtitles": {}}

    for call in (
        lambda allow: remote.extract_info("https://x/v", allow_private_hosts=allow, factory=FakeYDL),
        lambda allow: remote.download_audio("https://x/v", tmp_path, max_filesize_mb=1, allow_private_hosts=allow, factory=FakeYDL),
        lambda allow: remote.download_caption("https://x/v", "en", True, tmp_path, allow_private_hosts=allow, factory=FakeYDL),
    ):
        for allow in (False, True):
            try:
                call(allow)
            except TranscribeMediaError:
                pass  # the fake yields no files; only the guard matters here
    assert entered == [False, True] * 3


# --- the reviewer's attacks, replayed against real yt-dlp --------------------- #
# A "front" server on 127.0.0.1 stands in for a public website (it is exempted by acting as the
# configured proxy). It redirects to, or embeds, targets on 127.0.0.2, which must stay unreachable.

def _serve(bind, handler):
    server = http.server.ThreadingHTTPServer((bind, 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def attack_lab(monkeypatch):
    hits = {"secret": 0, "ftp": 0}

    class Secret(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits["secret"] += 1
            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Content-Length", "4")
            self.end_headers()
            self.wfile.write(b"ID3\x00")

        do_HEAD = do_GET

        def log_message(self, *args):
            pass

    try:
        secret = _serve("127.0.0.2", Secret)
    except OSError:
        pytest.skip("this system cannot bind 127.0.0.2")
    secret_url = f"http://127.0.0.2:{secret.server_port}/secret.mp3"

    ftp = socket.socket()
    ftp.bind(("127.0.0.1", 0))
    ftp.listen(5)
    ftp.settimeout(0.2)
    stop = threading.Event()

    def count_ftp():
        while not stop.is_set():
            try:
                conn, _ = ftp.accept()
                hits["ftp"] += 1
                conn.close()
            except OSError:
                continue

    threading.Thread(target=count_ftp, daemon=True).start()
    ftp_port = ftp.getsockname()[1]

    secret_dir = secret_url.rsplit("/", 1)[0] + "/"

    class Front(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.endswith(".m3u8"):
                # A playlist yt-dlp cannot download itself (SAMPLE-AES), so it would hand it to ffmpeg.
                body = ("#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:2\n#EXT-X-MEDIA-SEQUENCE:0\n"
                        f'#EXT-X-KEY:METHOD=SAMPLE-AES,URI="{secret_dir}key"\n#EXTINF:2.0,\n'
                        f"{secret_dir}seg.ts\n#EXT-X-ENDLIST\n").encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/vnd.apple.mpegurl")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.endswith("/page"):
                body = (f'<html><body><video src="{secret_url}"></video>'
                        f'<video src="ftp://127.0.0.1:{ftp_port}/x.mp3"></video></body></html>').encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(302)
                self.send_header("Location", secret_url)
                self.end_headers()

        do_HEAD = do_GET

        def log_message(self, *args):
            pass

    front = _serve("127.0.0.1", Front)
    for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{front.server_port}")
    monkeypatch.setenv("http_proxy", f"http://127.0.0.1:{front.server_port}")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,127.0.0.2")
    monkeypatch.setenv("no_proxy", "127.0.0.1,127.0.0.2")
    yield hits, f"http://127.0.0.1:{front.server_port}"
    stop.set()
    ftp.close()
    front.shutdown()
    secret.shutdown()


def attempt(url, tmp_path, allow_private):
    try:
        remote.download_audio(url, tmp_path, max_filesize_mb=5, allow_private_hosts=allow_private)
    except TranscribeMediaError as exc:
        return str(exc)
    return None


def test_a_redirect_to_a_forbidden_address_is_not_followed(attack_lab, tmp_path):
    hits, front = attack_lab
    # Control: with the guard off the redirect IS followed, so this test cannot pass vacuously.
    assert attempt(front + "/redirect", tmp_path, allow_private=True) is None and hits["secret"] > 0
    hits["secret"] = 0
    error = attempt(front + "/redirect", tmp_path, allow_private=False)
    assert error is not None and hits["secret"] == 0


def test_media_urls_embedded_in_a_page_are_not_fetched(attack_lab, tmp_path):
    hits, front = attack_lab
    attempt(front + "/page", tmp_path, allow_private=False)
    assert hits["secret"] == 0 and hits["ftp"] == 0


def test_only_safe_protocols_are_ever_selected_for_download(attack_lab, tmp_path):
    """ftp/rtmp downloads are done by ffmpeg, whose sockets the guard cannot see."""
    import yt_dlp

    hits, front = attack_lab

    def run(format_selector):
        hits.update(secret=0, ftp=0)
        opts = remote._options(format=format_selector, outtmpl=str(tmp_path / "%(id)s.%(ext)s"))
        opts.pop("extract_flat")
        opts.pop("playlistend")
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.extract_info(front + "/page", download=True)
        except Exception:
            pass
        return hits["ftp"]

    if run("bestaudio/best") == 0:
        pytest.skip("this yt-dlp version no longer follows embedded ftp media, so there is nothing to prevent")
    assert run(remote.AUDIO_FORMAT) == 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is needed to show the hole this closes")
def test_yt_dlp_cannot_hand_a_download_to_ffmpeg_to_reach_a_forbidden_address(attack_lab, tmp_path):
    """HLS with a feature yt-dlp cannot handle is passed to ffmpeg, whose sockets the guard cannot see."""
    import yt_dlp

    hits, front = attack_lab
    url = front + "/attacker/a.m3u8"

    def run(remove_setting=None):
        hits.update(secret=0, ftp=0)
        opts = remote._options(format=remote.AUDIO_FORMAT, outtmpl=str(tmp_path / "%(id)s.%(ext)s"))
        opts.pop("extract_flat")
        opts.pop("playlistend")
        if remove_setting:
            opts.pop(remove_setting)
        try:
            with network_guard(), yt_dlp.YoutubeDL(opts) as ydl:
                ydl.extract_info(url, download=True)
        except Exception:
            pass
        return hits["secret"]

    # Control: without the setting ffmpeg really does fetch the key and segments, guard or no guard.
    if run(remove_setting="ffmpeg_location") == 0:
        pytest.skip("this yt-dlp or ffmpeg no longer fetches such playlists, so there is nothing to prevent")
    assert run() == 0


def test_the_public_download_function_never_lets_ffmpeg_fetch_for_it(attack_lab, tmp_path):
    hits, front = attack_lab
    error = attempt(front + "/attacker/a.m3u8", tmp_path, allow_private=False)
    assert error is not None and "ffmpeg" in error and hits["secret"] == 0


def test_an_endpoint_that_is_not_the_proxy_is_not_exempt_just_because_it_shares_the_proxys_address(
        listener, monkeypatch):
    """A proxy on 127.0.0.1:A must not make every other service on 127.0.0.1 reachable."""
    clear_proxies(monkeypatch)
    port, accepted = listener
    other = socket.socket()
    other.bind(("127.0.0.1", 0))
    other.listen(1)
    try:
        monkeypatch.setenv("HTTPS_PROXY", f"http://127.0.0.1:{port}")
        with network_guard():
            socket.create_connection(("127.0.0.1", port), timeout=2).close()  # the proxy itself
            with pytest.raises(BlockedAddressError, match="127.0.0.1"):
                socket.create_connection(("127.0.0.1", other.getsockname()[1]), timeout=2)
    finally:
        other.close()


@pytest.mark.parametrize("value, endpoint", [
    ("http://127.0.0.1:3128", ("127.0.0.1", 3128)),
    ("127.0.0.1:3128", ("127.0.0.1", 3128)),               # no scheme, as many tools accept
    ("http://user:pw@10.1.2.3:8080/", ("10.1.2.3", 8080)),
    ("http://10.1.2.3", ("10.1.2.3", 80)),                  # default ports follow the scheme
    ("https://10.1.2.3", ("10.1.2.3", 443)),
    ("socks5://10.1.2.3", ("10.1.2.3", 1080)),
    ("http://[::1]:8888", ("::1", 8888)),
    ("http://[0:0:0:0:0:0:0:1]:8888", ("::1", 8888)),       # spelled differently, same address
])
def test_a_proxy_is_exempt_at_exactly_its_own_address_and_port(monkeypatch, value, endpoint):
    clear_proxies(monkeypatch)
    monkeypatch.setenv("HTTP_PROXY", value)
    assert netguard._proxy_endpoints() == {endpoint}


@pytest.mark.parametrize("value", ["", "   ", "http://", "http://host:notaport", "http://127.0.0.1:99999", "proxy.invalid"])
def test_unusable_proxy_settings_exempt_nothing(monkeypatch, value):
    clear_proxies(monkeypatch)
    monkeypatch.setenv("HTTP_PROXY", value)
    assert netguard._proxy_endpoints() == set()


def test_the_exemption_matches_the_address_however_it_is_written(monkeypatch):
    clear_proxies(monkeypatch)
    monkeypatch.setenv("HTTP_PROXY", "http://[::1]:8888")
    with network_guard():
        netguard._audit("socket.connect", (None, ("0:0:0:0:0:0:0:1", 8888, 0, 0)))  # allowed
        with pytest.raises(BlockedAddressError):
            netguard._audit("socket.connect", (None, ("::1", 8889, 0, 0)))
