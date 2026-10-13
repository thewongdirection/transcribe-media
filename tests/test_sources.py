import socket

import pytest

from transcribe_media import remote, sources
from transcribe_media.errors import InputError
from transcribe_media.sources import is_url, resolve_input, safe_stem, validate_url


def resolver_for(*addresses):
    def resolve(host, port, proto=0):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, 0)) for a in addresses]

    return resolve


def unresolvable(host, port, proto=0):
    raise socket.gaierror("no dns")


@pytest.mark.parametrize("text, expected", [
    ("https://youtu.be/x", True), ("http://a/b", True), ("ftp://a", True),
    ("C:\\Users\\me\\a.mp3", False), ("/tmp/a.mp3", False), ("a.mp3", False),
])
def test_is_url(text, expected):
    assert is_url(text) is expected


@pytest.mark.parametrize("text, expected", [
    ("My Talk: Part 1/2", "My-Talk-Part-1-2"),
    ("../../etc/passwd", "etc-passwd"),
    ("日本語のタイトル", "日本語のタイトル"),
    ("???", "transcript"),
    ("CON", "_CON"),
    ("a" * 200, "a" * 80),
])
def test_safe_stem(text, expected):
    assert safe_stem(text) == expected


@pytest.mark.parametrize("url", ["ftp://x/y", "file:///etc/passwd", "javascript:alert(1)"])
def test_non_http_schemes_are_rejected(url):
    with pytest.raises(InputError, match="http"):
        validate_url(url, resolver=resolver_for("8.8.8.8"))


def test_url_credentials_are_rejected():
    with pytest.raises(InputError, match="credentials"):
        validate_url("https://user:pw@example.com/v", resolver=resolver_for("8.8.8.8"))


def test_url_without_host_is_rejected():
    with pytest.raises(InputError, match="no host"):
        validate_url("https:///path")


@pytest.mark.parametrize("address", ["127.0.0.1", "10.1.2.3", "192.168.0.9", "169.254.169.254", "::1", "fe80::1%eth0"])
def test_private_targets_are_refused(address):
    with pytest.raises(InputError, match="not a public host"):
        validate_url("https://internal.example/v", resolver=resolver_for(address))


def test_any_private_answer_among_several_is_refused():
    with pytest.raises(InputError):
        validate_url("https://x.example/v", resolver=resolver_for("8.8.8.8", "10.0.0.1"))


def test_public_host_is_accepted():
    url = "https://www.youtube.com/watch?v=abc"
    assert validate_url(url, resolver=resolver_for("142.251.150.4")) == url


def test_private_hosts_can_be_allowed_explicitly():
    assert validate_url("http://192.168.1.5/v", allow_private_hosts=True, resolver=unresolvable)


def test_unresolvable_host_is_not_blocked_so_proxies_keep_working():
    assert validate_url("https://via-proxy.example/v", resolver=unresolvable)


def test_local_file_resolution(tmp_path):
    f = tmp_path / "My Talk.mp3"
    f.write_bytes(b"x")
    src = resolve_input(str(f), workdir=tmp_path)
    assert (src.path, src.stem, src.title, src.url) == (f, "My-Talk", "My Talk.mp3", None)


def test_missing_local_file(tmp_path):
    with pytest.raises(InputError, match="not found"):
        resolve_input(str(tmp_path / "nope.mp3"), workdir=tmp_path)


def test_directory_is_not_a_file(tmp_path):
    with pytest.raises(InputError):
        resolve_input(str(tmp_path), workdir=tmp_path)


def test_url_resolution_downloads_audio(tmp_path, monkeypatch):
    dl = tmp_path / "abc123.webm"
    dl.write_bytes(b"x")
    info = {"id": "abc123", "title": "A: Great/Talk"}
    monkeypatch.setattr(remote, "extract_info", lambda url, **kw: info)
    monkeypatch.setattr(remote, "download_audio", lambda url, d, **kw: dl)
    logs = []
    src = resolve_input("https://example.com/v", workdir=tmp_path, allow_private_hosts=True, log=logs.append)
    assert (src.path, src.stem, src.title, src.info) == (dl, "A-Great-Talk-abc123", "A: Great/Talk", info)
    assert src.url == "https://example.com/v" and len(logs) == 2


def test_url_validation_runs_before_any_network_use(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "validate_url", lambda *a, **k: (_ for _ in ()).throw(InputError("blocked")))
    monkeypatch.setattr(remote, "extract_info", lambda *a, **k: pytest.fail("must not be reached"))
    with pytest.raises(InputError, match="blocked"):
        resolve_input("http://169.254.169.254/latest", workdir=tmp_path)


# --- Windows network paths: opening one sends the user's credentials to that host ------------------------------- #

@pytest.mark.parametrize("text", [
    r"\\attacker\share\a.mp3", "//attacker/share/a.mp3", r"\\attacker.example\c$\a.mp3",
    r"\\?\UNC\attacker\share\a.mp3", r"\\?\unc\attacker\share\a.mp3", r"\\10.0.0.9\share\a.mp3",
    r"\\.\UNC\attacker\share\a.mp3", r"\\.\unc\attacker\share\a.mp3",
])
def test_unc_paths_are_recognised(text):
    assert sources.is_network_path(text)


@pytest.mark.parametrize("text", [
    r"C:\Users\me\a.mp3", "C:/Users/me/a.mp3", r"\Users\me\a.mp3", "relative/a.mp3", "a.mp3",
    r"\\?\C:\Users\me\a.mp3", r"\\.\C:\a.mp3", "/home/me/a.mp3", "",
])
def test_local_paths_are_not_network_paths(text):
    assert not sources.is_network_path(text)


def test_a_network_path_is_refused_on_windows_before_anything_touches_it():
    with pytest.raises(InputError, match="network path"):
        sources.refuse_network_path(r"\\attacker\share\a.mp3", "the input", windows=True)
    sources.refuse_network_path(r"C:\a.mp3", "the input", windows=True)


def test_elsewhere_a_backslashed_name_is_just_an_odd_file_name():
    sources.refuse_network_path(r"\\attacker\share\a.mp3", "the input", windows=False)


def test_the_refusal_names_the_offending_path_without_control_characters():
    with pytest.raises(InputError) as caught:
        sources.refuse_network_path("\\\\evil\\s\n\x1b[2J.mp3", "the input", windows=True)
    assert "\n" not in str(caught.value) and "\x1b" not in str(caught.value)


@pytest.fixture
def on_windows(monkeypatch):
    monkeypatch.setattr(sources.sys, "platform", "win32")


def test_the_input_is_checked_before_the_file_system_is_asked_about_it(on_windows, tmp_path):
    with pytest.raises(InputError, match="network path"):
        resolve_input(r"\\attacker\share\a.mp3", workdir=tmp_path)


def test_output_locations_are_checked_too(on_windows):
    from transcribe_media import cli

    with pytest.raises(InputError, match="network path"):
        cli.plan_outputs(r"\\attacker\share\out", "stem", ["txt"], overwrite=False)
    with pytest.raises(InputError, match="network path"):
        cli.plan_audio_copy(r"\\attacker\share\copy.wav", overwrite=False)
