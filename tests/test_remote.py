import pytest

from transcribe_media import remote
from transcribe_media.errors import DownloadError, InputError


def make_factory(info=None, error=None, record=None):
    class FakeYDL:
        def __init__(self, opts):
            if record is not None:
                record.append(opts)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            if error:
                raise error
            return info

    return FakeYDL


def test_extract_info_never_reads_cookies_or_downloads():
    opts = []
    info = remote.extract_info("https://x/v", factory=make_factory({"id": "a"}, record=opts))
    assert info == {"id": "a"}
    (o,) = opts
    assert o["skip_download"] and o["noplaylist"] and o["js_runtimes"]
    assert not {"cookiefile", "cookiesfrombrowser", "username", "password"} & o.keys()


def test_playlists_are_refused():
    with pytest.raises(InputError, match="playlist"):
        remote.extract_info("https://x/p", factory=make_factory({"_type": "playlist"}))


@pytest.mark.parametrize("info", [None, {}])
def test_empty_result_is_a_download_error(info):
    with pytest.raises(DownloadError, match="no media"):
        remote.extract_info("https://x/v", factory=make_factory(info))


def test_extractor_failures_are_wrapped():
    with pytest.raises(DownloadError, match="private video"):
        remote.extract_info("https://x/v", factory=make_factory(error=RuntimeError("private video")))


def test_download_audio_returns_the_downloaded_file(tmp_path):
    audio_file = tmp_path / "abc.webm"
    audio_file.write_bytes(b"x")
    opts = []
    info = {"requested_downloads": [{"filepath": str(audio_file)}]}
    path = remote.download_audio("https://x/v", tmp_path, max_filesize_mb=10, factory=make_factory(info, record=opts))
    assert path == audio_file
    assert opts[0]["max_filesize"] == 10 * 1024 * 1024
    assert opts[0]["format"] == remote.AUDIO_FORMAT


def test_download_audio_reports_size_cap_when_nothing_was_saved(tmp_path):
    with pytest.raises(DownloadError, match="max-download-mb=5"):
        remote.download_audio("https://x/v", tmp_path, max_filesize_mb=5, factory=make_factory({}))


@pytest.mark.parametrize("auto", [True, False])
def test_download_caption_selects_track_kind(tmp_path, auto):
    cap = tmp_path / "abc.en.json3"
    cap.write_text("{}")
    opts = []
    info = {"requested_subtitles": {"en": {"filepath": str(cap)}}}
    path = remote.download_caption("https://x/v", "en", auto, tmp_path, factory=make_factory(info, record=opts))
    assert path == cap
    assert opts[0]["writeautomaticsub"] is auto and opts[0]["writesubtitles"] is (not auto)
    assert opts[0]["subtitleslangs"] == ["en"] and opts[0]["subtitlesformat"] == "json3"


def test_download_caption_missing_track(tmp_path):
    with pytest.raises(DownloadError, match="'en' caption"):
        remote.download_caption("https://x/v", "en", True, tmp_path, factory=make_factory({}))


def test_cookies_are_never_used_unless_a_browser_is_named():
    opts = []
    remote.extract_info("https://x/v", factory=make_factory({"id": "a"}, record=opts))
    assert "cookiesfrombrowser" not in opts[0]


def test_named_browser_is_passed_through_for_every_operation(tmp_path):
    opts = []
    factory = make_factory({"id": "a"}, record=opts)
    remote.extract_info("https://x/v", cookies_from_browser="firefox", factory=factory)
    try:
        remote.download_audio("https://x/v", tmp_path, max_filesize_mb=1, cookies_from_browser="firefox", factory=factory)
    except Exception:
        pass  # no file is produced by the fake; only the options matter here
    try:
        remote.download_caption("https://x/v", "en", False, tmp_path, cookies_from_browser="firefox", factory=factory)
    except Exception:
        pass
    assert len(opts) == 3 and all(o["cookiesfrombrowser"] == ("firefox",) for o in opts)


@pytest.mark.parametrize("browser", ["firefox:/etc/passwd", "chrome+gnomekeyring", "../x", "netscape"])
def test_only_plain_known_browser_names_are_accepted(browser):
    created = []
    def factory(opts):
        created.append(opts)
        return make_factory({"id": "a"})(opts)

    with pytest.raises(InputError, match="unsupported browser"):
        remote.extract_info("https://x/v", cookies_from_browser=browser, factory=factory)
    assert created == []


def test_yt_dlp_is_silenced_so_errors_are_reported_once(capsys):
    opts = []
    remote.extract_info("https://x/v", factory=make_factory({"id": "a"}, record=opts))
    logger = opts[0]["logger"]
    logger.error("ERROR: boom")  # must not print
    logger.warning("w")
    logger.info("i")
    logger.debug("d")
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


@pytest.mark.parametrize("message", [
    "Sign in to confirm you\u2019re not a bot", "unable to download video data: HTTP Error 403: Forbidden",
])
def test_login_walls_get_an_actionable_hint(message):
    with pytest.raises(DownloadError, match="--cookies-from-browser"):
        remote.extract_info("https://x/v", factory=make_factory(error=RuntimeError(message)))


def test_unrelated_errors_get_no_cookie_hint():
    with pytest.raises(DownloadError) as info:
        remote.extract_info("https://x/v", factory=make_factory(error=RuntimeError("video unavailable")))
    assert "cookies" not in str(info.value)
