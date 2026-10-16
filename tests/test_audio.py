import http.server
import shutil
import subprocess
import threading
import wave

import pytest

from transcribe_media import audio
from transcribe_media.errors import AudioError, FfmpegMissingError, NoAudioStreamError

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture
def tone(tmp_path):
    path = tmp_path / "tone.mp3"
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=1.5:sample_rate=44100", str(path))
    return path


def test_probe_duration(tone):
    assert audio.probe_duration(tone) == pytest.approx(1.5, abs=0.2)


def test_decode_produces_16khz_mono_pcm(tone, tmp_path):
    out = tmp_path / "out.wav"
    audio.decode_to_wav(tone, out)
    with wave.open(str(out)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)


def test_decode_with_loudness_normalisation(tone, tmp_path):
    out = tmp_path / "norm.wav"
    audio.decode_to_wav(tone, out, normalize=True)
    assert out.stat().st_size > 0


def test_decoded_audio_loads_as_normalised_float32(tone, tmp_path):
    out = tmp_path / "out.wav"
    audio.decode_to_wav(tone, out)
    samples = audio.load_samples(out)
    assert samples.dtype.name == "float32" and samples.ndim == 1
    assert len(samples) == pytest.approx(1.5 * 16000, abs=3200)
    assert -1.0 <= samples.min() and samples.max() <= 1.0 and samples.max() > 0.1


def test_load_samples_rejects_wrong_layout(tmp_path):
    path = tmp_path / "stereo44k.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(b"\x00\x00" * 200)
    with pytest.raises(AudioError, match="16-bit mono 16000 Hz"):
        audio.load_samples(path)


def test_load_samples_reports_unreadable_files(tmp_path):
    path = tmp_path / "garbage.wav"
    path.write_bytes(b"this is not a wav file")
    with pytest.raises(AudioError, match="cannot read"):
        audio.load_samples(path)


def test_video_without_audio_is_rejected(tmp_path):
    path = tmp_path / "silent.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=c=black:s=64x64:d=1", str(path))
    with pytest.raises(NoAudioStreamError):
        audio.ensure_audio_stream(path)


def test_audio_track_is_extracted_from_video(tmp_path):
    path = tmp_path / "movie.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=c=black:s=64x64:d=1", "-f", "lavfi", "-i", "sine=d=1", "-shortest", str(path))
    audio.ensure_audio_stream(path)
    out = tmp_path / "movie.wav"
    audio.decode_to_wav(path, out)
    assert out.stat().st_size > 0


def test_non_media_file_is_an_audio_error(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not media")
    with pytest.raises(AudioError):
        audio.ensure_audio_stream(path)


def test_filename_starting_with_dash_is_not_parsed_as_an_option(tone, tmp_path, monkeypatch):
    weird = tmp_path / "-weird.mp3"
    shutil.copy(tone, weird)
    monkeypatch.chdir(tmp_path)
    audio.ensure_audio_stream(weird.relative_to(tmp_path))
    audio.decode_to_wav(weird.relative_to(tmp_path), tmp_path / "w.wav")


def test_missing_tools_give_install_hint(monkeypatch):
    monkeypatch.setattr(audio.shutil, "which", lambda tool: None)
    with pytest.raises(FfmpegMissingError, match="ffmpeg and ffprobe"):
        audio.require_ffmpeg()


def test_crafted_playlist_cannot_make_ffmpeg_fetch_urls(tmp_path):
    hits = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(404)
            self.end_headers()

        do_HEAD = do_GET

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        playlist = tmp_path / "evil.m3u8"
        playlist.write_text(
            "#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\n"
            f"http://127.0.0.1:{server.server_port}/seg.ts\n#EXT-X-ENDLIST\n"
        )
        # Control: when HTTP is explicitly permitted FFmpeg does reach the server
        # (newer builds already restrict local playlists by default). If this
        # build cannot, the assertions below would pass vacuously, so skip.
        subprocess.run(
            ["ffprobe", "-v", "quiet", "-protocol_whitelist", "file,http,https,tcp,tls,crypto", "-i", str(playlist)],
            capture_output=True, timeout=30,
        )
        if not hits:
            pytest.skip("this ffmpeg build does not fetch HTTP playlist segments")
        hits.clear()

        with pytest.raises(AudioError):
            audio.ensure_audio_stream(playlist)
        with pytest.raises(AudioError):
            audio.decode_to_wav(playlist, tmp_path / "x.wav")
        assert hits == []
    finally:
        server.shutdown()
