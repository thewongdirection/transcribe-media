import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from transcribe_media import cli, job, preflight, remote, sources
from transcribe_media.environment import Environment
from transcribe_media.errors import BadRecordingError, InputError, OverwriteRefusedError, WriteError
from transcribe_media.job import JobResult
from transcribe_media.transcribe import TranscriptionResult

from .conftest import NOT_READY, make_hw

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def parse(*argv):
    return cli.build_parser().parse_args(list(argv))


# --- formats and output planning ------------------------------------------- #

def test_default_formats_are_text_and_json():
    assert cli.resolve_formats(None, None) == ["txt", "json"]


def test_formats_follow_the_output_suffix_when_not_given():
    assert cli.resolve_formats(None, "talk.srt") == ["srt"]
    assert cli.resolve_formats(None, "talk") == ["txt", "json"]


def test_explicit_formats_are_validated_and_deduplicated():
    assert cli.resolve_formats("srt, txt,srt", None) == ["srt", "txt"]
    with pytest.raises(InputError, match="docx"):
        cli.resolve_formats("txt,docx", None)
    with pytest.raises(InputError):
        cli.resolve_formats(",", None)


def test_default_output_goes_next_to_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plan = cli.plan_outputs(None, "talk", ["txt", "json"], overwrite=False)
    assert plan.files == {"txt": tmp_path / "talk.transcript.txt", "json": tmp_path / "talk.transcript.json"}
    assert plan.base == tmp_path / "talk.transcript"


def test_output_with_known_suffix_is_used_as_given(tmp_path):
    plan = cli.plan_outputs(str(tmp_path / "out.srt"), "x", ["srt"], overwrite=False)
    assert plan.files == {"srt": tmp_path / "out.srt"}


def test_output_without_suffix_is_a_base_name_for_every_format(tmp_path):
    plan = cli.plan_outputs(str(tmp_path / "out"), "x", ["txt", "json"], overwrite=False)
    assert set(plan.files.values()) == {tmp_path / "out.txt", tmp_path / "out.json"}


def test_output_directory_gets_a_derived_name(tmp_path):
    plan = cli.plan_outputs(str(tmp_path), "talk", ["txt"], overwrite=False)
    assert plan.files == {"txt": tmp_path / "talk.transcript.txt"}


def test_stdout_output_uses_first_format():
    plan = cli.plan_outputs("-", "x", ["srt", "txt"], overwrite=False)
    assert plan.to_stdout and plan.stdout_format == "srt" and plan.files == {} and plan.base is None


def test_existing_output_is_protected_unless_overwrite(tmp_path):
    target = tmp_path / "out.txt"
    target.write_text("precious")
    with pytest.raises(OverwriteRefusedError, match="--overwrite"):
        cli.plan_outputs(str(target), "x", ["txt"], overwrite=False)
    assert cli.plan_outputs(str(target), "x", ["txt"], overwrite=True).files == {"txt": target}


def test_write_text_is_utf8_atomic_and_creates_parents(tmp_path):
    target = tmp_path / "deep" / "dir" / "t.txt"
    cli.write_text(target, "你好\nworld\n")
    assert target.read_bytes() == "你好\nworld\n".encode()
    assert list(target.parent.iterdir()) == [target]


def test_write_failure_is_a_write_error(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(WriteError):
        cli.write_text(blocker / "child.txt", "x")


def source_with(url, **info):
    return sources.MediaSource(Path("a"), "a", "a", url=url, info=info or None)


def test_no_url_for_local_files():
    assert cli.public_url(sources.MediaSource(Path("a"), "a", "a")) is None


def test_signed_query_strings_never_reach_the_output():
    # Regression: the extractor's own "canonical" URL for a direct link still carries the signature.
    leaky = source_with("https://bucket.example/a.mp3?X-Amz-Signature=SECRET&token=abc#frag",
                        extractor_key="Generic", id="a",
                        webpage_url="https://bucket.example/a.mp3?X-Amz-Signature=SECRET&token=abc#frag")
    assert cli.public_url(leaky) == "https://bucket.example/a.mp3"
    assert "SECRET" not in cli.public_url(source_with("https://h.example:8443/p?x=SECRET"))


def test_credentials_and_ports_are_handled():
    assert cli.public_url(source_with("https://user:pw@h.example:8443/a.mp3?k=1")) == "https://h.example:8443/a.mp3"


def test_youtube_urls_are_rebuilt_from_the_video_id():
    yt = source_with("https://youtu.be/dQw4w9WgXcQ?si=tracking&list=PL1", extractor_key="Youtube", id="dQw4w9WgXcQ")
    assert cli.public_url(yt) == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_a_youtube_id_that_is_not_an_id_is_not_trusted():
    odd = source_with("https://www.youtube.com/watch?v=x&t=1", extractor_key="Youtube", id="a/b?c=d")
    assert cli.public_url(odd) == "https://www.youtube.com/watch"


# --- captions --------------------------------------------------------------- #

def youtube_source(**info):
    base = {"extractor_key": "Youtube", "subtitles": {}, "automatic_captions": {}}
    base.update(info)
    return sources.MediaSource(Path("a"), "a", "t", url="https://www.youtube.com/watch?v=x", info=base)


RESULT = SimpleNamespace(language="en", segments=[{"text": "hello big world", "speaker": None}])
PLAN = cli.OutputPlan({})


def test_file_inputs_have_no_caption_logic(tmp_path):
    source = sources.MediaSource(Path("a.mp3"), "a", "a.mp3")
    assert cli.handle_captions(parse("a.mp3"), source, RESULT, PLAN, tmp_path) == (None, [])


def test_available_captions_are_reported_but_not_fetched(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "download_caption", lambda *a, **k: pytest.fail("must not fetch"))
    info, notices = cli.handle_captions(
        parse("u"), youtube_source(subtitles={"en": [{}]}), RESULT, PLAN, tmp_path)
    assert info == {"manual": ["en"], "auto": []}
    assert len(notices) == 1 and "--compare-captions" in notices[0]


def test_compare_with_no_captions_says_so(tmp_path):
    _, notices = cli.handle_captions(parse("u", "--compare-captions"), youtube_source(), RESULT, PLAN, tmp_path)
    assert "no YouTube captions" in notices[0]


def test_compare_in_another_language_explains_why_nothing_was_fetched(tmp_path):
    source = youtube_source(subtitles={"de": [{}]})
    _, notices = cli.handle_captions(parse("u", "--compare-captions"), source, RESULT, PLAN, tmp_path)
    assert "detected language" in notices[0]


def test_compare_fetches_compares_and_saves(tmp_path, monkeypatch):
    track = tmp_path / "x.en.json3"
    track.write_text(json.dumps({"events": [{"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "hello world"}]}]}))
    calls = {}

    def fake_download(url, language, auto, dest, **kw):
        calls.update(language=language, auto=auto)
        return track

    monkeypatch.setattr(remote, "download_caption", fake_download)
    plan = cli.OutputPlan({"txt": tmp_path / "out.transcript.txt"})
    info, notices = cli.handle_captions(
        parse("u", "--compare-captions"), youtube_source(subtitles={"en": [{}]}), RESULT, plan, tmp_path)
    assert calls == {"language": "en", "auto": False}
    assert info["comparison"]["similarity"] == pytest.approx(2 / 3, abs=1e-3)
    assert "66.7%" in notices[0]
    assert (tmp_path / "out.transcript.youtube-manual-en.txt").read_text() == "hello world\n"


def test_failed_caption_fetch_is_a_notice_not_a_failure(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise remote.DownloadError("429 too many requests")

    monkeypatch.setattr(remote, "download_caption", boom)
    _, notices = cli.handle_captions(
        parse("u", "--compare-captions"), youtube_source(subtitles={"en": [{}]}), RESULT, PLAN, tmp_path)
    assert "could not fetch" in notices[0] and "429" in notices[0]


# --- full pipeline with a fake model ---------------------------------------- #

@needs_ffmpeg
def test_end_to_end_writes_transcript_and_metadata(pipeline, tone, tmp_path, capsys):
    out = tmp_path / "result"
    assert cli.main([str(tone), "-o", str(out), "--timestamps"]) == 0
    assert (out.with_suffix(".txt")).read_text() == "[00:00] Hello from the test.\n"
    meta = json.loads(out.with_suffix(".json").read_text())["meta"]
    assert meta["language"] == "en" and meta["asr"]["model"] == "large-v3"
    assert meta["asr"]["backend"] == "local" and meta["asr"]["hardware"] == "CPU"
    assert meta["source"] == {"kind": "file", "title": "talk.wav", "url": None}
    assert meta["diarization"]["status"] == "skipped" and meta["quality"]["verdict"] == "ok"
    assert pipeline.closed
    stdout = capsys.readouterr().out
    assert "Language: en (99%)" in stdout and "on CPU [local]" in stdout
    assert "Speaker labels: skipped - no Hugging Face token found" in stdout


@needs_ffmpeg
def test_options_reach_the_model(pipeline, tone, tmp_path):
    cli.main([str(tone), "-o", str(tmp_path / "r"), "--language", "de", "--prompt", "Zorblax",
              "--beam-size", "9", "--condition-on-previous", "--normalize"])
    assert pipeline.kwargs["language"] == "de" and pipeline.kwargs["prompt"] == "Zorblax"
    assert pipeline.kwargs["beam_size"] == 9 and pipeline.kwargs["condition_on_previous_text"] is True


@needs_ffmpeg
def test_stdout_output_keeps_the_transcript_alone_on_stdout(pipeline, tone, capsys):
    assert cli.main([str(tone), "-o", "-"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "Hello from the test.\n"
    assert "Language:" in captured.err


@needs_ffmpeg
def test_overwrite_is_refused_before_any_model_work(tone, tmp_path, monkeypatch, capsys):
    (tmp_path / "r.txt").write_text("keep me")
    monkeypatch.setattr(job, "choose_profile", lambda *a: pytest.fail("must not load a model"))
    assert cli.main([str(tone), "-o", str(tmp_path / "r.txt")]) == 11
    assert (tmp_path / "r.txt").read_text() == "keep me"
    assert "--overwrite" in capsys.readouterr().err


@needs_ffmpeg
def test_keep_audio_saves_the_decoded_wav(pipeline, tone, tmp_path):
    kept = tmp_path / "kept.wav"
    cli.main([str(tone), "-o", str(tmp_path / "r"), "--keep-audio", str(kept)])
    assert kept.stat().st_size > 1000


@needs_ffmpeg
def test_low_language_confidence_adds_a_notice(tone, tmp_path, pipeline, capsys):
    pipeline.result = TranscriptionResult(pipeline.segments, "en", 0.55, 3.0, "large-v3", "cpu", "int8")
    cli.main([str(tone), "-o", str(tmp_path / "r")])
    assert "language detection is uncertain" in capsys.readouterr().out


@needs_ffmpeg
@pytest.mark.parametrize("policy, code, writes", [("warn", 0, True), ("skip", 0, False), ("fail", 12, False)])
def test_bad_recording_policies(pipeline, tone, tmp_path, policy, code, writes, capsys):
    pipeline.segments, pipeline.result = [], TranscriptionResult([], "en", 0.99, 60.0, "large-v3", "cpu", "int8")
    out = tmp_path / "r"
    assert cli.main([str(tone), "-o", str(out), "--on-bad", policy]) == code
    assert out.with_suffix(".txt").exists() is writes
    assert "no speech detected" in capsys.readouterr().err


@needs_ffmpeg
def test_video_without_audio_exits_10(tmp_path, capsys):
    video = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1", str(video)], check=True)
    assert cli.main([str(video), "-o", str(tmp_path / "r")]) == 10
    assert "no audio stream" in capsys.readouterr().err


@pytest.mark.usefixtures("ffmpeg_not_needed")
def test_unknown_input_exits_2(tmp_path, capsys):
    assert cli.main([str(tmp_path / "nope.mp3")]) == 2
    assert "not found" in capsys.readouterr().err


def test_missing_input_exits_2(capsys):
    assert cli.main([]) == 2
    assert "no input" in capsys.readouterr().err


@pytest.mark.usefixtures("ffmpeg_not_needed")
def test_private_urls_are_blocked_before_any_download(monkeypatch, capsys):
    monkeypatch.setattr(remote, "extract_info", lambda *a, **k: pytest.fail("must not reach the network"))
    assert cli.main(["http://127.0.0.1:8080/secret.mp3"]) == 2
    assert "not a public host" in capsys.readouterr().err


def test_keyboard_interrupt_exits_130(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run", lambda args: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert cli.main(["x.mp3"]) == 130


def test_bad_recording_error_maps_to_its_exit_code(monkeypatch):
    monkeypatch.setattr(cli, "run", lambda args: (_ for _ in ()).throw(BadRecordingError("empty")))
    assert cli.main(["x.mp3"]) == 12


# --- choosing where to run ---------------------------------------------------- #

class FakeModalBackend:
    """Records how the CLI built it, and answers like a rented GPU would."""

    name = "modal"
    instances = []

    def __init__(self, **kwargs):
        self.kwargs, self.options = kwargs, None
        FakeModalBackend.instances.append(self)

    def run(self, wav, options, log):
        self.options = options
        log.notice("Uploading the audio (fake)")
        return JobResult(
            [{"start": 0.0, "end": 2.0, "text": "Hello from the cloud.", "speaker": None,
              "avg_logprob": -0.1, "no_speech_prob": 0.01, "compression_ratio": 1.1}],
            "en", 0.99, 3.0, "large-v3", "cuda", "float16", backend="modal", hardware="NVIDIA L4")


@pytest.fixture
def cloud(pipeline, monkeypatch):
    FakeModalBackend.instances.clear()
    monkeypatch.setattr(cli, "ModalBackend", FakeModalBackend)
    monkeypatch.setattr(cli, "cloud_problem", lambda: None)  # credentials present
    monkeypatch.delenv(cli.BACKEND_ENV, raising=False)
    return FakeModalBackend


@needs_ffmpeg
def test_nothing_is_uploaded_by_default_even_when_the_cloud_is_ready(cloud, pipeline, tone, tmp_path, capsys):
    assert cli.main([str(tone), "-o", str(tmp_path / "r")]) == 0
    assert cloud.instances == [] and "on CPU [local]" in capsys.readouterr().out


@needs_ffmpeg
def test_local_runs_only_suggest_the_cloud_for_long_audio(cloud, pipeline, tone, tmp_path, capsys):
    cli.main([str(tone), "-o", str(tmp_path / "a"), "--min-cloud-minutes", "0.001"])
    out = capsys.readouterr().out
    assert "Note: No GPU on this machine" in out and "--backend auto" in out and cloud.instances == []


@needs_ffmpeg
def test_backend_modal_runs_in_the_cloud_with_the_users_settings(cloud, tone, tmp_path, capsys):
    argv = [str(tone), "-o", str(tmp_path / "r"), "--backend", "modal", "--cloud-gpu", "L40S",
            "--cloud-max-minutes", "30", "--cloud-hf-secret", "my-hf", "--language", "en"]
    assert cli.main(argv) == 0
    (backend,) = cloud.instances
    assert backend.kwargs == {"gpu": "L40S", "max_minutes": 30, "hf_secret": "my-hf", "quiet": False}
    assert backend.options.language == "en"
    out = capsys.readouterr()
    assert "on NVIDIA L4 [modal]" in out.out and "Uploading the audio" in out.err
    meta = json.loads((tmp_path / "r.json").read_text())["meta"]
    assert meta["asr"]["backend"] == "modal" and meta["asr"]["hardware"] == "NVIDIA L4"


@needs_ffmpeg
def test_auto_uses_the_cloud_for_long_audio_without_a_gpu(cloud, tone, tmp_path, capsys):
    cli.main([str(tone), "-o", str(tmp_path / "r"), "--backend", "auto", "--min-cloud-minutes", "0.001"])
    assert len(cloud.instances) == 1 and "No GPU on this machine: using a cloud GPU" in capsys.readouterr().out


@needs_ffmpeg
def test_the_note_names_the_cloud_environment_it_runs_in(cloud, tone, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "detect_environment", lambda: Environment("Google Colab", True))
    cli.main([str(tone), "-o", str(tmp_path / "r"), "--backend", "auto", "--min-cloud-minutes", "0.001"])
    assert "No GPU on this machine (Google Colab)" in capsys.readouterr().out


@needs_ffmpeg
def test_auto_keeps_short_audio_local(cloud, tone, tmp_path):
    cli.main([str(tone), "-o", str(tmp_path / "r"), "--backend", "auto"])
    assert cloud.instances == []


@needs_ffmpeg
def test_auto_prefers_a_local_gpu(cloud, tone, tmp_path, monkeypatch):
    monkeypatch.setattr(job, "local_gpu_available", lambda options: True)
    cli.main([str(tone), "-o", str(tmp_path / "r"), "--backend", "auto", "--min-cloud-minutes", "0.001"])
    assert cloud.instances == []


@needs_ffmpeg
def test_auto_falls_back_to_the_cpu_when_the_cloud_is_not_set_up(cloud, tone, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "cloud_problem", lambda: NOT_READY)
    assert cli.main([str(tone), "-o", str(tmp_path / "r"), "--backend", "auto",
                     "--min-cloud-minutes", "0.001"]) == 0
    assert "using the CPU" in capsys.readouterr().out and cloud.instances == []


def test_backend_modal_without_credentials_fails_immediately(cloud, tone, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "cloud_problem", lambda: NOT_READY)
    monkeypatch.setattr(cli.sources, "resolve_input", lambda *a, **k: pytest.fail("must fail before touching the input"))
    assert cli.main([str(tone), "-o", str(tmp_path / "r"), "--backend", "modal"]) == 6
    assert "modal token new" in capsys.readouterr().err and cloud.instances == []


@needs_ffmpeg
def test_the_backend_can_be_set_by_environment_for_cloud_sessions(cloud, tone, tmp_path, monkeypatch):
    monkeypatch.setenv(cli.BACKEND_ENV, "modal")
    cli.main([str(tone), "-o", str(tmp_path / "r")])
    assert len(cloud.instances) == 1


@pytest.mark.parametrize("extra, message", [
    (["--backend", "modal", "--offline"], "--offline forbids network use"),
    (["--backend", "auto", "--device", "cpu"], "cloud chooses its own"),
    (["--backend", "tpu"], "unknown backend"),
])
def test_conflicting_backend_options_are_rejected_up_front(extra, message, capsys, tone):
    assert cli.main([str(tone), *extra]) == 2
    assert message in capsys.readouterr().err


# --- diagnostics ------------------------------------------------------------ #

def test_check_reports_and_sets_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(cli, "detect_hardware", lambda cache: make_hw())
    monkeypatch.setattr(cli, "cloud_problem", lambda: NOT_READY)
    ok = [preflight.Check("A", preflight.OK, "fine")]
    monkeypatch.setattr(preflight, "run_checks", lambda *a, **k: ok)
    assert cli.main(["--check"]) == 0
    assert "ready to transcribe" in capsys.readouterr().out
    bad = [preflight.Check("B", preflight.FAIL, "broken", "fix it")]
    monkeypatch.setattr(preflight, "run_checks", lambda *a, **k: bad)
    assert cli.main(["--check"]) == 1
    assert "fix it" in capsys.readouterr().out


@needs_ffmpeg
def test_self_test_runs_the_chosen_backend_with_a_synthetic_tone(pipeline, capsys):
    pipeline.result = TranscriptionResult([], "en", 0.9, 3.0, "large-v3", "cpu", "int8")
    assert cli.main(["--self-test"]) == 0
    assert "SELF-TEST PASSED: large-v3 (int8) on CPU via local" in capsys.readouterr().out
    assert pipeline.closed


@needs_ffmpeg
def test_cloud_self_test_never_uploads_anything_but_a_tone_and_skips_speaker_labels(cloud, capsys):
    assert cli.main(["--self-test", "--backend", "modal"]) == 0
    (backend,) = cloud.instances
    assert backend.options.diarize == "off"
    assert "via modal" in capsys.readouterr().out


def test_test_tone_is_a_valid_16khz_wav(tmp_path):
    import wave

    path = tmp_path / "t.wav"
    cli.write_test_tone(path, seconds=0.5)
    with wave.open(str(path)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getnframes()) == (16000, 1, 8000)


def test_parser_rejects_unsupported_browser(capsys):
    with pytest.raises(SystemExit) as exc:
        parse("u", "--cookies-from-browser", "firefox:/etc")
    assert exc.value.code == 2


# --- the comparison measures words, not labels --------------------------------------------------------------- #

def caption_track(tmp_path, text):
    track = tmp_path / "x.en.json3"
    track.write_text(json.dumps({"events": [{"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": text}]}]}))
    return track


def labelled(speaker):
    return SimpleNamespace(language="en", segments=[{"text": "hello big world", "speaker": speaker}])


@pytest.mark.parametrize("speaker", [None, "Speaker 1"])
def test_speaker_labels_do_not_change_the_similarity_score(tmp_path, monkeypatch, speaker):
    monkeypatch.setattr(remote, "download_caption", lambda *a, **k: caption_track(tmp_path, "hello big world"))
    info, _ = cli.handle_captions(parse("u", "--compare-captions"), youtube_source(subtitles={"en": [{}]}),
                                  labelled(speaker), cli.OutputPlan({}), tmp_path)
    assert info["comparison"]["similarity"] == 1.0 and info["comparison"]["our_words"] == 3


def test_a_saved_caption_file_is_never_replaced_without_permission(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "download_caption", lambda *a, **k: caption_track(tmp_path, "hello big world"))
    existing = tmp_path / "out.transcript.youtube-manual-en.txt"
    existing.write_text("precious")
    plan = cli.OutputPlan({"txt": tmp_path / "out.transcript.txt"})
    args = ("u", "--compare-captions")
    _, notices = cli.handle_captions(parse(*args), youtube_source(subtitles={"en": [{}]}), labelled(None), plan, tmp_path)
    assert existing.read_text() == "precious" and any("already exists" in n and "--overwrite" in n for n in notices)
    cli.handle_captions(parse(*args, "--overwrite"), youtube_source(subtitles={"en": [{}]}), labelled(None), plan, tmp_path)
    assert existing.read_text() == "hello big world\n"


def test_asking_to_compare_a_local_file_says_why_nothing_happened(tmp_path):
    source = sources.MediaSource(Path("a.mp3"), "a", "a.mp3")
    info, notices = cli.handle_captions(parse("a.mp3", "--compare-captions"), source, RESULT, PLAN, tmp_path)
    assert info is None and "local file" in notices[0] and "--compare-captions" in notices[0]
    assert cli.handle_captions(parse("a.mp3"), source, RESULT, PLAN, tmp_path) == (None, [])  # silent otherwise


def test_the_url_in_the_metadata_goes_through_the_cleaner():
    leaky = source_with("https://bucket.example/a.mp3?X-Amz-Signature=SECRET#frag", extractor_key="Generic", id="a")
    result = JobResult([], "en", 0.9, 1.0, "large-v3", "cpu", "int8")
    report = cli.quality.assess([], 1.0)
    assert cli.build_meta(leaky, result, report, None, [])["source"]["url"] == "https://bucket.example/a.mp3"
