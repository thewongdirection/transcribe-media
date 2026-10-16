"""Every command-line option must actually change what happens, and bad input must fail early and cleanly."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from transcribe_media import audio, cli, remote, sources
from transcribe_media.errors import (
    AudioError,
    InputError,
    NoAudioStreamError,
    OverwriteRefusedError,
    WriteError,
)

from .conftest import NOT_READY

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def parse(*argv):
    return cli.build_parser().parse_args(list(argv))


def recorded(monkeypatch, module, name):
    """Replace ``module.name`` with a recorder that returns what it is given back untouched."""
    calls = []

    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args[0] if args else [], 0, "", "")

    monkeypatch.setattr(module, name, record)
    return calls


# --- ffmpeg is told what the user asked for ---------------------------------------------------- #

def test_decoding_is_plain_unless_normalising_or_capping(monkeypatch):
    calls = recorded(monkeypatch, audio, "_run")
    audio.decode_to_wav(Path("in.mp3"), Path("out.wav"))
    (command,), _ = calls[0]
    assert "-af" not in command and "-t" not in command
    assert command[command.index("-ac") + 1] == "1" and command[command.index("-ar") + 1] == "16000"


def test_normalising_adds_the_loudness_filter(monkeypatch):
    calls = recorded(monkeypatch, audio, "_run")
    audio.decode_to_wav(Path("in.mp3"), Path("out.wav"), normalize=True)
    (command,), _ = calls[0]
    assert command[command.index("-af") + 1].startswith("loudnorm")


def test_the_duration_cap_limits_how_much_audio_is_decoded(monkeypatch):
    calls = recorded(monkeypatch, audio, "_run")
    audio.decode_to_wav(Path("in.mp3"), Path("out.wav"), max_seconds=7200)
    (command,), _ = calls[0]
    assert command[command.index("-t") + 1] == "7200.000"
    assert command.index("-t") > command.index("-i")  # an output option: it limits what is written


def test_every_ffmpeg_and_ffprobe_call_is_restricted_to_plain_files_and_pipes(monkeypatch):
    calls = recorded(monkeypatch, audio, "_run")
    monkeypatch.setattr(audio, "_probe_json", audio._probe_json)  # the real one, using the recorder
    audio.decode_to_wav(Path("in.mp3"), Path("out.wav"))
    audio.encode_flac(Path("in.wav"), Path("out.flac"))
    with pytest.raises(NoAudioStreamError):  # the recorder returns empty output; only the command matters
        audio.ensure_audio_stream(Path("in.mp3"))
    assert len(calls) == 3
    for (command,), _ in calls:
        assert command[command.index("-protocol_whitelist") + 1] == "file,pipe"


@needs_ffmpeg
def test_a_playlist_or_stream_description_is_refused_instead_of_followed(tmp_path):
    playlist = tmp_path / "list.m3u8"
    playlist.write_text("#EXTM3U\n#EXT-X-TARGETDURATION:5\n#EXTINF:5,\nsegment.ts\n#EXT-X-ENDLIST\n")
    with pytest.raises(AudioError):
        audio.ensure_audio_stream(playlist)


@needs_ffmpeg
def test_a_concat_list_of_local_files_is_refused_even_though_ffmpeg_can_read_it(tone, tmp_path):
    """The list names a neighbouring file, which ffmpeg happily reads; only our format check stops it."""
    concat = tmp_path / "list.ffconcat"
    concat.write_text(f"ffconcat version 1.0\nfile {tone.name}\n")
    control = subprocess.run(["ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe", "-show_entries",
                              "format=format_name", "-of", "csv=p=0", str(concat)], capture_output=True, text=True)
    assert control.returncode == 0 and "concat" in control.stdout  # readable, so the refusal below is ours
    with pytest.raises(AudioError, match="playlist or stream description"):
        audio.ensure_audio_stream(concat)


@needs_ffmpeg
def test_the_declared_duration_comes_back_with_the_stream_check(tone):
    info = audio.ensure_audio_stream(tone)
    assert info.duration == pytest.approx(1.0, abs=0.1) and info.format_name


# --- cut-off recordings ---------------------------------------------------------------------------- #

@pytest.mark.parametrize("declared, decoded, cap, warned", [
    (3600, 3590, 36000, False),      # normal rounding
    (3600, 1800, 36000, True),       # half the file is missing
    (3600, 3419, 36000, True),       # just under 95%
    (3600, 3420, 36000, False),      # exactly at the limit
    (36000, 3600, 3600, False),      # the user capped it on purpose, and the cap was reached
    (36000, 1800, 3600, True),       # capped, but even the cap was not reached
    (None, 100, 36000, False),       # nothing declared, nothing to compare
    (3600, None, 36000, False),
])
def test_truncation_warning(declared, decoded, cap, warned):
    notice = cli.truncation_notice(declared, decoded, cap)
    assert (notice is not None) is warned
    if warned:
        assert "cut off or damaged" in notice


# --- duration limits ------------------------------------------------------------------------------ #

@needs_ffmpeg
def test_media_longer_than_the_limit_is_refused_before_decoding(pipeline, tone, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(audio, "decode_to_wav", lambda *a, **k: pytest.fail("must refuse before decoding"))
    assert cli.main([str(tone), "-o", str(tmp_path / "r"), "--max-duration-hours", "0.0001"]) == 2
    assert "exceeds --max-duration-hours" in capsys.readouterr().err


@needs_ffmpeg
def test_the_limit_also_caps_the_decode_in_case_the_container_lies(pipeline, tone, tmp_path, monkeypatch):
    seen = []
    real = audio.decode_to_wav
    monkeypatch.setattr(audio, "decode_to_wav", lambda *a, **k: (seen.append(k.get("max_seconds")), real(*a, **k)))
    cli.main([str(tone), "-o", str(tmp_path / "r"), "--max-duration-hours", "2"])
    assert seen == [7200.0]


@pytest.mark.parametrize("value", ["0", "-1"])
def test_a_non_positive_limit_is_rejected(tone, value, capsys):
    assert cli.main([str(tone), "--max-duration-hours", value]) == 2
    assert "must be positive" in capsys.readouterr().err


@needs_ffmpeg
def test_a_cut_off_file_is_flagged_in_the_notes(pipeline, tone, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(audio, "probe_duration", lambda path: 0.2)  # far less than the 1 s the file declares
    cli.main([str(tone), "-o", str(tmp_path / "r")])
    assert "cut off or damaged" in capsys.readouterr().out


# --- decoding choices reach the model --------------------------------------------------------------- #

@needs_ffmpeg
def test_voice_activity_filtering_is_on_unless_switched_off(pipeline, tone, tmp_path):
    cli.main([str(tone), "-o", str(tmp_path / "a")])
    assert pipeline.kwargs["vad_filter"] is True
    cli.main([str(tone), "-o", str(tmp_path / "b"), "--no-vad"])
    assert pipeline.kwargs["vad_filter"] is False


@needs_ffmpeg
def test_the_self_test_forces_the_decoder_to_run(pipeline, capsys):
    """A tone has no speech, so voice filtering would skip the decoder and prove nothing."""
    assert cli.main(["--self-test"]) == 0
    assert pipeline.kwargs["vad_filter"] is False and pipeline.kwargs["language"] == "en"


@needs_ffmpeg
def test_offline_and_no_download_reach_the_job(pipeline, tone, tmp_path, monkeypatch):
    seen = []
    real = cli.job_options
    monkeypatch.setattr(cli, "job_options", lambda args: seen.append(real(args)) or seen[-1])
    cli.main([str(tone), "-o", str(tmp_path / "r"), "--offline", "--no-download"])
    assert seen[0].offline is True and seen[0].no_download is True


def test_the_check_command_respects_offline_and_the_chosen_model(monkeypatch, capsys):
    import dataclasses

    from transcribe_media import preflight

    from .conftest import make_hw

    seen = {}
    monkeypatch.setattr(cli, "detect_hardware", lambda cache: make_hw())
    monkeypatch.setattr(cli, "cloud_problem", lambda: NOT_READY)

    def run_checks(hw, profile, **kwargs):
        seen.update(kwargs, profile=profile)
        return []

    monkeypatch.setattr(preflight, "run_checks", run_checks)
    monkeypatch.setattr(preflight, "format_report", lambda checks: "")
    monkeypatch.setattr(preflight, "has_failures", lambda checks: False)
    cli.main(["--check", "--offline", "--model", "small"])
    assert seen["offline"] is True and seen["profile"].model == "small"
    assert dataclasses.is_dataclass(seen["environment"])


# --- arguments are validated by the parser ---------------------------------------------------------- #

@pytest.mark.parametrize("argv", [
    ["x", "--language", "english"], ["x", "--language", "e1"],
    ["x", "--beam-size", "0"], ["x", "--beam-size", "999"], ["x", "--beam-size", "five"],
    ["x", "--num-speakers", "0"], ["x", "--max-speakers", "100000"],
    ["x", "--cloud-max-minutes", "0"], ["x", "--cloud-max-minutes", "99999"],
    ["x", "--max-download-mb", "0"], ["x", "--max-download-mb", "nine"],
    ["x", "--device", "tpu"], ["x", "--compute-type", "int4"], ["x", "--on-bad", "explode"],
])
def test_nonsense_values_are_rejected_before_any_work_starts(argv, capsys):
    with pytest.raises(SystemExit) as exit_info:
        parse(*argv)
    assert exit_info.value.code == 2 and "error" in capsys.readouterr().err


def test_a_language_code_is_normalised():
    assert parse("x", "--language", " DE ").language == "de"


def test_conflicting_speaker_counts_are_an_input_error(tone, capsys):
    assert cli.main([str(tone), "--min-speakers", "5", "--max-speakers", "2"]) == 2
    assert "min speakers exceeds max speakers" in capsys.readouterr().err


def test_a_local_model_folder_cannot_be_sent_to_the_cloud(tone, capsys, monkeypatch):
    monkeypatch.setattr(cli, "cloud_problem", lambda: None)
    assert cli.main([str(tone), "--backend", "modal", "--model", "/home/me/my-model"]) == 2
    assert "built-in models" in capsys.readouterr().err


def test_a_cloud_run_may_pick_any_built_in_model(tone, monkeypatch, capsys):
    monkeypatch.setattr(cli, "cloud_problem", lambda: None)
    args = parse(str(tone), "--backend", "modal", "--model", "medium")
    cli.validate_backend_args(args)


# --- offline means offline ---------------------------------------------------------------------------- #

def test_offline_cannot_be_combined_with_a_url(monkeypatch, capsys):
    monkeypatch.setattr(sources, "resolve_input", lambda *a, **k: pytest.fail("must refuse first"))
    assert cli.main(["https://example.com/a.mp3", "--offline"]) == 2
    assert "--offline forbids network use" in capsys.readouterr().err


def test_offline_cannot_be_combined_with_caption_comparison(tone, capsys):
    assert cli.main([str(tone), "--offline", "--compare-captions"]) == 2
    assert "--offline forbids network use" in capsys.readouterr().err


# --- network settings reach every download -------------------------------------------------------------- #

def stop_after_resolving(monkeypatch):
    seen = {}

    def resolve(text, **kwargs):
        seen.update(kwargs)
        raise InputError("stop here")

    monkeypatch.setattr(sources, "resolve_input", resolve)
    return seen


@pytest.mark.usefixtures("ffmpeg_not_needed")
def test_download_settings_reach_the_input_resolver(monkeypatch):
    seen = stop_after_resolving(monkeypatch)
    cli.main(["https://example.com/a.mp3", "--max-download-mb", "5", "--allow-private-hosts",
              "--cookies-from-browser", "firefox"])
    assert seen["max_download_mb"] == 5 and seen["allow_private_hosts"] is True
    assert seen["cookies_from_browser"] == "firefox"


@pytest.mark.usefixtures("ffmpeg_not_needed")
def test_cookies_are_off_unless_asked_for(monkeypatch):
    seen = stop_after_resolving(monkeypatch)
    cli.main(["https://example.com/a.mp3"])
    assert seen["cookies_from_browser"] is None and seen["allow_private_hosts"] is False
    assert seen["max_download_mb"] == 2048


def test_the_caption_download_uses_the_same_network_settings(tmp_path, monkeypatch):
    from types import SimpleNamespace

    seen = {}

    def download(url, language, auto, dest, **kwargs):
        seen.update(kwargs)
        raise remote.DownloadError("stop here")

    monkeypatch.setattr(remote, "download_caption", download)
    source = sources.MediaSource(Path("a"), "a", "t", url="https://www.youtube.com/watch?v=x", info={
        "extractor_key": "Youtube", "subtitles": {"en": [{}]}, "automatic_captions": {}})
    result = SimpleNamespace(language="en", segments=[{"text": "hello", "speaker": None}])
    cli.handle_captions(parse("u", "--compare-captions", "--cookies-from-browser", "chrome", "--allow-private-hosts"),
                        source, result, cli.OutputPlan({}), tmp_path)
    assert seen == {"cookies_from_browser": "chrome", "allow_private_hosts": True}


# --- output files: checked early, written atomically ------------------------------------------------------- #

def test_a_file_name_the_file_system_cannot_hold_fails_now_not_after_the_transcription(tmp_path):
    too_long = tmp_path / ("x" * 300 + ".transcript.txt")
    with pytest.raises(WriteError, match="cannot"):
        cli.plan_outputs(str(too_long), "stem", ["txt"], overwrite=False)


def test_an_unusable_output_folder_fails_early(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(WriteError, match="cannot write"):
        cli.plan_outputs(str(blocker / "sub" / "r.txt"), "stem", ["txt"], overwrite=False)


def test_checking_a_path_leaves_nothing_behind_but_the_folder(tmp_path):
    target = tmp_path / "new" / "deeper" / "r.txt"
    cli.check_writable(target)
    assert (tmp_path / "new" / "deeper").is_dir() and list((tmp_path / "new" / "deeper").iterdir()) == []


def test_a_trailing_slash_names_a_folder_that_is_created(tmp_path):
    plan = cli.plan_outputs(str(tmp_path / "out") + "/", "my-talk", ["txt", "json"], overwrite=False)
    assert plan.files["txt"] == tmp_path / "out" / "my-talk.transcript.txt"
    assert (tmp_path / "out").is_dir()


def test_an_interrupted_write_leaves_neither_a_partial_file_nor_a_temporary_one(tmp_path, monkeypatch):
    target = tmp_path / "r.txt"
    target.write_text("previous")

    def explode(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", explode)
    with pytest.raises(WriteError, match="disk full"):
        cli.write_text(target, "new text")
    assert target.read_text() == "previous" and [p.name for p in tmp_path.iterdir()] == ["r.txt"]


def test_text_is_written_with_unix_line_endings_whatever_the_platform(tmp_path):
    target = tmp_path / "r.txt"
    cli.write_text(target, "a\nb\n")
    assert target.read_bytes() == b"a\nb\n"


# --- --keep-audio ---------------------------------------------------------------------------------------- #

def test_keeping_audio_is_optional():
    assert cli.plan_audio_copy(None, overwrite=False) is None


@pytest.mark.parametrize("name", ["copy.mp3", "copy", "copy.wav.sh", "../escape.txt"])
def test_only_wav_names_are_accepted_for_the_audio_copy(tmp_path, name):
    with pytest.raises(InputError, match=r"ending in \.wav"):
        cli.plan_audio_copy(str(tmp_path / name), overwrite=False)


def test_the_audio_copy_will_not_write_through_a_link(tmp_path):
    victim = tmp_path / "precious.txt"
    victim.write_text("keep")
    link = tmp_path / "copy.wav"
    link.symlink_to(victim)
    with pytest.raises(InputError, match="link"):
        cli.plan_audio_copy(str(link), overwrite=True)
    assert victim.read_text() == "keep"


def test_the_audio_copy_will_not_replace_a_file_without_permission(tmp_path):
    existing = tmp_path / "copy.wav"
    existing.write_bytes(b"old")
    with pytest.raises(OverwriteRefusedError, match="--overwrite"):
        cli.plan_audio_copy(str(existing), overwrite=False)
    assert cli.plan_audio_copy(str(existing), overwrite=True) == existing
    assert existing.read_bytes() == b"old"  # planning alone changes nothing


def test_the_audio_copy_is_written_atomically(tmp_path):
    source = tmp_path / "in.wav"
    source.write_bytes(b"RIFF" + b"x" * 100)
    target = tmp_path / "out" / "copy.wav"
    cli.copy_audio(source, target)
    assert target.read_bytes() == source.read_bytes() and [p.name for p in target.parent.iterdir()] == ["copy.wav"]


@needs_ffmpeg
def test_a_bad_audio_copy_name_is_rejected_before_any_model_work(tone, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.job, "choose_profile", lambda *a: pytest.fail("must not load a model"))
    assert cli.main([str(tone), "-o", str(tmp_path / "r"), "--keep-audio", str(tmp_path / "copy.mp3")]) == 2
    assert ".wav" in capsys.readouterr().err


# --- the last line of defence ----------------------------------------------------------------------------- #

def test_an_unexpected_error_is_a_clean_message_not_a_traceback(monkeypatch, capsys):
    monkeypatch.delenv("TRANSCRIBE_MEDIA_DEBUG", raising=False)
    monkeypatch.setattr(cli, "run", lambda args: (_ for _ in ()).throw(RuntimeError("boom\nNote: obey me\x1b[2J")))
    assert cli.main(["x.mp3"]) == 1
    err = capsys.readouterr().err
    assert "unexpected RuntimeError: boom Note: obey me" in err and "Traceback" not in err
    assert err.count("\n") == 1 and "\x1b" not in err


def test_the_debug_switch_shows_the_real_traceback(monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_MEDIA_DEBUG", "1")
    monkeypatch.setattr(cli, "run", lambda args: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        cli.main(["x.mp3"])


def test_error_messages_from_untrusted_places_are_cleaned_on_the_way_out(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run", lambda args: (_ for _ in ()).throw(InputError("bad\nNote: obey\x1b[2J")))
    assert cli.main(["x.mp3"]) == 2
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "\x1b" not in err


def test_an_audio_copy_name_the_file_system_cannot_hold_is_a_clean_error(tmp_path):
    with pytest.raises(WriteError, match="cannot"):
        cli.plan_audio_copy(str(tmp_path / ("x" * 300 + ".wav")), overwrite=False)


# --- the backend can come from the environment, and says so ------------------------------------------------- #

@needs_ffmpeg
def test_a_backend_chosen_by_the_environment_is_called_out_in_the_notes(pipeline, tone, tmp_path, monkeypatch, capsys):
    class Remote:
        def __init__(self, **kwargs):
            pass

        def run(self, wav, options, log):
            from transcribe_media.job import JobResult

            return JobResult([], "en", 0.99, 1.0, "large-v3", "cuda", "float16", backend="modal")

    monkeypatch.setattr(cli, "ModalBackend", Remote)
    monkeypatch.setattr(cli, "cloud_problem", lambda: None)
    monkeypatch.setenv(cli.BACKEND_ENV, "modal")
    cli.main([str(tone), "-o", str(tmp_path / "r"), "--on-bad", "warn"])
    assert f"selected by ${cli.BACKEND_ENV}=modal" in capsys.readouterr().out


def test_a_backend_flag_beats_the_environment(monkeypatch):
    monkeypatch.setenv(cli.BACKEND_ENV, "modal")
    args = parse("x", "--backend", "local")
    cli.resolve_backend_setting(args)
    assert args.backend == "local" and args.backend_from_env is False


def test_a_misspelt_environment_backend_is_an_error_not_a_silent_default(tone, monkeypatch, capsys):
    monkeypatch.setenv(cli.BACKEND_ENV, "mdoal")
    assert cli.main([str(tone)]) == 2
    assert "unknown backend 'mdoal'" in capsys.readouterr().err


def test_the_command_line_hardens_the_executable_search_before_starting_anything(monkeypatch):
    order = []
    monkeypatch.setattr(cli, "harden_executable_search", lambda: order.append("hardened"))
    monkeypatch.setattr(cli, "run", lambda args: order.append("run") or 0)
    assert cli.main(["x.mp3"]) == 0
    assert order == ["hardened", "run"]


# --- cheap checks come before expensive work ---------------------------------------------------------------------- #

def never_reached(monkeypatch, what):
    def forbidden(name):
        def fail(*args, **kwargs):
            pytest.fail(f"{what} must be rejected before {name}")

        return fail

    monkeypatch.setattr(sources, "resolve_input", forbidden("the download"))
    monkeypatch.setattr(audio, "decode_to_wav", forbidden("the decode"))


@pytest.mark.parametrize("argv, message", [
    (["--min-speakers", "5", "--max-speakers", "2"], "min speakers exceeds max speakers"),
    (["--prompt", "x" * 2001], "prompt"),
])
def test_bad_job_options_are_rejected_before_the_download_and_the_decode(tone, monkeypatch, capsys, argv, message):
    never_reached(monkeypatch, "bad options")
    assert cli.main([str(tone), *argv]) == 2
    assert message in capsys.readouterr().err


def test_a_bad_cloud_gpu_is_rejected_before_the_download_and_the_decode(tone, monkeypatch, capsys):
    never_reached(monkeypatch, "a bad GPU")
    monkeypatch.setattr(cli, "cloud_problem", lambda: None)
    assert cli.main([str(tone), "--backend", "auto", "--cloud-gpu", "B200"]) == 2
    assert "unsupported --cloud-gpu" in capsys.readouterr().err


@needs_ffmpeg
def test_a_local_run_ignores_cloud_settings_it_will_not_use(pipeline, tone, tmp_path):
    assert cli.main([str(tone), "-o", str(tmp_path / "r"), "--cloud-gpu", "B200"]) == 0


@pytest.mark.usefixtures("ffmpeg_not_needed")
def test_an_existing_explicit_output_is_found_before_the_download(tone, tmp_path, monkeypatch, capsys):
    (tmp_path / "r.txt").write_text("keep")
    never_reached(monkeypatch, "an output collision")
    assert cli.main([str(tone), "-o", str(tmp_path / "r.txt")]) == 11
    assert (tmp_path / "r.txt").read_text() == "keep"


@pytest.mark.usefixtures("ffmpeg_not_needed")
def test_an_unwritable_explicit_output_is_found_before_the_download(tone, tmp_path, monkeypatch, capsys):
    never_reached(monkeypatch, "an unwritable output")
    assert cli.main([str(tone), "-o", str(tmp_path / ("x" * 300 + ".txt"))]) == 9


def test_a_folder_output_still_waits_for_the_title_that_names_the_files(tmp_path):
    assert cli.plan_known_output(str(tmp_path) + "/", ["txt"], overwrite=False) is None
    assert cli.plan_known_output(str(tmp_path / "new") + "/", ["txt"], overwrite=False) is None
    assert cli.plan_known_output(None, ["txt"], overwrite=False) is None
    assert cli.plan_known_output("-", ["txt"], overwrite=False) is None
    assert cli.plan_known_output(str(tmp_path / "r"), ["txt", "json"], overwrite=False).files.keys() == {"txt", "json"}


def test_what_names_a_folder(tmp_path):
    assert cli.names_a_folder(str(tmp_path)) and cli.names_a_folder("new/") and not cli.names_a_folder("r.txt")
