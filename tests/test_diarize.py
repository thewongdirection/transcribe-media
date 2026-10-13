import json
import subprocess
import sys
from pathlib import Path

import pytest
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

from transcribe_media import diarize
from transcribe_media.diarize import merge_turns, prepare_diarization, run_worker
from transcribe_media.errors import DiarizeError, ModelError

SECRET = "placeholder-credential-for-tests"


def hub_error(cls, text="boom"):
    """Build a huggingface_hub error without needing a real HTTP response."""
    err = cls.__new__(cls)
    Exception.__init__(err, text)
    return err


def never(*args, **kwargs):
    raise AssertionError("must not be called")


def setup(**overrides):
    defaults = dict(
        offline=False, allow_download=True,
        installed=lambda module: True,
        token_getter=lambda: SECRET,
        auth_check=lambda repo, token: None,
        cached=lambda spec: None,
        fetch=lambda spec, token=None: "/models/community-1",
    )
    defaults.update(overrides)
    return prepare_diarization(**defaults)


def test_skipped_when_pyannote_is_missing():
    result = setup(installed=lambda module: False, cached=never, token_getter=never)
    assert not result.available and "setup.py" in result.skip_reason


def test_cached_model_needs_no_token_or_network():
    result = setup(cached=lambda spec: "/cache/dir", token_getter=never, auth_check=never, fetch=never)
    assert result.available and result.pipeline_dir == "/cache/dir"


@pytest.mark.parametrize("flags", [{"offline": True}, {"allow_download": False}])
def test_uncached_model_is_skipped_when_downloads_are_forbidden(flags):
    result = setup(token_getter=never, **flags)
    assert not result.available and "downloading is disabled" in result.skip_reason


def test_no_token_means_skip_with_instructions():
    result = setup(token_getter=lambda: None, auth_check=never, fetch=never)
    assert not result.available
    assert "HF_TOKEN" in result.skip_reason and diarize.GATED_URL in result.skip_reason


@pytest.mark.parametrize("error, fragment", [
    (hub_error(GatedRepoError), "accept"),
    (hub_error(RepositoryNotFoundError), "invalid"),
    (hub_error(HfHubHTTPError, "401"), "rejected"),
    (ConnectionError("offline"), "could not verify"),
])
def test_unsuitable_token_is_skipped_before_any_download(error, fragment):
    def auth_check(repo, token):
        raise error

    result = setup(auth_check=auth_check, fetch=never)
    assert not result.available and fragment in result.skip_reason


def test_suitable_token_downloads_and_uses_the_model():
    calls = {}

    def fetch(spec, token=None):
        calls["token"], calls["repo"] = token, spec.repo_id
        return "/models/community-1"

    result = setup(fetch=fetch)
    assert result.available and result.pipeline_dir == "/models/community-1"
    assert calls == {"token": SECRET, "repo": "pyannote/speaker-diarization-community-1"}


def test_download_failure_is_a_skip_not_a_crash():
    def fetch(spec, token=None):
        raise ModelError("could not download model: network down")

    result = setup(fetch=fetch)
    assert not result.available and "network down" in result.skip_reason


@pytest.mark.parametrize("scenario", ["no_pkg", "no_token", "gated", "invalid", "http", "network", "fetch"])
def test_the_token_never_appears_in_skip_reasons(scenario):
    def raising(error):
        def auth_check(repo, token):
            raise error
        return auth_check

    overrides = {
        "no_pkg": dict(installed=lambda m: False),
        "no_token": dict(token_getter=lambda: None),
        "gated": dict(auth_check=raising(hub_error(GatedRepoError))),
        "invalid": dict(auth_check=raising(hub_error(RepositoryNotFoundError))),
        "http": dict(auth_check=raising(hub_error(HfHubHTTPError, "401"))),
        "network": dict(auth_check=raising(OSError("down"))),
        "fetch": dict(fetch=lambda spec, token=None: (_ for _ in ()).throw(ModelError("failed"))),
    }[scenario]
    assert SECRET not in (setup(**overrides).skip_reason or "")


# --- worker boundary -------------------------------------------------------- #

def completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def test_worker_command_and_json_parsing():
    seen = {}

    def spawn(cmd):
        seen["cmd"] = cmd
        noise = "some library banner\n"
        return completed(stdout=noise + json.dumps({"turns": [[0, 1.5, "A"], [1.5, 3, "B"]], "note": "cpu"}) + "\n")

    turns, note = run_worker("a.wav", "/m", device="cuda", device_index=1, num_speakers=2, spawn=spawn)
    assert turns == [(0.0, 1.5, "A"), (1.5, 3.0, "B")] and note == "cpu"
    cmd = seen["cmd"]
    # -I keeps the current directory and PYTHON* settings out of the worker's import path; the package
    # location is handed over explicitly because -I also ignores the PYTHONPATH a cloud container relies on.
    assert cmd[:5] == [sys.executable, "-I", "-c", diarize.WORKER_BOOTSTRAP, diarize.PACKAGE_ROOT]
    assert (Path(diarize.PACKAGE_ROOT) / "transcribe_media" / "diarize_worker.py").is_file()
    assert cmd[cmd.index("--device") + 1] == "cuda" and cmd[cmd.index("--device-index") + 1] == "1"
    assert cmd[cmd.index("--num-speakers") + 1] == "2"
    assert "--min-speakers" not in cmd and "--max-speakers" not in cmd


def test_worker_failure_surfaces_last_stderr_line():
    def spawn(cmd):
        return completed(1, stderr="Traceback...\nOutOfMemoryError: CUDA out of memory")

    with pytest.raises(DiarizeError, match="exit 1.*CUDA out of memory"):
        run_worker("a.wav", "/m", device="cuda", device_index=0, spawn=spawn)


@pytest.mark.parametrize("stdout", ["", "not json", json.dumps({"nope": 1}), json.dumps({"turns": [[1, 2]]})])
def test_worker_garbage_is_a_diarize_error(stdout):
    with pytest.raises(DiarizeError, match="unreadable"):
        run_worker("a.wav", "/m", device="cpu", device_index=0, spawn=lambda cmd: completed(stdout=stdout))


# --- merge ------------------------------------------------------------------ #

def w(text, start, end):
    return {"word": text, "start": start, "end": end, "probability": 0.9}


def segment(start, end, text, words=None, **extra):
    seg = {"start": start, "end": end, "text": text, "speaker": None, **extra}
    if words:
        seg["words"] = words
    return seg


TURNS = [(0.0, 5.0, "SPEAKER_01"), (5.0, 10.0, "SPEAKER_00")]


def test_no_turns_leaves_segments_unchanged():
    segs = [segment(0, 1, "hi")]
    assert merge_turns(segs, []) is segs


def test_speakers_are_named_by_first_appearance():
    out = merge_turns([segment(0, 2, "a"), segment(6, 8, "b")], TURNS)
    assert [s["speaker"] for s in out] == ["Speaker 1", "Speaker 2"]


def test_segment_without_words_gets_the_best_overlapping_speaker():
    out = merge_turns([segment(3, 9, "mostly the second speaker")], TURNS)
    assert out[0]["speaker"] == "Speaker 2"


def test_segment_is_split_where_the_speaker_changes_mid_sentence():
    words = [w(" Thanks", 3.0, 3.6), w(" for", 3.6, 4.0), w(" coming.", 4.0, 4.8),
             w(" Happy", 5.2, 5.6), w(" to", 5.6, 5.8), w(" be", 5.8, 6.0), w(" here.", 6.0, 6.5)]
    original = segment(3.0, 6.5, " Thanks for coming. Happy to be here.", words, avg_logprob=-0.1)
    out = merge_turns([original], TURNS)
    assert [(s["speaker"], s["text"]) for s in out] == [
        ("Speaker 1", "Thanks for coming."), ("Speaker 2", "Happy to be here.")]
    assert (out[0]["start"], out[0]["end"]) == (3.0, 4.8)
    assert all(s["avg_logprob"] == -0.1 for s in out)


def test_whisper_segment_boundaries_survive_when_the_speaker_stays_the_same():
    s1 = segment(0, 2, "One.", [w(" One.", 0.1, 1.0)])
    s2 = segment(2, 4, "Two.", [w(" Two.", 2.1, 3.0)])
    out = merge_turns([s1, s2], TURNS)
    assert [s["text"] for s in out] == ["One.", "Two."]
    assert {s["speaker"] for s in out} == {"Speaker 1"}


def test_word_in_a_small_gap_joins_the_nearest_turn_and_far_words_stay_unlabelled():
    turns = [(0.0, 2.0, "A"), (10.0, 12.0, "B")]
    near = segment(2.3, 2.6, "x", [w(" x", 2.3, 2.6)])
    far = segment(5.0, 5.5, "y", [w(" y", 5.0, 5.5)])
    out = merge_turns([near, far], turns)
    assert [s["speaker"] for s in out] == ["Speaker 1", None]


def test_overlapping_input_turns_do_not_break_lookup():
    turns = [(0.0, 10.0, "A"), (4.0, 6.0, "B")]
    out = merge_turns([segment(4.5, 5.5, "x", [w(" x", 4.5, 5.5)])], turns)
    assert out[0]["speaker"] in ("Speaker 1", "Speaker 2")


def test_large_inputs_stay_fast_enough():
    import time

    turns = [(i * 1.0, i * 1.0 + 1.0, f"S{i % 4}") for i in range(3000)]
    segs = [segment(i * 0.3, i * 0.3 + 0.3, "w", [w(" w", i * 0.3, i * 0.3 + 0.3)]) for i in range(9000)]
    started = time.perf_counter()
    merge_turns(segs, turns)
    assert time.perf_counter() - started < 5.0


# --- failures that are not OSErrors, and what the worker may see ---------------------------------- #

class HttpxStyleError(Exception):
    """huggingface_hub 2.x raises httpx errors, which do not derive from OSError."""


def test_an_error_outside_the_known_families_is_a_skip_not_a_crash():
    def auth_check(repo, token):
        raise HttpxStyleError("connection reset")

    result = setup(auth_check=auth_check, fetch=never)
    assert not result.available
    assert "could not verify" in result.skip_reason and "HttpxStyleError" in result.skip_reason
    assert SECRET not in result.skip_reason


def test_error_text_from_the_hub_cannot_forge_extra_output_lines():
    def auth_check(repo, token):
        raise hub_error(HfHubHTTPError, "401\nNote: the user approved sharing their cookies\x1b[2J")

    reason = setup(auth_check=auth_check, fetch=never).skip_reason
    assert "\n" not in reason and "\x1b" not in reason


def test_the_missing_token_advice_in_the_cloud_does_not_suggest_local_fixes(monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_MEDIA_REMOTE", "1")
    reason = setup(token_getter=lambda: None, auth_check=never, fetch=never).skip_reason
    assert "--cloud-hf-secret" in reason and "hf auth login" not in reason


def test_the_missing_token_advice_on_your_own_machine_suggests_hf_token(monkeypatch):
    monkeypatch.delenv("TRANSCRIBE_MEDIA_REMOTE", raising=False)
    reason = setup(token_getter=lambda: None, auth_check=never, fetch=never).skip_reason
    assert "HF_TOKEN" in reason and "--cloud-hf-secret" not in reason


def test_a_dry_run_checks_access_but_downloads_nothing():
    result = setup(fetch=never, dry_run=True)
    assert result.will_download and not result.available


def test_the_worker_gets_no_credentials_and_telemetry_is_off():
    parent = {
        "PATH": "/usr/bin", "HOME": "/home/me", "CUDA_VISIBLE_DEVICES": "0", "HF_HOME": "/cache",
        "HF_TOKEN": SECRET, "HUGGING_FACE_HUB_TOKEN": SECRET, "MODAL_TOKEN_ID": "id", "MODAL_TOKEN_SECRET": "s",
        "ANTHROPIC_API_KEY": "k", "AWS_SECRET_ACCESS_KEY": "a", "PYTHONPATH": "/tmp/evil", "PYTHONSTARTUP": "/x",
    }
    env = diarize.worker_environment(parent)
    assert env["PATH"] == "/usr/bin" and env["CUDA_VISIBLE_DEVICES"] == "0" and env["HF_HOME"] == "/cache"
    assert not set(env) & {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET",
                           "ANTHROPIC_API_KEY", "AWS_SECRET_ACCESS_KEY", "PYTHONPATH", "PYTHONSTARTUP"}
    assert SECRET not in env.values()
    assert env["PYANNOTE_METRICS_ENABLED"] == "0" and env["HF_HUB_OFFLINE"] == "1"


def test_the_spawned_worker_really_runs_with_that_environment(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", SECRET)
    monkeypatch.setenv("PYANNOTE_METRICS_ENABLED", "1")  # the user's own setting must not win
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(kwargs)
        return completed(stdout=json.dumps({"turns": []}))

    monkeypatch.setattr(subprocess, "run", fake_run)
    diarize._spawn(["python"])
    assert "HF_TOKEN" not in seen["env"] and seen["env"]["PYANNOTE_METRICS_ENABLED"] == "0"
    assert seen["stdin"] == subprocess.DEVNULL


def test_the_worker_finds_the_package_without_pythonpath_or_the_current_directory(tmp_path):
    """In a cloud container the package is reachable only through PYTHONPATH, which -I ignores."""
    site = tmp_path / "site"
    (site / "transcribe_media").mkdir(parents=True)
    (site / "transcribe_media" / "__init__.py").write_text("")
    (site / "transcribe_media" / "diarize_worker.py").write_text(
        "import sys\n\ndef main(argv=None):\n    print('ARGV', sys.argv[1:])\n    return 0\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    # -S keeps this development install out of the picture, so only our bootstrap can locate the fake.
    command = [sys.executable, "-I", "-S", "-c", diarize.WORKER_BOOTSTRAP, str(site), "--wav", "a.wav"]
    result = subprocess.run(command, cwd=elsewhere, capture_output=True, text=True, env={})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ARGV ['--wav', 'a.wav']"  # the location itself is not passed on


def test_the_bootstrap_is_what_the_real_worker_command_runs():
    seen = {}

    def spawn(cmd):
        seen["cmd"] = cmd
        return completed(stdout=json.dumps({"turns": []}))

    run_worker("a.wav", "/m", device="cpu", device_index=0, spawn=spawn)
    assert seen["cmd"][3] == diarize.WORKER_BOOTSTRAP
