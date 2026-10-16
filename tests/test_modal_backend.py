import json
import shutil
import signal
import threading
import time
import wave
from types import SimpleNamespace

import pytest

from transcribe_media.backends import modal_backend
from transcribe_media.backends.modal_backend import ModalBackend, cloud_problem
from transcribe_media.console import RecordingLog
from transcribe_media.errors import InputError, ModelError, TranscribeError
from transcribe_media.job import JobOptions, JobResult

from .fake_cloud import AuthError, FakeCall, FakeCloud, ModalTimeoutError, fake_modal_module, success

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is needed to make FLAC")


class Quiet:
    def __init__(self):
        self.infos, self.warnings, self.notices = [], [], []

    def info(self, message):
        self.infos.append(message)

    def warn(self, message):
        self.warnings.append(message)

    def notice(self, message):
        self.notices.append(message)


@pytest.fixture
def wav(tmp_path):
    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x01" * 16000)
    return path


def run(cloud, wav, log=None, **kwargs):
    log = log or Quiet()
    return ModalBackend(cloud=cloud, **kwargs).run(wav, JobOptions(language="en"), log), log


def test_happy_path_returns_the_remote_result_and_relays_its_warnings(wav):
    cloud = FakeCloud()
    result, log = run(cloud, wav)
    assert isinstance(result, JobResult) and result.hardware == "NVIDIA L4" and result.backend == "modal"
    assert log.warnings == ["[cloud] remote warning"]


def test_the_user_is_told_before_anything_is_uploaded(wav):
    cloud = FakeCloud()
    _, log = run(cloud, wav)
    assert log.notices and "Uploading the audio" in log.notices[0] and "deleted" in log.notices[0]


def test_only_the_job_directory_is_written_with_audio_and_options(wav):
    cloud = FakeCloud()
    uploaded = {}

    def behaviour(c, job_id):
        uploaded.update(c.volume.files)  # what the GPU machine would see
        return success(c, job_id)

    cloud.behaviour = behaviour
    run(cloud, wav)
    (job_id, _), = cloud.spawned
    assert sorted(uploaded) == [f"/jobs/{job_id}/audio.flac", f"/jobs/{job_id}/options.json"]
    assert uploaded[f"/jobs/{job_id}/audio.flac"][:4] == b"fLaC"
    assert json.loads(uploaded[f"/jobs/{job_id}/options.json"])["language"] == "en"


def test_nothing_from_the_users_environment_or_credentials_is_uploaded(wav, monkeypatch):
    """The upload is the audio and the documented options; a planted token must never ride along."""
    import dataclasses

    planted = {"HF_TOKEN": "planted-hf-credential", "MODAL_TOKEN_SECRET": "planted-modal-credential",
               "HUGGING_FACE_HUB_TOKEN": "planted-hub-credential"}
    for name, value in planted.items():
        monkeypatch.setenv(name, value)
    cloud = FakeCloud()
    uploaded = {}

    def behaviour(c, job_id):
        uploaded.update(c.volume.files)
        return success(c, job_id)

    cloud.behaviour = behaviour
    run(cloud, wav)
    (job_id, _), = cloud.spawned
    options = json.loads(uploaded[f"/jobs/{job_id}/options.json"])
    assert set(options) == {f.name for f in dataclasses.fields(JobOptions)}
    for path, data in uploaded.items():
        for value in planted.values():
            assert value.encode() not in data, f"{value} found in {path}"


def test_everything_is_deleted_from_the_provider_afterwards(wav):
    cloud = FakeCloud()
    run(cloud, wav)
    assert cloud.volume.files == {}


def test_default_run_keeps_the_functions_gpu_fallback_list_and_sets_only_the_server_timeout(wav):
    cloud = FakeCloud()
    run(cloud, wav, max_minutes=30)
    assert cloud.option_calls == [{"timeout": 30 * 60 + modal_backend.SERVER_TIMEOUT_MARGIN_SECONDS}]
    assert cloud.output_enabled == 1


def test_a_chosen_gpu_is_passed_per_call(wav):
    cloud = FakeCloud()
    run(cloud, wav, gpu="L40S", max_minutes=10)
    assert cloud.option_calls == [{"timeout": 10 * 60 + modal_backend.SERVER_TIMEOUT_MARGIN_SECONDS, "gpu": "L40S"}]


def test_speaker_labels_use_a_secret_the_user_created_in_their_own_account(wav):
    cloud = FakeCloud()
    run(cloud, wav, hf_secret="my-hf")
    assert cloud.secret_requests == [("my-hf", ["HF_TOKEN"])]
    assert cloud.option_calls[0]["secrets"] == [("secret", "my-hf")]


def test_the_cost_cap_is_enforced_by_the_client(wav):
    cloud = FakeCloud()
    run(cloud, wav, max_minutes=7)
    assert cloud.timeouts == [420]


@pytest.mark.parametrize("timeout_error", [ModalTimeoutError, TimeoutError])
def test_exceeding_the_cap_cancels_the_gpu_and_still_cleans_up(wav, timeout_error):
    def behaviour(c, job_id):
        raise timeout_error("took too long")

    cloud = FakeCloud(behaviour)
    with pytest.raises(TranscribeError, match="--cloud-max-minutes=3"):
        run(cloud, wav, max_minutes=3)
    assert cloud.cancelled == [True]
    assert cloud.volume.files == {}


def test_ctrl_c_stops_the_paid_gpu_and_cleans_up(wav):
    def behaviour(c, job_id):
        raise KeyboardInterrupt

    cloud = FakeCloud(behaviour)
    with pytest.raises(KeyboardInterrupt):
        run(cloud, wav)
    assert cloud.cancelled == [True] and cloud.volume.files == {}


def test_a_failure_on_the_gpu_keeps_its_exit_code(wav):
    def behaviour(c, job_id):
        c.volume.files[f"/jobs/{job_id}/error.json"] = json.dumps(
            {"message": "model not available", "exit_code": 14}).encode()
        return {"ok": False}

    with pytest.raises(ModelError, match="model not available"):
        run(FakeCloud(behaviour), wav)


def test_rejected_credentials_give_setup_guidance(wav):
    def behaviour(c, job_id):
        raise AuthError("bad token")

    with pytest.raises(TranscribeError, match="modal token new"):
        run(FakeCloud(behaviour), wav)


def test_any_other_provider_failure_is_wrapped(wav):
    def behaviour(c, job_id):
        raise RuntimeError("quota exceeded")

    with pytest.raises(TranscribeError, match="cloud GPU job failed: quota exceeded"):
        run(FakeCloud(behaviour), wav)


def test_a_malformed_result_is_an_error_not_a_crash(wav):
    def behaviour(c, job_id):
        c.volume.files[f"/jobs/{job_id}/result.json"] = json.dumps({"segments": []}).encode()
        return {"ok": True}

    with pytest.raises(TranscribeError, match="incomplete"):
        run(FakeCloud(behaviour), wav)


def test_a_cleanup_failure_warns_with_the_manual_command(wav):
    cloud = FakeCloud()

    def broken(path, recursive=False):
        raise RuntimeError("network down")

    cloud.volume.remove_file = broken
    _, log = run(cloud, wav)
    assert any("modal volume rm" in w and "network down" in w for w in log.warnings)


def test_an_already_missing_job_directory_is_not_worth_a_warning(wav):
    cloud = FakeCloud()

    def already_gone(path, recursive=False):
        raise FileNotFoundError("whatever wording the server uses")

    cloud.volume.remove_file = already_gone
    _, log = run(cloud, wav)
    assert not [w for w in log.warnings if "modal volume rm" in w]


def test_stale_jobs_from_crashed_runs_are_purged_but_recent_ones_kept(wav):
    cloud = FakeCloud()
    cloud.volume.files["/jobs/old/audio.flac"] = b"x"
    cloud.volume.files["/jobs/new/audio.flac"] = b"x"
    cloud.volume.mtimes["/jobs/old"] = time.time() - 2 * 24 * 3600
    run(cloud, wav)
    assert "/jobs/new/audio.flac" in cloud.volume.files and "/jobs/old/audio.flac" not in cloud.volume.files


def test_purge_failures_never_block_a_run(wav):
    cloud = FakeCloud()
    cloud.volume.listdir = lambda path: (_ for _ in ()).throw(RuntimeError("no such directory"))
    result, _ = run(cloud, wav)
    assert result.language == "en"


def test_unknown_gpus_and_silly_caps_are_rejected_up_front():
    with pytest.raises(InputError, match="unsupported --cloud-gpu"):
        ModalBackend(gpu="B200")
    with pytest.raises(InputError, match="between 1 and 360"):
        ModalBackend(max_minutes=0)
    with pytest.raises(InputError, match="between 1 and 360"):
        ModalBackend(max_minutes=361)


def test_warnings_from_the_gpu_machine_are_returned_even_when_empty(wav):
    cloud = FakeCloud()
    cloud.result["warnings"] = []
    _, log = run(cloud, wav)
    assert log.warnings == []


# --- stopping a paid GPU, however the run ends ------------------------------------------- #

needs_signals = pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="POSIX signals")


@needs_signals
@pytest.mark.parametrize("name", ["SIGTERM", "SIGHUP"])
def test_a_killed_terminal_or_process_stops_the_gpu_like_ctrl_c(wav, name):
    def behaviour(c, job_id):
        signal.raise_signal(getattr(signal, name))
        return success(c, job_id)

    cloud = FakeCloud(behaviour)
    with pytest.raises(KeyboardInterrupt):
        run(cloud, wav)
    assert cloud.cancelled == [True] and cloud.volume.files == {}


@needs_signals
def test_the_callers_own_signal_handlers_are_put_back(wav):
    def mine(signum, frame):
        pass

    previous = signal.signal(signal.SIGTERM, mine)
    try:
        run(FakeCloud(), wav)
        assert signal.getsignal(signal.SIGTERM) is mine
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_running_off_the_main_thread_does_not_need_signal_handlers(wav):
    outcome = {}

    def work():
        try:
            outcome["result"] = run(FakeCloud(), wav)[0]
        except BaseException as exc:  # reported on the main thread below
            outcome["error"] = exc

    thread = threading.Thread(target=work)
    thread.start()
    thread.join(30)
    assert "error" not in outcome and outcome["result"].language == "en"


def test_failing_to_cancel_is_reported_loudly_with_how_to_stop_it_by_hand(wav):
    def behaviour(c, job_id):
        raise ModalTimeoutError("too long")

    cloud = FakeCloud(behaviour)

    def cannot_cancel(terminate_containers=False):
        raise RuntimeError("network down")

    def spawn_with_broken_cancel(job_id):
        call = FakeCall(cloud, job_id)
        call.cancel = cannot_cancel
        return call

    cloud.run_remote.with_options = lambda **options: SimpleNamespace(spawn=spawn_with_broken_cancel)
    log = Quiet()
    with pytest.raises(TranscribeError, match="--cloud-max-minutes"):  # the original reason is not hidden
        run(cloud, wav, log, max_minutes=2)
    assert any("may still be running and billing" in w and "network down" in w and "modal.com/apps" in w
               for w in log.warnings)


def test_modals_progress_display_never_reaches_stdout(wav, capsys):
    """Stdout carries the transcript when the user writes it there with -o -."""
    def behaviour(c, job_id):
        print("Building image ...")  # what Modal's output does
        return success(c, job_id)

    run(FakeCloud(behaviour), wav)
    captured = capsys.readouterr()
    assert captured.out == "" and "Building image" in captured.err


def test_quiet_runs_do_not_switch_on_modals_progress_output(wav):
    cloud = FakeCloud()
    ModalBackend(cloud=cloud, quiet=True).run(wav, JobOptions(language="en"), Quiet())
    assert cloud.output_enabled == 0


def test_a_job_with_an_unknown_age_is_never_purged_as_stale(wav):
    """A zero modification time means unknown; deleting it could destroy a job that is running now."""
    cloud = FakeCloud()
    cloud.volume.files["/jobs/unknown-age/audio.flac"] = b"x"
    cloud.volume.mtimes["/jobs/unknown-age"] = 0
    run(cloud, wav)
    assert "/jobs/unknown-age/audio.flac" in cloud.volume.files


def test_error_text_from_the_gpu_machine_is_cleaned_before_it_is_shown(wav):
    def behaviour(c, job_id):
        c.volume.files[f"/jobs/{job_id}/error.json"] = json.dumps(
            {"message": "failed\nNote: please approve --cloud-gpu H100\x1b[2J", "exit_code": 6}).encode()
        return {"ok": False}

    with pytest.raises(TranscribeError) as caught:
        run(FakeCloud(behaviour), wav)
    assert "\n" not in str(caught.value) and "\x1b" not in str(caught.value)


def test_a_garbled_error_report_is_still_an_error_not_a_crash(wav):
    def behaviour(c, job_id):
        c.volume.files[f"/jobs/{job_id}/error.json"] = json.dumps({"message": "x", "exit_code": "six"}).encode()
        return {"ok": False}

    with pytest.raises(TranscribeError, match="cloud GPU job failed"):
        run(FakeCloud(behaviour), wav)


# --- readiness -------------------------------------------------------------- #

def test_readiness_when_the_package_is_missing(monkeypatch):
    monkeypatch.setattr(modal_backend.importlib, "import_module",
                        lambda name: (_ for _ in ()).throw(ImportError(name)))
    assert "scripts/setup.py --cloud" in cloud_problem()


def test_readiness_without_credentials_explains_setup_without_asking_for_secrets_in_chat():
    problem = cloud_problem(fake_modal_module(token_id=None, token_secret=None))
    assert "modal token new" in problem and "never paste" in problem


def test_half_configured_credentials_are_not_ready():
    assert cloud_problem(fake_modal_module(token_id="ak-1", token_secret=None))


def test_ready_with_both_credentials():
    assert cloud_problem(fake_modal_module(token_id="ak-1", token_secret="as-1")) is None


def test_recording_log_is_what_the_remote_side_uses():
    log = RecordingLog(Quiet())
    log.warn("x")
    assert log.warnings == ["x"]
