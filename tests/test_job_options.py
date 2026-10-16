"""Job options are untrusted on the receiving side; run_job must degrade, never lose a finished transcript."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from transcribe_media import job
from transcribe_media.errors import DiarizeError, InputError, TranscribeError
from transcribe_media.job import JobOptions
from transcribe_media.profiles import AsrProfile
from transcribe_media.transcribe import TranscriptionResult

from .conftest import make_gpu, make_hw

CUDA = AsrProfile("large-v3", "cuda", "float16", device_index=1)
CPU = AsrProfile("large-v3", "cpu", "int8")
SEGMENTS = [{"start": 0.0, "end": 2.0, "text": "hello", "speaker": None}]


class Log:
    def __init__(self):
        self.infos, self.warnings = [], []

    def info(self, message):
        self.infos.append(message)

    notice = info

    def warn(self, message):
        self.warnings.append(message)


def finished(device="cpu"):
    return TranscriptionResult([dict(s) for s in SEGMENTS], "en", 0.99, 2.0, "large-v3", device, "int8")


# --- option validation ---------------------------------------------------------- #

@pytest.mark.parametrize("bad", [
    {"language": "english"}, {"language": "EN"}, {"language": "e"}, {"language": 5},
    {"prompt": "x" * (job.MAX_TEXT + 1)}, {"prompt": 5},
    {"beam_size": 0}, {"beam_size": job.MAX_BEAM_SIZE + 1}, {"beam_size": True}, {"beam_size": "5"},
    {"model": ""}, {"model": "m" * 301}, {"model": 3},
    {"device": "tpu"}, {"compute_type": "int4"}, {"diarize": "maybe"},
    {"vad": "yes"}, {"offline": 1}, {"no_download": None}, {"condition_on_previous_text": "no"},
    {"num_speakers": 0}, {"num_speakers": job.MAX_SPEAKERS + 1}, {"min_speakers": True}, {"max_speakers": "2"},
    {"min_speakers": 5, "max_speakers": 2},
])
def test_unsuitable_option_values_are_rejected_with_a_value_error(bad):
    with pytest.raises(ValueError):
        JobOptions(**bad)


def test_the_extremes_that_are_allowed_still_work():
    options = JobOptions(language="zh", beam_size=job.MAX_BEAM_SIZE, model="large-v3", prompt="x" * job.MAX_TEXT,
                         num_speakers=job.MAX_SPEAKERS, min_speakers=2, max_speakers=2)
    assert JobOptions.from_dict(options.to_dict()) == options


def test_options_arriving_as_json_are_validated_too():
    """The cloud side builds options from a file it did not write."""
    with pytest.raises(ValueError):
        JobOptions.from_dict({"beam_size": "5"})
    with pytest.raises(ValueError, match="unknown job options"):
        JobOptions.from_dict({"beam_size": 5, "cuda_visible_devices": "0"})


# --- how many GPUs CTranslate2 can really use --------------------------------------- #

def fake_ctranslate2(monkeypatch, count=None, error=None):
    def get_cuda_device_count():
        if error:
            raise error
        return count

    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(get_cuda_device_count=get_cuda_device_count))


def test_the_cuda_device_count_comes_from_ctranslate2(monkeypatch):
    fake_ctranslate2(monkeypatch, count=2)
    assert job.cuda_device_count() == 2


@pytest.mark.parametrize("failure", [RuntimeError("driver/library version mismatch"), OSError("libcuda.so missing")])
def test_a_driver_problem_means_no_usable_cuda_rather_than_a_crash(monkeypatch, failure):
    fake_ctranslate2(monkeypatch, error=failure)
    assert job.cuda_device_count() == 0


def test_a_missing_ctranslate2_means_no_usable_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "ctranslate2", None)  # makes the import fail
    assert job.cuda_device_count() == 0


# --- is a local GPU available? -------------------------------------------------------- #

def machine(monkeypatch, hw, cuda_devices=1):
    monkeypatch.setattr(job, "detect_hardware", lambda cache: hw)
    monkeypatch.setattr(job, "configure_cuda_environment", lambda: [])
    monkeypatch.setattr(job, "cuda_device_count", lambda: cuda_devices)


def test_a_cpu_request_never_probes_the_hardware(monkeypatch):
    monkeypatch.setattr(job, "detect_hardware", lambda cache: pytest.fail("not needed for --device cpu"))
    assert job.local_gpu_available(JobOptions(device="cpu")) is False


def test_a_working_gpu_counts_and_a_broken_driver_does_not(monkeypatch):
    machine(monkeypatch, make_hw(gpus=[make_gpu()]), cuda_devices=1)
    assert job.local_gpu_available(JobOptions()) is True
    machine(monkeypatch, make_hw(gpus=[make_gpu()]), cuda_devices=0)
    assert job.local_gpu_available(JobOptions()) is False


def test_a_machine_without_a_gpu_has_none(monkeypatch):
    machine(monkeypatch, make_hw())
    assert job.local_gpu_available(JobOptions()) is False


@pytest.mark.parametrize("options, message", [
    (JobOptions(device="cuda"), "no NVIDIA GPU"),
    (JobOptions(compute_type="int8"), "INT8"),
])
def test_an_impossible_request_is_reported_instead_of_being_mistaken_for_no_gpu(monkeypatch, options, message):
    """If this returned False the audio could be sent to the cloud because of a typo."""
    machine(monkeypatch, make_hw(gpus=[make_gpu()]) if options.compute_type == "int8" else make_hw())
    with pytest.raises(InputError, match=message):
        job.local_gpu_available(options)


# --- choosing the profile -------------------------------------------------------------- #

def test_a_small_gpu_gets_its_downgrade_explained(monkeypatch):
    machine(monkeypatch, make_hw(gpus=[make_gpu(name="NVIDIA GeForce RTX 5050", vram_mb=4096)]))
    log = Log()
    profile, where = job.choose_profile(JobOptions(), log)
    assert profile.model == "medium" and where == "NVIDIA GeForce RTX 5050"
    assert any("less" in w and "accurate" in w for w in log.warnings)
    assert log.infos == ["Using medium (float16) on NVIDIA GeForce RTX 5050"]


# --- the GPU run fails after the profile was chosen ---------------------------------------- #

def stub_stages(monkeypatch, transcribe):
    asked = []

    def choose(options, log):
        asked.append(options.device)
        return (CUDA, "RTX 5080") if options.device != "cpu" else (CPU, "CPU")

    monkeypatch.setattr(job, "choose_profile", choose)
    monkeypatch.setattr(job, "_transcribe", transcribe)
    monkeypatch.setattr(job, "apply_diarization", lambda options, wav, segs, profile, log, **kw: (segs, {"status": "off"}))
    return asked


def test_a_gpu_run_that_fails_is_retried_on_the_cpu_with_the_same_accuracy(monkeypatch):
    attempts = []

    def transcribe(wav, options, profile, log):
        attempts.append(profile.device)
        if profile.device == "cuda":
            raise TranscribeError("Library libcublas.so.12 is not found")
        return finished("cpu")

    asked = stub_stages(monkeypatch, transcribe)
    log = Log()
    result = job.run_job(Path("a.wav"), JobOptions(), log)
    assert attempts == ["cuda", "cpu"] and asked == ["auto", "cpu"]
    assert (result.device, result.hardware) == ("cpu", "CPU")
    assert any("retrying on the CPU" in w and "libcublas" in w for w in log.warnings)


def test_an_explicit_cuda_request_is_not_silently_moved_to_the_cpu(monkeypatch):
    def transcribe(wav, options, profile, log):
        raise TranscribeError("CUDA out of memory")

    asked = stub_stages(monkeypatch, transcribe)
    with pytest.raises(TranscribeError, match="out of memory"):
        job.run_job(Path("a.wav"), JobOptions(device="cuda"), Log())
    assert asked == ["cuda"]


def test_a_failure_on_the_cpu_is_final(monkeypatch):
    attempts = []

    def transcribe(wav, options, profile, log):
        attempts.append(profile.device)
        raise TranscribeError("model is corrupt")

    stub_stages(monkeypatch, transcribe)
    with pytest.raises(TranscribeError, match="corrupt"):
        job.run_job(Path("a.wav"), JobOptions(device="cpu"), Log())
    assert attempts == ["cpu"]  # exactly one attempt, never a loop


def test_the_cpu_retry_failing_too_surfaces_that_error(monkeypatch):
    def transcribe(wav, options, profile, log):
        raise TranscribeError(f"failed on {profile.device}")

    stub_stages(monkeypatch, transcribe)
    with pytest.raises(TranscribeError, match="failed on cpu"):
        job.run_job(Path("a.wav"), JobOptions(), Log())


# --- speaker labels must never cost the transcript ------------------------------------------- #

def stub_prepare(monkeypatch, outcome):
    def prepare(**kwargs):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(job.diarize, "prepare_diarization", prepare)


@pytest.mark.parametrize("failure", [
    RuntimeError("httpx.ConnectError: name resolution failed"), ValueError("bad pipeline"), KeyError("x"),
])
def test_any_unexpected_failure_while_labelling_is_a_skip(monkeypatch, failure):
    stub_prepare(monkeypatch, failure)
    log = Log()
    segments, status = job.apply_diarization(JobOptions(), Path("a.wav"), SEGMENTS, CPU, log)
    assert segments is SEGMENTS
    assert status["status"] == "skipped" and type(failure).__name__ in status["reason"]
    assert any("speaker labels skipped" in w for w in log.warnings)


def test_an_unexpected_failure_while_labelling_is_an_error_only_when_labels_are_required(monkeypatch):
    stub_prepare(monkeypatch, RuntimeError("boom"))
    with pytest.raises(DiarizeError, match="RuntimeError"):
        job.apply_diarization(JobOptions(diarize="on"), Path("a.wav"), SEGMENTS, CPU, Log())


def test_a_failure_in_the_merge_step_is_a_skip_as_well(monkeypatch):
    stub_prepare(monkeypatch, job.diarize.DiarizationSetup(pipeline_dir="/m"))
    monkeypatch.setattr(job.diarize, "run_worker", lambda *a, **k: ([(0.0, 1.0, "A")], ""))
    monkeypatch.setattr(job.diarize, "merge_turns", lambda *a: (_ for _ in ()).throw(TypeError("odd segment")))
    _, status = job.apply_diarization(JobOptions(), Path("a.wav"), SEGMENTS, CPU, Log())
    assert status["status"] == "skipped" and "TypeError" in status["reason"]


def test_there_is_nothing_to_label_when_no_speech_was_found(monkeypatch):
    stub_prepare(monkeypatch, AssertionError("must not even look for the model"))
    segments, status = job.apply_diarization(JobOptions(), Path("a.wav"), [], CPU, Log())
    assert segments == [] and status["status"] == "skipped" and "nobody to label" in status["reason"]


def test_the_chosen_gpu_index_and_offline_choices_reach_the_diarization_step(monkeypatch):
    seen = {}

    def prepare(**kwargs):
        seen["prepare"] = kwargs
        return job.diarize.DiarizationSetup(pipeline_dir="/m")

    def worker(wav, pipeline, **kwargs):
        seen["worker"] = kwargs
        return [(0.0, 5.0, "A")], ""

    monkeypatch.setattr(job.diarize, "prepare_diarization", prepare)
    monkeypatch.setattr(job.diarize, "run_worker", worker)
    job.apply_diarization(JobOptions(offline=True, no_download=True, min_speakers=2, max_speakers=4),
                          Path("a.wav"), [dict(s) for s in SEGMENTS], CUDA, Log())
    assert seen["prepare"] == {"offline": True, "allow_download": False}
    assert (seen["worker"]["device"], seen["worker"]["device_index"]) == ("cuda", 1)
    assert (seen["worker"]["min_speakers"], seen["worker"]["max_speakers"]) == (2, 4)


# --- --diarize on must not throw away a finished transcript ------------------------------------------------ #

def test_required_speaker_labels_are_checked_before_the_slow_transcription_starts(monkeypatch):
    stub_prepare(monkeypatch, job.diarize.DiarizationSetup(skip_reason="no Hugging Face token found"))
    transcribed = []
    monkeypatch.setattr(job, "choose_profile", lambda options, log: pytest.fail("must fail before the model loads"))
    monkeypatch.setattr(job, "_transcribe", lambda *a: transcribed.append(1))
    with pytest.raises(DiarizeError, match="token"):
        job.run_job(Path("a.wav"), JobOptions(diarize="on"), Log())
    assert transcribed == []


def test_an_unexpected_failure_while_checking_required_labels_is_still_a_diarize_error(monkeypatch):
    stub_prepare(monkeypatch, RuntimeError("connection reset"))
    with pytest.raises(DiarizeError, match="RuntimeError"):
        job.require_diarization(JobOptions(diarize="on"))


@pytest.mark.parametrize("mode", ["auto", "off"])
def test_other_modes_do_not_check_early(monkeypatch, mode):
    stub_prepare(monkeypatch, AssertionError("not needed before the transcription"))
    assert job.require_diarization(JobOptions(diarize=mode)) is None


def test_the_early_check_is_reused_not_repeated(monkeypatch):
    calls = []
    setup = job.diarize.DiarizationSetup(pipeline_dir="/m")
    monkeypatch.setattr(job.diarize, "prepare_diarization", lambda **kw: calls.append(kw) or setup)
    monkeypatch.setattr(job.diarize, "run_worker", lambda *a, **k: ([(0.0, 5.0, "A")], ""))
    monkeypatch.setattr(job, "choose_profile", lambda options, log: (CPU, "CPU"))
    monkeypatch.setattr(job, "_transcribe", lambda wav, options, profile, log: finished())
    result = job.run_job(Path("a.wav"), JobOptions(diarize="on"), Log())
    assert len(calls) == 1 and result.diarization["status"] == "applied"


def test_a_worker_that_finds_nobody_does_not_claim_labels_were_applied(monkeypatch):
    stub_prepare(monkeypatch, job.diarize.DiarizationSetup(pipeline_dir="/m"))
    monkeypatch.setattr(job.diarize, "run_worker", lambda *a, **k: ([], ""))
    log = Log()
    segments, status = job.apply_diarization(JobOptions(), Path("a.wav"), [dict(s) for s in SEGMENTS], CPU, log)
    assert status == {"status": "skipped", "reason": "no speakers were detected"}
    assert segments[0]["speaker"] is None and any("no speakers" in w for w in log.warnings)


# --- a GPU-only precision cannot follow a fallback to the CPU ----------------------------------------------------- #

def test_unusable_cuda_with_a_gpu_precision_requested_falls_back_to_a_precision_the_cpu_has(monkeypatch):
    machine(monkeypatch, make_hw(gpus=[make_gpu()]), cuda_devices=0)
    profile, where = job.choose_profile(JobOptions(compute_type="float16"), Log())
    assert (profile.device, profile.compute_type, where) == ("cpu", "int8", "CPU")


def test_the_retry_after_a_failed_gpu_run_drops_the_gpu_precision_too(monkeypatch):
    asked = []

    def choose(options, log):
        asked.append((options.device, options.compute_type))
        return (CUDA, "RTX 5080") if options.device == "auto" else (CPU, "CPU")

    monkeypatch.setattr(job, "choose_profile", choose)
    monkeypatch.setattr(job, "_transcribe", lambda wav, options, profile, log: (
        finished() if profile.device == "cpu" else (_ for _ in ()).throw(TranscribeError("no kernel image"))))
    monkeypatch.setattr(job, "apply_diarization", lambda options, wav, segs, profile, log, **kw: (segs, {"status": "off"}))
    job.run_job(Path("a.wav"), JobOptions(compute_type="float16"), Log())
    assert asked == [("auto", "float16"), ("cpu", "auto")]


@pytest.mark.parametrize("compute_type", ["float16", "int8_float16"])
def test_asking_for_a_gpu_precision_on_a_machine_without_one_is_a_clear_error(monkeypatch, compute_type):
    machine(monkeypatch, make_hw())
    with pytest.raises(InputError, match="needs a GPU"):
        job.choose_profile(JobOptions(compute_type=compute_type), Log())
    with pytest.raises(InputError, match="needs a GPU"):
        job.choose_profile(JobOptions(device="cpu", compute_type=compute_type), Log())


@pytest.mark.parametrize("compute_type", ["int8", "float32", "auto"])
def test_cpu_precisions_are_fine_on_the_cpu(monkeypatch, compute_type):
    machine(monkeypatch, make_hw())
    assert job.choose_profile(JobOptions(compute_type=compute_type), Log())[0].device == "cpu"


def test_a_cloud_run_may_ask_for_a_gpu_precision_without_a_local_gpu(monkeypatch):
    """Deciding whether a local GPU exists must not fail just because the cloud one will be used."""
    machine(monkeypatch, make_hw())
    assert job.local_gpu_available(JobOptions(compute_type="float16")) is False
