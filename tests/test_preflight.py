import pytest

from transcribe_media import diarize, preflight
from transcribe_media.environment import Environment
from transcribe_media.preflight import FAIL, OK, WARN, Check
from transcribe_media.profiles import AsrProfile

from .conftest import make_gpu, make_hw

GPU_PROFILE = AsrProfile("large-v3", "cuda", "float16", notes=())
CPU_PROFILE = AsrProfile("large-v3", "cpu", "int8", notes=("Running on the CPU: slow.",))


def by_name(checks):
    return {c.name: c for c in checks}


def test_python_version_check():
    assert preflight.check_python().status == OK


@pytest.mark.parametrize("text, expected", [("4.8.2", (4, 8, 2)), ("4.6", (4, 6)), ("4.6.3.post1", (4, 6, 3)), ("x", (0,))])
def test_version_parsing(text, expected):
    assert preflight._as_tuple(text) == expected


def test_hardware_report_names_the_gpu_architecture_and_chosen_model():
    checks = by_name(preflight.check_hardware(make_hw(gpus=[make_gpu()]), GPU_PROFILE))
    assert "RTX 5080" in checks["Hardware"].detail and "sm_120" in checks["Hardware"].detail
    assert checks["Selected model"].detail == "large-v3 / float16 on cuda"


def test_profile_notes_are_surfaced_as_warnings():
    checks = preflight.check_hardware(make_hw(), CPU_PROFILE)
    assert [c.status for c in checks if c.name == "Note"] == [WARN]
    assert "no NVIDIA GPU detected" in checks[0].detail


def test_missing_faster_whisper_is_a_failure(monkeypatch):
    monkeypatch.setattr(preflight, "_version", lambda package: None)
    (check,) = preflight.check_asr_packages(make_hw())
    assert check.status == FAIL and "setup.py" in check.remedy


def test_blackwell_with_an_old_ctranslate2_is_a_failure(monkeypatch):
    versions = {"faster-whisper": "1.2.1", "ctranslate2": "4.5.0"}
    monkeypatch.setattr(preflight, "_version", versions.get)
    checks = by_name(preflight.check_asr_packages(make_hw(gpus=[make_gpu()])))
    assert checks["CTranslate2 for Blackwell"].status == FAIL
    versions["ctranslate2"] = "4.8.2"
    assert "CTranslate2 for Blackwell" not in by_name(preflight.check_asr_packages(make_hw(gpus=[make_gpu()])))


def test_older_ctranslate2_is_fine_on_older_gpus(monkeypatch):
    monkeypatch.setattr(preflight, "_version", {"faster-whisper": "1.2.1", "ctranslate2": "4.5.0"}.get)
    old_gpu = make_gpu("RTX 3070 Ti", 8192, (8, 6))
    assert len(preflight.check_asr_packages(make_hw(gpus=[old_gpu]))) == 1


def test_cuda_check_is_skipped_without_a_gpu():
    assert preflight.check_cuda(make_hw(), lambda: 0) is None


def test_cuda_check_reports_load_failures_and_missing_devices(monkeypatch):
    hw = make_hw(gpus=[make_gpu()])
    monkeypatch.setattr(preflight, "configure_cuda_environment", lambda: ["libcublas.so.12: not found"])
    broken = preflight.check_cuda(hw, lambda: 0)
    assert broken.status == FAIL and "1 library load error" in broken.detail
    monkeypatch.setattr(preflight, "configure_cuda_environment", lambda: [])
    assert preflight.check_cuda(hw, lambda: 1).status == OK
    assert preflight.check_cuda(hw, lambda: 0).status == FAIL


def test_model_cache_states(monkeypatch):
    monkeypatch.setattr(preflight.models, "cached_path", lambda spec: "/cache")
    assert preflight.check_model_cache(GPU_PROFILE).detail == "large-v3 is cached"
    monkeypatch.setattr(preflight.models, "cached_path", lambda spec: None)
    missing = preflight.check_model_cache(GPU_PROFILE)
    assert missing.status == WARN and "3.1 GB" in missing.detail
    custom = preflight.check_model_cache(AsrProfile("/my/model", "cpu", "int8"))
    assert custom.status == OK and "custom model" in custom.detail


def test_ytdlp_and_javascript_runtime(monkeypatch):
    monkeypatch.setattr(preflight, "_version", lambda package: "2026.8.19")
    monkeypatch.setattr(preflight.shutil, "which", lambda tool: "/usr/bin/node" if tool == "node" else None)
    checks = by_name(preflight.check_ytdlp())
    assert checks["JavaScript runtime"].detail.startswith("node")
    monkeypatch.setattr(preflight.shutil, "which", lambda tool: None)
    assert by_name(preflight.check_ytdlp())["JavaScript runtime"].status == WARN
    monkeypatch.setattr(preflight, "_version", lambda package: None)
    assert preflight.check_ytdlp()[0].status == FAIL


@pytest.mark.parametrize("setup, status, fragment", [
    (diarize.DiarizationSetup(pipeline_dir="/m"), OK, "model cached"),
    (diarize.DiarizationSetup(will_download=True), OK, "downloaded on first use"),
    (diarize.DiarizationSetup(skip_reason="no Hugging Face token found"), WARN, "will be skipped: no Hugging Face"),
])
def test_speaker_label_readiness(monkeypatch, setup, status, fragment):
    seen = {}
    monkeypatch.setattr(preflight.diarize, "prepare_diarization", lambda **kw: seen.update(kw) or setup)
    check = preflight.check_diarization(offline=False)
    assert (check.status, fragment in check.detail) == (status, True)
    assert seen["dry_run"] is True  # the doctor must never download anything


def test_environment_check_explains_ephemeral_sessions():
    assert preflight.check_environment(Environment()) is None
    sandbox = preflight.check_environment(Environment("a Claude Code cloud session", True))
    assert "rebuilt each session" in sandbox.detail
    persistent = preflight.check_environment(Environment("GitHub Codespaces", False))
    assert "rebuilt" not in persistent.detail


def test_cloud_check_when_ready():
    check = preflight.check_cloud(make_hw(), None)
    assert check.status == OK and "uploads the audio" in check.detail


def test_cloud_check_only_warns_when_there_is_no_local_gpu_to_fall_back_on():
    problem = "no Modal credentials"
    assert preflight.check_cloud(make_hw(), problem).status == WARN
    assert preflight.check_cloud(make_hw(gpus=[make_gpu()]), problem).status == OK


def test_full_run_assembles_every_section(monkeypatch):
    monkeypatch.setattr(preflight, "_version", lambda package: "4.8.2" if package == "ctranslate2" else "1.2.1")
    monkeypatch.setattr(preflight, "configure_cuda_environment", lambda: [])
    monkeypatch.setattr(preflight.models, "cached_path", lambda spec: "/cache")
    monkeypatch.setattr(preflight.diarize, "prepare_diarization",
                        lambda **kw: diarize.DiarizationSetup(skip_reason="no token"))
    names = [c.name for c in preflight.run_checks(
        make_hw(gpus=[make_gpu()]), GPU_PROFILE, offline=False, device_count=lambda: 1,
        environment=Environment("Google Colab", True), cloud_problem=None)]
    for expected in ("Platform", "Python", "FFmpeg", "Environment", "Hardware", "Selected model", "faster-whisper",
                     "CUDA runtime", "Model files", "yt-dlp", "Speaker labels", "Cloud GPU"):
        assert expected in names, expected


def test_report_formatting_and_failure_detection():
    checks = [Check("A", OK, "fine", "ignored because ok"), Check("B", WARN, "meh", "try this"),
              Check("C", FAIL, "broken", "fix this")]
    report = preflight.format_report(checks)
    assert "[OK  ] A" in report and "[WARN] B" in report and "[FAIL] C" in report
    assert "-> try this" in report and "-> fix this" in report and "ignored because ok" not in report
    assert "problems found" in report and preflight.has_failures(checks)
    healthy = preflight.format_report(checks[:2])
    assert "ready to transcribe" in healthy and not preflight.has_failures(checks[:2])
