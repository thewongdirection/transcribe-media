from pathlib import Path

import pytest

from transcribe_media import audio, cli, diarize, environment, job
from transcribe_media.environment import Environment
from transcribe_media.hardware import GpuInfo, Hardware
from transcribe_media.profiles import AsrProfile
from transcribe_media.transcribe import TranscriptionResult

PROFILE = AsrProfile("large-v3", "cpu", "int8")
NOT_READY = "no Modal credentials: run `modal token new`"


# Everything about the machine and the user's accounts that could change what a test sees.
_HOST_SETTINGS = (
    "TRANSCRIBE_MEDIA_BACKEND", "TRANSCRIBE_MEDIA_REMOTE", "TRANSCRIBE_MEDIA_DEBUG",
    "TRANSCRIBE_MEDIA_NO_AUTOSETUP", "TRANSCRIBE_MEDIA_VENV",
    "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_PROFILE",
    "CUDA_VISIBLE_DEVICES", "YTDLP_NO_PLUGINS",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    *(signal[0] for signal in environment._SIGNALS),
)


@pytest.fixture(autouse=True)
def hermetic_environment(monkeypatch, tmp_path_factory):
    """Run every test as if on a bare machine: no accounts, no cloud session, no proxy, an empty home.

    Tests that need a setting set it themselves, so results never depend on where the suite runs.
    """
    for name in _HOST_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("MODAL_CONFIG_PATH", str(home / "no-modal-config.toml"))


@pytest.fixture
def ffmpeg_not_needed(monkeypatch):
    """For tests that stop before any media is decoded, so they also pass on a machine without FFmpeg."""
    monkeypatch.setattr(audio, "require_ffmpeg", lambda: None)


def make_gpu(name="NVIDIA GeForce RTX 5080", vram_mb=16303, cc=(12, 0), index=0):
    return GpuInfo(index=index, name=name, vram_mb=vram_mb, compute_capability=cc, driver="576.80")


def make_hw(gpus=(), ram_gb=64.0, system="Linux", machine="x86_64", free_disk_gb=500.0):
    return Hardware(
        system=system,
        machine=machine,
        cpu_threads=16,
        ram_gb=ram_gb,
        free_disk_gb=free_disk_gb,
        gpus=tuple(gpus),
    )


@pytest.fixture
def rtx5080_box():
    """The default target: RTX 5080 (16 GB, Blackwell sm_120), 64 GB RAM."""
    return make_hw(gpus=[make_gpu()])


# --- the whole pipeline with a fake model: real ffmpeg decoding, file IO and orchestration ---- #

class FakeTranscriber:
    def __init__(self, segments=None, language="en", probability=0.99, duration=3.0):
        self.segments = segments if segments is not None else [
            {"start": 0.0, "end": 2.0, "text": "Hello from the test.", "speaker": None,
             "avg_logprob": -0.1, "no_speech_prob": 0.01, "compression_ratio": 1.1}]
        self.result = TranscriptionResult(self.segments, language, probability, duration,
                                          "large-v3", "cpu", "int8")
        self.closed = False
        self.kwargs = None

    def transcribe(self, wav, **kwargs):
        self.kwargs = kwargs
        assert Path(wav).is_file()
        return self.result

    def close(self):
        self.closed = True


@pytest.fixture
def pipeline(monkeypatch):
    """Stub the model and the machine; the real ffmpeg decode, file IO and orchestration stay."""
    fake = FakeTranscriber()
    monkeypatch.setattr(job, "choose_profile", lambda options, log: (PROFILE, "CPU"))
    monkeypatch.setattr(job, "load_transcriber", lambda options, profile: fake)
    monkeypatch.setattr(job.diarize, "prepare_diarization",
                        lambda **kw: diarize.DiarizationSetup(skip_reason="no Hugging Face token found"))
    monkeypatch.setattr(job, "local_gpu_available", lambda options: False)
    monkeypatch.setattr(cli, "cloud_problem", lambda: NOT_READY)
    monkeypatch.setattr(cli, "detect_environment", lambda: Environment())  # never depend on the host
    return fake


@pytest.fixture
def tone(tmp_path):
    path = tmp_path / "talk.wav"
    cli.write_test_tone(path, seconds=1.0)
    return path
