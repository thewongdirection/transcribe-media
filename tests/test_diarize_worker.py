import argparse
import json
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from transcribe_media import diarize_worker as worker
from transcribe_media.errors import AudioError


class FakeTensor:
    def __init__(self, array):
        self.array = array

    def unsqueeze(self, dim):
        return FakeTensor(self.array[None, :])


FAKE_TORCH = SimpleNamespace(from_numpy=FakeTensor)


def write_wav(path, samples, channels=1, width=2, rate=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(np.asarray(samples, dtype=np.int16).tobytes())


def test_waveform_is_normalised_to_unit_range(tmp_path):
    path = tmp_path / "a.wav"
    write_wav(path, [0, 16384, -32768])
    tensor, rate = worker.load_waveform(str(path), FAKE_TORCH)
    assert rate == 16000 and tensor.array.shape == (1, 3)
    assert tensor.array.dtype == np.float32
    assert tensor.array.tolist() == [[0.0, 0.5, -1.0]]


def test_non_mono_audio_is_rejected(tmp_path):
    path = tmp_path / "stereo.wav"
    write_wav(path, [0, 0, 0, 0], channels=2)
    with pytest.raises(AudioError, match="16-bit mono"):
        worker.load_waveform(str(path), FAKE_TORCH)


def fake_cuda(available=True, capability=(12, 0), arch_list=("sm_90", "sm_120")):
    return SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: available,
        get_device_capability=lambda index: capability,
        get_arch_list=lambda: list(arch_list),
    ))


def test_cuda_supported_when_the_build_has_kernels_for_the_card():
    assert worker.cuda_supported(fake_cuda(), 0) is None


def test_cuda_unsupported_explains_blackwell_requirement():
    reason = worker.cuda_supported(fake_cuda(arch_list=("sm_86", "sm_90")), 0)
    assert "sm_120" in reason and "torch >= 2.8" in reason


def test_cuda_unavailable_is_reported():
    assert "cannot see" in worker.cuda_supported(fake_cuda(available=False), 0)


def test_annotation_is_flattened_and_sorted():
    def track(start, end):
        return SimpleNamespace(start=start, end=end)

    annotation = SimpleNamespace(itertracks=lambda yield_label: iter([
        (track(5, 6), "t2", "SPEAKER_01"), (track(0, 2), "t1", "SPEAKER_00")]))
    assert worker.annotation_to_turns(annotation) == [[0.0, 2.0, "SPEAKER_00"], [5.0, 6.0, "SPEAKER_01"]]


def test_main_prints_one_json_line_on_success(monkeypatch, capsys):
    monkeypatch.setattr(worker, "configure_cuda_environment", lambda: [])
    monkeypatch.setattr(worker, "run", lambda args: {"turns": [[0.0, 1.0, "A"]], "note": ""})
    assert worker.main(["--wav", "a.wav", "--pipeline", "/m"]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["turns"] == [[0.0, 1.0, "A"]]


def test_main_reports_failures_on_stderr_with_exit_1(monkeypatch, capsys):
    monkeypatch.setattr(worker, "configure_cuda_environment", lambda: [])

    def boom(args):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(worker, "run", boom)
    assert worker.main(["--wav", "a.wav", "--pipeline", "/m"]) == 1
    captured = capsys.readouterr()
    assert "RuntimeError: model exploded" in captured.err and captured.out == ""


def test_speaker_constraints_are_forwarded_to_the_pipeline(monkeypatch):
    received = {}

    class Output:
        exclusive_speaker_diarization = SimpleNamespace(itertracks=lambda yield_label: iter([]))

    class Pipeline:
        def to(self, device):
            received["device"] = device

        def __call__(self, audio, **kwargs):
            received["kwargs"] = kwargs
            return Output()

    fake_torch = SimpleNamespace(
        **vars(fake_cuda(arch_list=("sm_120",))),
        device=lambda kind, index: f"{kind}:{index}",
        from_numpy=FakeTensor,
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setitem(__import__("sys").modules, "pyannote", SimpleNamespace())
    monkeypatch.setitem(__import__("sys").modules, "pyannote.audio",
                        SimpleNamespace(Pipeline=SimpleNamespace(from_pretrained=lambda path: Pipeline())))
    monkeypatch.setattr(worker, "load_waveform", lambda path, torch: ("wave", 16000))
    args = argparse.Namespace(pipeline="/m", wav="a.wav", device="cuda", device_index=0,
                              num_speakers=None, min_speakers=2, max_speakers=4)
    payload = worker.run(args)
    assert received["kwargs"] == {"min_speakers": 2, "max_speakers": 4}
    assert received["device"] == "cuda:0" and payload == {"turns": [], "note": ""}
