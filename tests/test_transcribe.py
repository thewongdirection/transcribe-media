import wave
from types import SimpleNamespace

import numpy as np
import pytest

from transcribe_media.errors import TranscribeError
from transcribe_media.profiles import AsrProfile
from transcribe_media.transcribe import ACCURACY_OPTIONS, Transcriber, explain_error

PROFILE = AsrProfile("large-v3", "cuda", "float16", device_index=1)


@pytest.fixture
def wav(tmp_path):
    """One second of silence in the format the pipeline produces."""
    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    return str(path)


def word(text, start, end, probability=0.9):
    return SimpleNamespace(word=text, start=start, end=end, probability=probability)


def seg(start, end, text, words=None):
    return SimpleNamespace(
        start=start, end=end, text=text, avg_logprob=-0.2, no_speech_prob=0.01,
        compression_ratio=1.1, words=words,
    )


class FakeModel:
    def __init__(self, *args, **kwargs):
        self.init = (args, kwargs)
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append((path, kwargs))
        segments = [
            seg(0.0, 2.0, " Hello there. ", [word(" Hello", 0.0, 0.8), word(" there.", 0.9, 2.0)]),
            seg(2.5, 4.0, " Second line. "),
        ]
        info = SimpleNamespace(language="en", language_probability=0.98, duration=10.0)
        return iter(segments), info


def make(**kwargs):
    holder = {}

    def factory(*args, **kw):
        holder["model"] = FakeModel(*args, **kw)
        return holder["model"]

    return Transcriber("/models/large-v3", PROFILE, model_factory=factory, **kwargs), holder


def test_model_is_loaded_with_the_selected_profile():
    _, holder = make()
    args, kwargs = holder["model"].init
    assert args == ("/models/large-v3",)
    assert (kwargs["device"], kwargs["device_index"], kwargs["compute_type"]) == ("cuda", 1, "float16")


def test_accuracy_first_decoding_options_are_applied(wav):
    transcriber, holder = make()
    transcriber.transcribe(wav, prompt="Acme, Zorblax")
    audio_arg, kw = holder["model"].calls[0]
    # Regression: decoding must not go through faster-whisper's PyAV call, which
    # breaks with recent PyAV releases, so the model receives samples, not a path.
    assert isinstance(audio_arg, np.ndarray) and audio_arg.dtype == np.float32
    assert audio_arg.shape == (16000,)
    assert kw["beam_size"] == 5 and kw["word_timestamps"] is True and kw["vad_filter"] is True
    assert kw["hallucination_silence_threshold"] == 2.0
    assert kw["condition_on_previous_text"] is False
    # The prompt goes in as hotwords: faster-whisper adds those to EVERY window, whereas an
    # initial prompt is forgotten after the first 30 seconds when text is not carried over.
    assert kw["hotwords"] == "Acme, Zorblax" and "initial_prompt" not in kw and kw["language"] is None
    assert kw["temperature"][0] == 0.0 and len(kw["temperature"]) > 1
    assert all(kw[k] == v for k, v in ACCURACY_OPTIONS.items())


def test_every_option_exists_in_the_installed_faster_whisper():
    """The fake model accepts anything, so check names against the real signature."""
    import inspect

    from faster_whisper import WhisperModel

    accepted = set(inspect.signature(WhisperModel.transcribe).parameters)
    passed = set(ACCURACY_OPTIONS) | {
        "language", "hotwords", "beam_size", "condition_on_previous_text", "vad_filter",
    }
    assert passed <= accepted, f"unknown faster-whisper options: {sorted(passed - accepted)}"


def test_user_overrides_reach_the_model(wav):
    transcriber, holder = make()
    transcriber.transcribe(wav, language="de", beam_size=10, condition_on_previous_text=True)
    kw = holder["model"].calls[0][1]
    assert (kw["language"], kw["beam_size"], kw["condition_on_previous_text"]) == ("de", 10, True)


def test_segments_are_plain_dicts_with_words_and_quality_signals(wav):
    transcriber, _ = make()
    result = transcriber.transcribe(wav)
    first, second = result.segments
    assert first["text"] == "Hello there." and first["speaker"] is None
    assert first["words"][0] == {"start": 0.0, "end": 0.8, "word": " Hello", "probability": 0.9}
    assert "words" not in second
    assert first["avg_logprob"] == -0.2 and first["compression_ratio"] == 1.1
    assert (result.language, result.language_probability, result.duration) == ("en", 0.98, 10.0)
    assert (result.model, result.device, result.compute_type) == ("large-v3", "cuda", "float16")


def test_progress_reports_each_segment_against_total_duration(wav):
    transcriber, _ = make()
    seen = []
    transcriber.transcribe(wav, progress=lambda done, total: seen.append((done, total)))
    assert seen == [(2.0, 10.0), (4.0, 10.0)]


def test_load_failure_is_explained():
    def factory(*a, **k):
        raise RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")

    with pytest.raises(TranscribeError, match="setup.py"):
        Transcriber("/m", PROFILE, model_factory=factory)


def test_inference_failure_is_wrapped(wav):
    transcriber, holder = make()
    holder["model"].transcribe = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("CUDA out of memory"))
    with pytest.raises(TranscribeError, match="out of memory.*medium"):
        transcriber.transcribe(wav)


def test_close_releases_the_model():
    transcriber, _ = make()
    transcriber.close()
    assert transcriber._model is None


@pytest.mark.parametrize("message, fragment", [
    ("Library cudnn_ops64_9.dll is not found", "CUDA libraries"),
    ("CUDA failed: no kernel image is available for execution", "Blackwell"),
    ("something unrelated", None),
])
def test_explain_error(message, fragment):
    out = explain_error(RuntimeError(message))
    assert out.startswith(message)
    if fragment:
        assert fragment in out
    else:
        assert out == message


def test_vad_can_be_switched_off(wav):
    transcriber, holder = make()
    transcriber.transcribe(wav, vad_filter=False)
    assert holder["model"].calls[0][1]["vad_filter"] is False


def test_language_detection_votes_over_all_windows_instead_of_stopping_at_the_first_confident_one():
    # faster-whisper stops at the first window whose probability EXCEEDS the threshold; 1.0 never can be.
    assert ACCURACY_OPTIONS["language_detection_threshold"] == 1.0
    assert ACCURACY_OPTIONS["language_detection_segments"] == 4


def test_out_of_memory_advice_differs_in_the_cloud(monkeypatch):
    monkeypatch.delenv("TRANSCRIBE_MEDIA_REMOTE", raising=False)
    assert "--device cpu" in explain_error(RuntimeError("CUDA out of memory"))
    monkeypatch.setenv("TRANSCRIBE_MEDIA_REMOTE", "1")
    assert "--device cpu" not in explain_error(RuntimeError("CUDA out of memory"))
