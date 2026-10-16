"""Accuracy-first transcription with faster-whisper (CTranslate2).

Decoding settings favour correctness over speed: full beam search, the whole
temperature-fallback ladder, voice-activity filtering and a hallucination
guard, with word timestamps (needed for speaker assignment and good subtitles).
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import audio
from .environment import is_remote
from .errors import TranscribeError
from .profiles import AsrProfile

Segment = dict[str, Any]
Progress = Callable[[float, float], None]  # (seconds transcribed, total seconds)

ACCURACY_OPTIONS: dict[str, Any] = {
    "best_of": 5,
    # Re-decode a window at a higher temperature when it fails the checks below.
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "word_timestamps": True,
    # Skip text Whisper invents across long silences (needs word timestamps).
    "hallucination_silence_threshold": 2.0,
    # Detect the language on up to four 30 s windows and take a majority vote. A threshold of
    # 1.0 can never be exceeded, so faster-whisper does not stop at the first confident window
    # (which a misleading intro could otherwise decide).
    "language_detection_segments": 4,
    "language_detection_threshold": 1.0,
}
DEFAULT_BEAM_SIZE = 5

_CUDA_LOAD_FAILURE = ("not found", "cannot be loaded", "cannot open shared object", "could not load")
_CUDA_LIBS = ("cublas", "cudnn", "libcudart", "nvrtc")
_ARCH_FAILURE = ("no kernel image", "unsupported gpu", "unsupported compute capability", "sm_120")


def explain_error(exc: BaseException) -> str:
    """Append an actionable remedy to common GPU failures."""
    msg = str(exc)
    low = msg.lower()
    if any(k in low for k in _CUDA_LIBS) and any(k in low for k in _CUDA_LOAD_FAILURE):
        return (
            f"{msg}\n  -> The NVIDIA CUDA libraries could not be loaded. Re-run scripts/setup.py "
            "(it installs the CUDA 12.8 cuBLAS/cuDNN wheels) or use --device cpu."
        )
    if any(k in low for k in _ARCH_FAILURE):
        return (
            f"{msg}\n  -> This GPU architecture is not supported by the installed CTranslate2. "
            "Blackwell (RTX 50-series) needs ctranslate2 >= 4.6.3 with CUDA 12.8: re-run scripts/setup.py."
        )
    if "out of memory" in low:
        hint = "pass --model medium" if is_remote() else "pass --model medium or --device cpu"
        return f"{msg}\n  -> GPU out of memory: close other GPU programs, or {hint}."
    return msg


@dataclass(frozen=True)
class TranscriptionResult:
    segments: list[Segment]
    language: str
    language_probability: float
    duration: float
    model: str
    device: str
    compute_type: str


def _default_factory() -> Callable[..., Any]:
    from faster_whisper import WhisperModel

    return WhisperModel


def _segment_dict(seg: Any) -> Segment:
    entry: Segment = {
        "start": float(seg.start),
        "end": float(seg.end),
        "text": seg.text.strip(),
        "speaker": None,
        "avg_logprob": seg.avg_logprob,
        "no_speech_prob": seg.no_speech_prob,
        "compression_ratio": seg.compression_ratio,
    }
    if seg.words:
        entry["words"] = [
            {"start": float(w.start), "end": float(w.end), "word": w.word, "probability": w.probability}
            for w in seg.words
        ]
    return entry


class Transcriber:
    def __init__(self, model_dir: str, profile: AsrProfile, *, model_factory: Callable[..., Any] | None = None):
        self.profile = profile
        self._model_name = profile.model
        try:
            factory = model_factory or _default_factory()
            self._model = factory(
                model_dir,
                device=profile.device,
                device_index=profile.device_index,
                compute_type=profile.compute_type,
                cpu_threads=max(4, (os.cpu_count() or 8) // 2),
            )
        except Exception as exc:  # CTranslate2 raises RuntimeError/ValueError for many causes
            raise TranscribeError(
                f"could not load model {profile.model!r} on {profile.device} "
                f"({profile.compute_type}): {explain_error(exc)}"
            ) from exc

    def transcribe(
        self,
        wav_path: str,
        *,
        language: str | None = None,
        prompt: str | None = None,
        beam_size: int = DEFAULT_BEAM_SIZE,
        condition_on_previous_text: bool = False,
        vad_filter: bool = True,
        progress: Progress | None = None,
    ) -> TranscriptionResult:
        """Transcribe a 16 kHz mono WAV.

        ``condition_on_previous_text`` is off by default: one bad window would
        otherwise poison the next and large-v3 can fall into a repetition loop.
        ``prompt`` is passed as hotwords, which faster-whisper adds to the context of
        every window; an initial prompt would be forgotten after the first 30 seconds.
        """
        try:
            samples = audio.load_samples(Path(wav_path))
            segments_iter, info = self._model.transcribe(
                samples,
                language=language,
                hotwords=prompt,
                beam_size=beam_size,
                condition_on_previous_text=condition_on_previous_text,
                vad_filter=vad_filter,
                **ACCURACY_OPTIONS,
            )
            segments = []
            for seg in segments_iter:  # iterating is what actually runs inference
                segments.append(_segment_dict(seg))
                if progress:
                    progress(float(seg.end), float(info.duration))
        except Exception as exc:
            raise TranscribeError(
                f"transcription failed on {self.profile.device} ({self.profile.compute_type}): "
                f"{explain_error(exc)}"
            ) from exc
        return TranscriptionResult(
            segments=segments,
            language=info.language,
            language_probability=float(info.language_probability),
            duration=float(info.duration),
            model=self._model_name,
            device=self.profile.device,
            compute_type=self.profile.compute_type,
        )

    def close(self) -> None:
        """Release the model (and its GPU memory) before another model needs the card."""
        self._model = None
        gc.collect()
