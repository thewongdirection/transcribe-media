"""Transcript quality verdict from Whisper's own per-segment signals.

Flags empty, near-silent or noisy recordings whose transcripts are likely
garbled or hallucinated. Verdicts are heuristics, so by default they warn
rather than discard output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

VERDICT_OK = "ok"
VERDICT_SUSPECT = "suspect"
VERDICT_BAD = "bad"

MIN_SPEECH_COVERAGE = 0.15
MAX_COMPRESSION_RATIO = 2.4
MIN_AVG_LOGPROB = -1.0
MAX_NO_SPEECH_PROB = 0.6
# Hindi and Urdu sound nearly identical, so detection there is shakier.
LANGUAGE_CONFIDENCE_FLOOR = 0.8
LANGUAGE_CONFIDENCE_FLOOR_HI_UR = 0.9


@dataclass
class QualityReport:
    verdict: str
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def is_bad(self) -> bool:
        return self.verdict == VERDICT_BAD

    def summary(self) -> str:
        head = f"quality: {self.verdict.upper()}"
        return f"{head} - {'; '.join(self.reasons)}" if self.reasons else head

    def as_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "reasons": self.reasons, "metrics": self.metrics}


def language_note(language: str | None, probability: float | None) -> str | None:
    """A warning when auto-detection was not confident, else None."""
    if not language or probability is None:
        return None
    floor = LANGUAGE_CONFIDENCE_FLOOR_HI_UR if language in ("hi", "ur") else LANGUAGE_CONFIDENCE_FLOOR
    if probability >= floor:
        return None
    return (
        f"language detection is uncertain (detected '{language}' at {probability:.2f}); "
        "if the transcript looks wrong, re-run with --language"
    )


def _mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _round(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def assess(segments: list[dict[str, Any]], duration: float | None) -> QualityReport:
    """Judge a finished transcript."""
    text = "".join(s.get("text", "") for s in segments).strip()
    if not text:
        return QualityReport(VERDICT_BAD, ["no speech detected (empty transcript)"], {"num_segments": 0})

    speech_time = sum(max(0.0, float(s["end"]) - float(s["start"])) for s in segments)
    coverage = speech_time / duration if duration and duration > 0 else None
    no_speech = _mean([s.get("no_speech_prob") for s in segments])
    logprob = _mean([s.get("avg_logprob") for s in segments])
    ratios = [s["compression_ratio"] for s in segments if s.get("compression_ratio") is not None]
    max_ratio = max(ratios) if ratios else None

    metrics = {
        "num_segments": len(segments),
        "char_count": len(text),
        "speech_coverage": _round(coverage),
        "avg_no_speech_prob": _round(no_speech),
        "avg_logprob": _round(logprob),
        "max_compression_ratio": _round(max_ratio),
    }
    bad: list[str] = []
    suspect: list[str] = []

    strong_no_speech = no_speech is not None and no_speech >= MAX_NO_SPEECH_PROB
    if coverage is not None and (coverage < 0.02 or (strong_no_speech and coverage < MIN_SPEECH_COVERAGE)):
        bad.append(f"little or no speech (coverage {coverage:.0%})")
    else:
        if coverage is not None and coverage < MIN_SPEECH_COVERAGE:
            suspect.append(f"low speech coverage ({coverage:.0%})")
        if strong_no_speech:
            suspect.append(f"high no-speech probability ({no_speech:.2f})")
    if logprob is not None and logprob < MIN_AVG_LOGPROB:
        suspect.append(f"low transcription confidence (avg_logprob {logprob:.2f})")
    if max_ratio is not None and max_ratio > MAX_COMPRESSION_RATIO:
        suspect.append(f"repetitive output (compression ratio {max_ratio:.2f}) - possible noise")

    if bad:
        return QualityReport(VERDICT_BAD, bad + suspect, metrics)
    return QualityReport(VERDICT_SUSPECT if suspect else VERDICT_OK, suspect, metrics)
