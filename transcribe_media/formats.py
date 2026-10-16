"""Transcript writers. Standard library only.

A transcript is a list of segment dicts:
``{"start": float, "end": float, "text": str, "speaker": str | None}``
with times in seconds. ``speaker`` is set only when diarization ran.
"""

from __future__ import annotations

import html
import json
import re
from typing import Any, Iterable

Segment = dict[str, Any]
FORMATS = ("txt", "json", "srt", "vtt")


def _timestamp(seconds: float, *, comma: bool) -> str:
    millis = int(round(max(seconds, 0.0) * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{',' if comma else '.'}{millis:03d}"


def _clock(seconds: float) -> str:
    total = int(round(max(seconds, 0.0)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _spoken(segments: Iterable[Segment]) -> list[Segment]:
    return [s for s in segments if s["text"].strip()]


def to_prose(segments: Iterable[Segment]) -> str:
    """Just the words, with no timestamps or speaker labels (for comparing with another transcript)."""
    return " ".join(seg["text"].strip() for seg in _spoken(segments))


def _cue_text(seg: Segment) -> str:
    """Cue text for SRT/VTT: a blank line would end the cue, so blank lines are dropped."""
    return re.sub(r"\n\s*\n", "\n", seg["text"].strip())


def to_text(segments: Iterable[Segment], *, timestamps: bool = False) -> str:
    lines = []
    for seg in _spoken(segments):
        prefix = []
        if timestamps:
            prefix.append(f"[{_clock(seg['start'])}]")
        if seg.get("speaker"):
            prefix.append(f"{seg['speaker']}:")
        lines.append(" ".join([*prefix, seg["text"].strip()]))
    return "\n".join(lines) + ("\n" if lines else "")


def to_srt(segments: Iterable[Segment]) -> str:
    blocks = []
    for index, seg in enumerate(_spoken(segments), start=1):
        text = _cue_text(seg)
        if seg.get("speaker"):
            text = f"{seg['speaker']}: {text}"
        start, end = _timestamp(seg["start"], comma=True), _timestamp(seg["end"], comma=True)
        blocks.append(f"{index}\n{start} --> {end}\n{text}\n")
    return "\n".join(blocks)


def to_vtt(segments: Iterable[Segment]) -> str:
    blocks = ["WEBVTT\n"]
    for seg in _spoken(segments):
        text = html.escape(_cue_text(seg), quote=False)  # WebVTT reads "<" and "&" as markup
        if seg.get("speaker"):
            text = f"<v {html.escape(seg['speaker'], quote=False)}>{text}"
        start, end = _timestamp(seg["start"], comma=False), _timestamp(seg["end"], comma=False)
        blocks.append(f"{start} --> {end}\n{text}\n")
    return "\n".join(blocks)


def to_json(segments: Iterable[Segment], meta: dict[str, Any] | None = None) -> str:
    payload: dict[str, Any] = {"meta": meta or {}, "segments": list(segments)}
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def render(segments: Iterable[Segment], fmt: str, *, meta: dict[str, Any] | None = None,
           timestamps: bool = False) -> str:
    segs = list(segments)
    if fmt == "txt":
        return to_text(segs, timestamps=timestamps)
    if fmt == "srt":
        return to_srt(segs)
    if fmt == "vtt":
        return to_vtt(segs)
    if fmt == "json":
        return to_json(segs, meta)
    raise ValueError(f"unknown output format {fmt!r} (choose from {', '.join(FORMATS)})")
