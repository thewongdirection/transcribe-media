"""YouTube caption availability and comparison against our own transcript.

The transcript is always produced by Whisper. YouTube captions are only ever
reported, and optionally fetched so their text can be compared with ours.

yt-dlp's ``automatic_captions`` mixes three kinds of entry, told apart by the
query string of each track URL (verified against live videos):

* genuine speech recognition: ``kind=asr`` and no ``tlang``  -> offered;
* machine translation of that track: has ``tlang``           -> never offered;
* HLS manifest renditions: no query string at all            -> ignored.

Requiring the positive ``kind=asr`` marker means an unrecognised entry is
under-reported rather than mistaken for a real transcript.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import parse_qs, urlparse

_ORIG_SUFFIX = "-orig"
# Track names become part of output filenames, so only plain language codes are accepted.
_LANGUAGE_CODE = re.compile(r"[A-Za-z0-9]{1,8}(-[A-Za-z0-9]{1,8}){0,3}")
# A whole cue that is a single sound tag such as "[Music]" or "(applause)".
_NON_SPEECH = re.compile(r"^(\[[^\]]*\]|\([^)]*\))$")
# Kana, CJK ideographs and Hangul count one token per character; other scripts per word.
_TOKEN = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]|[^\W_]+")


@dataclass(frozen=True)
class CaptionAvailability:
    manual: tuple[str, ...]  # creator-uploaded subtitle languages
    auto: tuple[str, ...]  # genuine (not machine-translated) auto-caption languages

    @property
    def any(self) -> bool:
        return bool(self.manual or self.auto)

    def notice(self) -> str:
        parts = []
        if self.manual:
            parts.append(f"creator-uploaded subtitles ({', '.join(self.manual)})")
        if self.auto:
            parts.append(f"auto-generated captions ({', '.join(self.auto)})")
        return (
            f"YouTube also has {' and '.join(parts)} for this video. "
            "Re-run with --compare-captions to fetch them and compare against this transcript."
        )

    def pick(self, language: str | None) -> tuple[str, bool] | None:
        """(track language, is_auto) best matching ``language``; manual beats auto."""
        for candidates, auto in ((self.manual, False), (self.auto, True)):
            for track in candidates:
                if language and track.split("-")[0].lower() == language.split("-")[0].lower():
                    return track, auto
        return None


def _is_speech_recognition(entries: Iterable[dict[str, Any]]) -> bool:
    for entry in entries:
        query = parse_qs(urlparse(entry.get("url", "")).query)
        if query.get("kind") == ["asr"] and "tlang" not in query:
            return True
    return False


def find_captions(info: dict[str, Any]) -> CaptionAvailability | None:
    """Caption availability for a YouTube video; None for other sites."""
    if not str(info.get("extractor_key", "")).lower().startswith("youtube"):
        return None
    manual = sorted(k for k in (info.get("subtitles") or {}) if k != "live_chat" and _LANGUAGE_CODE.fullmatch(k))
    auto: set[str] = set()
    for lang, entries in (info.get("automatic_captions") or {}).items():
        code = lang[: -len(_ORIG_SUFFIX)] if lang.endswith(_ORIG_SUFFIX) else lang
        if _LANGUAGE_CODE.fullmatch(code) and _is_speech_recognition(entries):
            auto.add(code)
    return CaptionAvailability(manual=tuple(manual), auto=tuple(sorted(auto)))


@dataclass(frozen=True)
class CaptionCue:
    start: float
    end: float
    text: str


def parse_json3(data: dict[str, Any]) -> list[CaptionCue]:
    """Speech cues from a YouTube json3 caption document."""
    cues = []
    for event in data.get("events", []):
        segs = event.get("segs")
        if not segs:
            continue
        text = "".join(s.get("utf8", "") for s in segs).strip()
        if not text or _NON_SPEECH.match(text):
            continue
        start = event.get("tStartMs", 0) / 1000
        cues.append(CaptionCue(start, start + event.get("dDurationMs", 0) / 1000, text))
    return cues


def load_json3(path: Path) -> list[CaptionCue]:
    return parse_json3(json.loads(path.read_text(encoding="utf-8")))


def tokenize(text: str) -> list[str]:
    """Case-folded words; CJK text is split per character since it has no spaces."""
    return _TOKEN.findall(text.casefold())


@dataclass(frozen=True)
class Comparison:
    similarity: float  # share of words that line up, 0..1
    our_words: int
    youtube_words: int

    def summary(self, label: str) -> str:
        return (
            f"Compared with YouTube {label}: {self.similarity:.1%} of words match "
            f"({self.our_words} words here, {self.youtube_words} there). "
            "Neither side is ground truth; review the differences before relying on either."
        )


def compare_texts(ours: str, theirs: str) -> Comparison:
    a, b = tokenize(ours), tokenize(theirs)
    if not a or not b:
        return Comparison(0.0, len(a), len(b))
    matched = sum(m.size for m in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())
    return Comparison(matched / max(len(a), len(b)), len(a), len(b))


def cues_text(cues: Sequence[CaptionCue]) -> str:
    return "\n".join(c.text for c in cues) + ("\n" if cues else "")
