import json

import pytest

from transcribe_media import formats

SEGMENTS = [
    {"start": 0.0, "end": 2.5, "text": " Hello world. ", "speaker": "Speaker 1"},
    {"start": 3661.5, "end": 3663.0, "text": "Later on.", "speaker": None},
    {"start": 4.0, "end": 5.0, "text": "   ", "speaker": None},
]


def test_plain_text_skips_blank_segments():
    assert formats.to_text(SEGMENTS) == "Speaker 1: Hello world.\nLater on.\n"


def test_text_with_timestamps_uses_hours_only_when_needed():
    out = formats.to_text(SEGMENTS, timestamps=True)
    assert out.splitlines() == ["[00:00] Speaker 1: Hello world.", "[1:01:02] Later on."]


def test_srt_numbering_and_millisecond_format():
    out = formats.to_srt(SEGMENTS)
    assert out.startswith("1\n00:00:00,000 --> 00:00:02,500\nSpeaker 1: Hello world.\n")
    assert "2\n01:01:01,500 --> 01:01:03,000\nLater on." in out
    assert "\n3\n" not in out


def test_vtt_header_and_voice_tag():
    out = formats.to_vtt(SEGMENTS)
    assert out.startswith("WEBVTT\n")
    assert "00:00:00.000 --> 00:00:02.500\n<v Speaker 1>Hello world." in out


def test_json_keeps_unicode_and_meta():
    data = json.loads(formats.to_json([{"start": 0, "end": 1, "text": "你好", "speaker": None}], {"language": "zh"}))
    assert data["meta"] == {"language": "zh"} and data["segments"][0]["text"] == "你好"
    assert "你好" in formats.to_json([{"text": "你好"}])


def test_empty_transcript_renders_empty_text():
    assert formats.to_text([]) == ""


def test_negative_times_are_clamped():
    assert formats._timestamp(-3, comma=True) == "00:00:00,000"


@pytest.mark.parametrize("fmt", formats.FORMATS)
def test_render_dispatch(fmt):
    assert formats.render(SEGMENTS, fmt, meta={"a": 1})


def test_render_rejects_unknown_format():
    with pytest.raises(ValueError, match="unknown output format"):
        formats.render(SEGMENTS, "docx")


# --- rounding, markup and blank lines ---------------------------------------------------------------------- #

@pytest.mark.parametrize("seconds, srt", [
    (1.9996, "00:00:02,000"),      # rounds up across a second
    (59.9996, "00:01:00,000"),     # ... and across a minute
    (3599.9996, "01:00:00,000"),   # ... and across an hour
    (0.0004, "00:00:00,000"),
    (2.0004, "00:00:02,000"),
    (2.0006, "00:00:02,001"),
])
def test_timestamps_round_to_the_nearest_millisecond(seconds, srt):
    assert formats._timestamp(seconds, comma=True) == srt
    assert formats._timestamp(seconds, comma=False) == srt.replace(",", ".")


def cue(text, speaker=None):
    return [{"start": 0.0, "end": 1.0, "text": text, "speaker": speaker}]


@pytest.mark.parametrize("text, escaped", [
    ("R&D budget", "R&amp;D budget"),
    ("if x <5 and y> 3", "if x &lt;5 and y&gt; 3"),
    ("a --> b", "a --&gt; b"),     # would otherwise look like a cue timing line
    ("<script>alert(1)</script>", "&lt;script&gt;alert(1)&lt;/script&gt;"),
])
def test_vtt_text_is_not_read_as_markup(text, escaped):
    assert f"\n{escaped}\n" in formats.to_vtt(cue(text))
    assert formats.to_vtt(cue(text, "Speaker 1")).count("<v Speaker 1>") == 1


def test_srt_and_text_keep_the_words_exactly_as_spoken():
    assert "R&D <5" in formats.to_srt(cue("R&D <5")) and formats.to_text(cue("R&D <5")) == "R&D <5\n"


@pytest.mark.parametrize("render", [formats.to_srt, formats.to_vtt])
def test_a_blank_line_inside_a_cue_cannot_end_it_early(render):
    out = render(cue("first\n\n  \nsecond"))
    assert "first\nsecond" in out and "\n\n\n" not in out


def test_prose_is_just_the_words_without_labels_or_times():
    segments = [{"start": 0, "end": 1, "text": " One two. ", "speaker": "Speaker 1"},
                {"start": 1, "end": 2, "text": "  ", "speaker": None},
                {"start": 2, "end": 3, "text": "Three.", "speaker": "Speaker 2"}]
    assert formats.to_prose(segments) == "One two. Three."
    assert formats.to_prose([]) == ""
