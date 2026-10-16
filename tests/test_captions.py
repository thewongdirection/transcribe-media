from transcribe_media.captions import (
    CaptionAvailability,
    compare_texts,
    cues_text,
    find_captions,
    parse_json3,
    tokenize,
)

ORIGINAL = "https://www.youtube.com/api/timedtext?v=x&lang=en&kind=asr&fmt=json3"
TRANSLATED = ORIGINAL + "&tlang=de"
# Shape of the HLS rendition entries yt-dlp also lists: no query string at all.
HLS = "https://manifest.googlevideo.com/api/manifest/hls_timedtext_playlist/id/x/tts_params/caps%3Dasr%26lang%3Dde-DE"


def youtube_info(subtitles=None, auto=None, extractor="Youtube"):
    return {"extractor_key": extractor, "subtitles": subtitles or {}, "automatic_captions": auto or {}}


def test_non_youtube_sites_are_not_reported():
    assert find_captions(youtube_info(subtitles={"en": [{}]}, extractor="Vimeo")) is None


def test_video_without_captions():
    found = find_captions(youtube_info())
    assert found is not None and not found.any


def test_machine_translations_are_never_offered():
    info = youtube_info(
        subtitles={"en": [{}], "de": [{}], "live_chat": [{}]},
        auto={
            "en": [{"url": ORIGINAL}],
            "en-orig": [{"url": ORIGINAL}],
            "de": [{"url": TRANSLATED}],
            "ja": [{"url": ORIGINAL + "&tlang=ja"}],
        },
    )
    found = find_captions(info)
    assert found.manual == ("de", "en")
    assert found.auto == ("en",)


def test_auto_tracks_that_are_all_translations_yield_nothing():
    info = youtube_info(subtitles={"en": [{}]}, auto={"en": [{"url": TRANSLATED}]})
    assert find_captions(info).auto == ()


def test_hls_rendition_entries_are_not_mistaken_for_speech_recognition():
    # Real shape from a video with creator subtitles but no ASR track: the
    # language shows up under automatic_captions only as HLS (vtt) entries.
    info = youtube_info(
        subtitles={"de-DE": [{}]},
        auto={"de-DE": [{"ext": "vtt", "url": HLS}], "ja": [{"url": HLS}, {"url": TRANSLATED}]},
    )
    assert find_captions(info).auto == ()


def test_genuine_track_is_found_even_beside_an_hls_entry():
    info = youtube_info(auto={"en": [{"ext": "vtt", "url": HLS}, {"ext": "json3", "url": ORIGINAL}]})
    assert find_captions(info).auto == ("en",)


def test_notice_names_both_kinds_and_the_flag():
    text = CaptionAvailability(("en", "de"), ("en",)).notice()
    assert "creator-uploaded subtitles (en, de)" in text
    assert "auto-generated captions (en)" in text
    assert "--compare-captions" in text


def test_pick_prefers_manual_and_matches_language_prefix():
    avail = CaptionAvailability(manual=("en-GB", "de"), auto=("en",))
    assert avail.pick("en") == ("en-GB", False)
    assert CaptionAvailability((), ("en",)).pick("EN-us") == ("en", True)
    assert avail.pick("ja") is None
    assert avail.pick(None) is None


def test_parse_json3_keeps_speech_and_drops_noise():
    data = {
        "events": [
            {"tStartMs": 320, "dDurationMs": 14260, "segs": [{"utf8": "[Music]"}]},
            {"tStartMs": 18790, "segs": [{"utf8": "\n"}]},
            {"tStartMs": 1000, "dDurationMs": 2500, "segs": [{"utf8": "We're no "}, {"utf8": "strangers"}]},
            {"tStartMs": 4000, "dDurationMs": 1000, "segs": [{"utf8": "(laughs) that was great (really)"}]},
            {"tStartMs": 5000, "dDurationMs": 1000, "segs": [{"utf8": "(applause)"}]},
            {"tStartMs": 6000},
        ]
    }
    cues = parse_json3(data)
    assert [c.text for c in cues] == ["We're no strangers", "(laughs) that was great (really)"]
    assert (cues[0].start, cues[0].end) == (1.0, 3.5)
    assert cues_text(cues).endswith("\n")


def test_tokenize_handles_case_punctuation_and_cjk():
    assert tokenize("Hello, World! It's 5pm.") == ["hello", "world", "it", "s", "5pm"]
    assert tokenize("你好 world ありがとう") == ["你", "好", "world", "あ", "り", "が", "と", "う"]


def test_identical_texts_match_fully_regardless_of_formatting():
    assert compare_texts("Hello, world.", "hello world").similarity == 1.0


def test_partial_overlap_is_reported():
    result = compare_texts("the quick brown fox jumps", "the quick red fox leaps")
    assert result.similarity == 3 / 5
    assert (result.our_words, result.youtube_words) == (5, 5)
    assert "60.0%" in result.summary("auto captions")


def test_empty_side_scores_zero():
    assert compare_texts("", "something").similarity == 0.0
