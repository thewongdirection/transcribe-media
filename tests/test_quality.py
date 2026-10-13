from transcribe_media.quality import VERDICT_BAD, VERDICT_OK, VERDICT_SUSPECT, assess, language_note


def seg(start, end, text="words here", no_speech=0.02, logprob=-0.2, ratio=1.2):
    return {"start": start, "end": end, "text": text, "no_speech_prob": no_speech,
            "avg_logprob": logprob, "compression_ratio": ratio}


def test_healthy_transcript_is_ok():
    report = assess([seg(0, 40), seg(40, 90)], duration=100)
    assert report.verdict == VERDICT_OK and report.reasons == []
    assert report.metrics["speech_coverage"] == 0.9
    assert report.summary() == "quality: OK"


def test_empty_transcript_is_bad():
    report = assess([seg(0, 1, text="  ")], duration=60)
    assert report.is_bad and "no speech" in report.reasons[0]


def test_almost_no_speech_is_bad():
    assert assess([seg(0, 1)], duration=600).verdict == VERDICT_BAD


def test_low_coverage_is_only_suspect():
    report = assess([seg(0, 12)], duration=100)
    assert report.verdict == VERDICT_SUSPECT and "coverage" in report.reasons[0]


def test_high_no_speech_probability_with_low_coverage_is_bad():
    assert assess([seg(0, 10, no_speech=0.9)], duration=100).is_bad


def test_low_confidence_and_repetition_are_flagged():
    report = assess([seg(0, 50, logprob=-1.4, ratio=3.0)], duration=60)
    assert report.verdict == VERDICT_SUSPECT
    joined = " ".join(report.reasons)
    assert "confidence" in joined and "repetitive" in joined


def test_unknown_duration_skips_coverage_checks():
    report = assess([seg(0, 5)], duration=None)
    assert report.verdict == VERDICT_OK and report.metrics["speech_coverage"] is None


def test_as_dict_round_trips_fields():
    report = assess([seg(0, 50, logprob=-1.4)], duration=60)
    assert set(report.as_dict()) == {"verdict", "reasons", "metrics"}


def test_language_note_thresholds():
    assert language_note("en", 0.95) is None
    assert "uncertain" in language_note("en", 0.6)
    assert language_note("hi", 0.85) is not None  # stricter floor for Hindi/Urdu
    assert language_note("fr", 0.85) is None
    assert language_note(None, 0.5) is None and language_note("en", None) is None
