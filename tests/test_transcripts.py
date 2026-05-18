from __future__ import annotations

from fst2framegraph.io.transcripts import clean_transcript


def test_clean_transcript_extracts_ad_text_and_discards_audio_marker() -> None:
    raw = """
    [ad text:] We invest in clean power. Visit example.com.
    [audio transcript:] dramatic music and voice-over repeats the slogan.
    """

    assert clean_transcript(raw) == "We invest in clean power. Visit example.com."


def test_clean_transcript_handles_marker_case_and_audio_only_rows() -> None:
    assert clean_transcript("[AD TEXT:] Safer homes for everyone [AUDIO TRANSCRIPT:] music") == (
        "Safer homes for everyone"
    )
    assert clean_transcript("[audio transcript:] music only") == ""
    assert clean_transcript(None) == ""
