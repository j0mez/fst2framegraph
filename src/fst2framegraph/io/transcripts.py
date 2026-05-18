from __future__ import annotations

from typing import Any
import re

import pandas as pd


_MARKER_RE = re.compile(
    r"\[(?P<label>ad\s+text|advert\s+text|text|audio\s+transcript|video\s+transcript|audio|video)\s*:\]",
    flags=re.IGNORECASE,
)
_AD_LABELS = {"ad text", "advert text", "text"}
_WS_RE = re.compile(r"\s+")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _normalise_label(label: str) -> str:
    return _WS_RE.sub(" ", label.strip().lower())


def _normalise_text(value: str) -> str:
    value = value.replace("\r", "\n")
    return _WS_RE.sub(" ", value).strip()


def clean_transcript(raw_text: object) -> str:
    """Extract analyzable ad copy from OxCCAL-style transcript fields.

    ``Transcript (text and audio)`` rows can combine written ad text with
    descriptive audio/video transcript sections. When explicit markers are
    present, only ``[ad text:]``/``[text:]`` sections are retained; audio and
    video sections are discarded. Unmarked text is returned after whitespace
    normalization.
    """
    if _is_missing(raw_text):
        return ""

    value = str(raw_text).strip()
    if not value:
        return ""

    matches = list(_MARKER_RE.finditer(value))
    if not matches:
        return _normalise_text(value)

    ad_segments: list[str] = []
    for index, match in enumerate(matches):
        label = _normalise_label(match.group("label"))
        segment_start = match.end()
        segment_end = matches[index + 1].start() if index + 1 < len(matches) else len(value)
        if label in _AD_LABELS:
            segment = _normalise_text(value[segment_start:segment_end])
            if segment:
                ad_segments.append(segment)

    return _normalise_text(" ".join(ad_segments))
