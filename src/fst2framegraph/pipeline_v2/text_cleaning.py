from __future__ import annotations

import re


MARKER_RE = re.compile(
    r"\[ad text:\]|\[audio transcript:\]|\[video transcript:\]|\[text:\]|\[audio:\]",
    flags=re.I,
)
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.I)
WS_RE = re.compile(r"\s+")


def clean_text_input(value: object) -> str:
    text = "" if value is None else str(value)
    if not text:
        return ""
    text = MARKER_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def normalise_for_dedupe(value: object) -> str:
    return WS_RE.sub(" ", clean_text_input(value)).strip().lower()
