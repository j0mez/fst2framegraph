from __future__ import annotations

import re

from fst2framegraph.io.transcripts import clean_transcript

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.I)
WS_RE = re.compile(r"\s+")


def clean_text_input(value: object) -> str:
    text = clean_transcript(value)
    if not text:
        return ""
    text = URL_RE.sub(" ", text)
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def normalise_for_dedupe(value: object) -> str:
    return WS_RE.sub(" ", clean_text_input(value)).strip().lower()
