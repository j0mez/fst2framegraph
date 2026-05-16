from __future__ import annotations

import re
import unicodedata


_WS = re.compile(r"\s+")


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()
    text = _WS.sub(" ", text)
    return text


def normalise_for_match(value: object) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[^\w\s%.-]", " ", text)
    text = _WS.sub(" ", text).strip()
    return text


def snakeish(value: object) -> str:
    text = clean_text(value)
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower() or "empty"
