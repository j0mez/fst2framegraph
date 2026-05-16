from __future__ import annotations

from .text import clean_text


def keep_filler(value: object, min_len: int = 1) -> bool:
    return len(clean_text(value)) >= min_len
