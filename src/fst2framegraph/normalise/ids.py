from __future__ import annotations

import hashlib

from .text import clean_text, snakeish


def short_hash(*parts: object, length: int = 12) -> str:
    payload = "||".join(clean_text(p) for p in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def make_document_id(doc: object) -> str:
    return f"doc_{snakeish(doc)[:60]}"


def make_sentence_id(doc: object, sentence: object, explicit: object | None = None) -> str:
    if explicit not in (None, ""):
        return f"sent_{snakeish(explicit)[:80]}"
    return f"sent_{short_hash(doc, sentence)}"


def make_frame_instance_id(doc: object, sentence_id: object, frame_name: object, frame_index: object | None = None) -> str:
    if frame_index not in (None, ""):
        return f"frame_{short_hash(doc, sentence_id, frame_name, frame_index)}"
    return f"frame_{short_hash(doc, sentence_id, frame_name)}"


def make_filler_id(filler_text: object) -> str:
    return f"filler_{short_hash(filler_text)}"
