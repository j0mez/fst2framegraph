from __future__ import annotations

import hashlib
import re

import pandas as pd

from .text_cleaning import clean_text_input, normalise_for_dedupe


WS_RE = re.compile(r"\s+")


def split_into_chunks(text: object, *, min_words: int = 2, max_words: int = 70) -> list[str]:
    clean = clean_text_input(text)
    if not clean:
        return []

    rough_parts: list[str] = []
    for line in clean.split("\n"):
        line = line.strip()
        if not line:
            continue
        rough_parts.extend(re.split(r"(?<=[.!?])\s+", line))

    chunks: list[str] = []
    for part in rough_parts:
        part = WS_RE.sub(" ", part).strip()
        if not part:
            continue
        words = part.split()
        if len(words) < min_words:
            continue
        if len(words) <= max_words:
            chunks.append(part)
            continue
        subparts = re.split(r"(?<=[,;:])\s+", part)
        buffer: list[str] = []
        for sub in subparts:
            sub_words = sub.split()
            if len(buffer) + len(sub_words) <= max_words:
                buffer.extend(sub_words)
            else:
                if len(buffer) >= min_words:
                    chunks.append(" ".join(buffer).strip())
                buffer = sub_words
        if len(buffer) >= min_words:
            chunks.append(" ".join(buffer).strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        key = normalise_for_dedupe(chunk)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)
    return deduped


def stable_chunk_hash(text: str) -> str:
    return hashlib.sha1(normalise_for_dedupe(text).encode("utf-8")).hexdigest()


def build_chunk_table(
    source_df: pd.DataFrame,
    *,
    min_words: int = 2,
    max_words: int = 70,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sentence_rows: list[dict[str, object]] = []
    mapping_rows: list[dict[str, object]] = []

    for _, row in source_df.iterrows():
        source_id = str(row["source_id"])
        source_doc_id = str(row["source_doc_id"])
        raw_text = row["raw_text"]
        chunks = split_into_chunks(raw_text, min_words=min_words, max_words=max_words)
        for chunk_index, chunk in enumerate(chunks):
            sentence_id = f"{source_id}__chunk_{chunk_index:03d}"
            unique_chunk_id = stable_chunk_hash(chunk)
            sentence_rows.append(
                {
                    "sentence_id": sentence_id,
                    "doc_id": source_doc_id,
                    "sentence": chunk,
                    "source_id": source_id,
                    "source_doc_id": source_doc_id,
                    "source_row_index": int(row["source_row_index"]),
                }
            )
            mapping_rows.append(
                {
                    "source_id": source_id,
                    "source_doc_id": source_doc_id,
                    "source_row_index": int(row["source_row_index"]),
                    "sentence_id": sentence_id,
                    "unique_chunk_id": unique_chunk_id,
                    "chunk_index": chunk_index,
                    "chunk_text": chunk,
                }
            )

    sentence_df = pd.DataFrame(sentence_rows)
    mapping_df = pd.DataFrame(mapping_rows)
    if sentence_df.empty:
        sentence_df = pd.DataFrame(
            columns=["sentence_id", "doc_id", "sentence", "source_id", "source_doc_id", "source_row_index"]
        )
    if mapping_df.empty:
        mapping_df = pd.DataFrame(
            columns=[
                "source_id",
                "source_doc_id",
                "source_row_index",
                "sentence_id",
                "unique_chunk_id",
                "chunk_index",
                "chunk_text",
            ]
        )
    return sentence_df, mapping_df
