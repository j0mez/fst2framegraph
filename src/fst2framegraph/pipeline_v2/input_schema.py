from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


TEXT_CANDIDATES = (
    "sentence",
    "text",
    "Transcript (text and audio)",
    "Transcript",
    "transcript",
)
ID_CANDIDATES = ("sentence_id", "id", "Unique ID")
DOC_CANDIDATES = ("doc_id", "document_id", "Unique ID")


@dataclass(frozen=True)
class InputColumns:
    text_col: str
    id_col: str | None
    doc_col: str | None
    metadata_cols: tuple[str, ...]


def _resolve_explicit_or_candidates(
    df: pd.DataFrame,
    *,
    explicit: str | None,
    candidates: tuple[str, ...],
    label: str,
    required: bool,
) -> str | None:
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(
                f"{label} column {explicit!r} was not found. Available columns: {list(df.columns)}"
            )
        return explicit

    for name in candidates:
        if name in df.columns:
            return name
    if not required:
        return None
    raise ValueError(
        f"Could not auto-detect {label} column. Provide --{label.replace('_', '-')}. "
        f"Available columns: {list(df.columns)}"
    )


def load_input_csv(
    path: str | Path,
    *,
    text_col: str | None = None,
    id_col: str | None = None,
    doc_col: str | None = None,
    metadata_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, InputColumns]:
    df = pd.read_csv(path)

    resolved_text = _resolve_explicit_or_candidates(
        df, explicit=text_col, candidates=TEXT_CANDIDATES, label="text_col", required=True
    )
    resolved_id = _resolve_explicit_or_candidates(
        df, explicit=id_col, candidates=ID_CANDIDATES, label="id_col", required=False
    )
    resolved_doc = _resolve_explicit_or_candidates(
        df, explicit=doc_col, candidates=DOC_CANDIDATES, label="doc_col", required=False
    )

    excluded = {resolved_text}
    if resolved_id:
        excluded.add(resolved_id)
    if resolved_doc:
        excluded.add(resolved_doc)
    if metadata_cols is None:
        resolved_metadata = tuple(col for col in df.columns if col not in excluded)
    else:
        missing = [col for col in metadata_cols if col not in df.columns]
        if missing:
            raise ValueError(f"metadata columns missing from input: {missing}")
        resolved_metadata = tuple(metadata_cols)

    result = pd.DataFrame()
    result["source_row_index"] = range(len(df))
    result["raw_text"] = df[resolved_text].fillna("").astype(str)
    if resolved_id:
        result["source_id"] = df[resolved_id].fillna("").astype(str)
    else:
        result["source_id"] = result["source_row_index"].map(lambda i: f"row_{i}")
    if resolved_doc:
        result["source_doc_id"] = df[resolved_doc].fillna("").astype(str)
    else:
        result["source_doc_id"] = result["source_id"]
    for col in resolved_metadata:
        result[col] = df[col]

    return result, InputColumns(
        text_col=resolved_text,
        id_col=resolved_id,
        doc_col=resolved_doc,
        metadata_cols=resolved_metadata,
    )
