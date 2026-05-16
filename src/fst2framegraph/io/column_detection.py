from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fst2framegraph.schema import ColumnMap


@dataclass(frozen=True)
class ColumnCandidates:
    doc: tuple[str, ...] = ("doc_id", "document_id", "Unique ID", "ad_id")
    sentence: tuple[str, ...] = ("sentence", "original_sentence", "parse_sentence", "sentence_text")
    sentence_id: tuple[str, ...] = ("sentence_id", "unique_chunk_id", "chunk_id")
    frame: tuple[str, ...] = ("frame_name", "frame", "Frame")
    fe: tuple[str, ...] = ("element_name", "fe_name", "frame_element_name", "Frame Element")
    filler: tuple[str, ...] = ("element_filler", "element_text", "filler", "frame_element_filler")
    frame_index: tuple[str, ...] = ("frame_index", "frame_id", "frame_no")
    target: tuple[str, ...] = ("target_text", "target", "trigger", "lexical_unit", "lu")
    target_start: tuple[str, ...] = ("target_start", "target_start_col", "lu_start", "trigger_start")
    target_end: tuple[str, ...] = ("target_end", "target_end_col", "lu_end", "trigger_end")
    filler_start: tuple[str, ...] = ("filler_start", "element_start", "fe_start", "span_start")
    filler_end: tuple[str, ...] = ("filler_end", "element_end", "fe_end", "span_end")
    confidence: tuple[str, ...] = ("confidence", "score", "probability")
    brand: tuple[str, ...] = ("Brand", "brand", "Brand_x")
    year: tuple[str, ...] = ("Years", "year", "Years_x")


def _find(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    exact = {c: c for c in columns}
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in exact:
            return exact[cand]
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def detect_columns(df: pd.DataFrame) -> ColumnMap:
    cols = list(df.columns)
    c = ColumnCandidates()

    missing = []
    doc_col = _find(cols, c.doc)
    sentence_col = _find(cols, c.sentence)
    frame_col = _find(cols, c.frame)
    fe_col = _find(cols, c.fe)
    filler_col = _find(cols, c.filler)
    for name, value in {
        "doc_col": doc_col,
        "sentence_col": sentence_col,
        "frame_col": frame_col,
        "fe_col": fe_col,
        "filler_col": filler_col,
    }.items():
        if value is None:
            missing.append(name)
    if missing:
        raise ValueError(
            "Could not detect required columns: "
            + ", ".join(missing)
            + f". Available columns: {cols}"
        )

    return ColumnMap(
        doc_col=doc_col,
        sentence_col=sentence_col,
        sentence_id_col=_find(cols, c.sentence_id),
        frame_col=frame_col,
        fe_col=fe_col,
        filler_col=filler_col,
        frame_index_col=_find(cols, c.frame_index),
        target_col=_find(cols, c.target),
        target_start_col=_find(cols, c.target_start),
        target_end_col=_find(cols, c.target_end),
        filler_start_col=_find(cols, c.filler_start),
        filler_end_col=_find(cols, c.filler_end),
        confidence_col=_find(cols, c.confidence),
        brand_col=_find(cols, c.brand),
        year_col=_find(cols, c.year),
    )
