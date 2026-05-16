from __future__ import annotations

import pandas as pd

from fst2framegraph.framebase.load_schema import FrameBaseSchema
from fst2framegraph.normalise.fillers import keep_filler
from fst2framegraph.normalise.ids import (
    make_document_id,
    make_filler_id,
    make_frame_instance_id,
    make_sentence_id,
)
from fst2framegraph.normalise.text import clean_text
from fst2framegraph.schema import ColumnMap


def _cell(row: pd.Series, col: str | None, default: str = "") -> str:
    if not col:
        return default
    return clean_text(row.get(col))


def _int_cell(row: pd.Series, col: str | None) -> int | None:
    if not col:
        return None
    value = row.get(col)
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except Exception:
        return None


def build_reified_tables(
    df: pd.DataFrame,
    cmap: ColumnMap,
    schema: FrameBaseSchema,
    min_filler_len: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, r in df.iterrows():
        filler = _cell(r, cmap.filler_col)
        if not keep_filler(filler, min_filler_len):
            continue
        doc_raw = _cell(r, cmap.doc_col)
        sentence = _cell(r, cmap.sentence_col)
        sentence_raw_id = _cell(r, cmap.sentence_id_col, default="") or None
        frame_name = _cell(r, cmap.frame_col)
        fe_name = _cell(r, cmap.fe_col)
        frame_index = _cell(r, cmap.frame_index_col, default="") or None
        target_text = _cell(r, cmap.target_col, default="")

        document_id = make_document_id(doc_raw)
        sentence_id = make_sentence_id(doc_raw, sentence, sentence_raw_id)
        frame_instance_id = make_frame_instance_id(doc_raw, sentence_id, frame_name, frame_index)
        filler_id = make_filler_id(filler)
        frame_iri, frame_valid = schema.get_frame_iri(frame_name)
        fe_iri, fe_valid = schema.get_fe_iri(frame_name, fe_name)

        rows.append(
            {
                "document_id": document_id,
                "source_document_id": doc_raw,
                "sentence_id": sentence_id,
                "sentence": sentence,
                "frame_instance_id": frame_instance_id,
                "frame_index": frame_index or "",
                "frame_name": frame_name,
                "framebase_frame_iri": frame_iri,
                "framebase_frame_validated": bool(frame_valid),
                "target_text": target_text,
                "target_start": _int_cell(r, cmap.target_start_col),
                "target_end": _int_cell(r, cmap.target_end_col),
                "fe_name": fe_name,
                "frame_element_iri": fe_iri,
                "frame_element_validated": bool(fe_valid),
                "filler_id": filler_id,
                "filler_text": filler,
                "filler_start": _int_cell(r, cmap.filler_start_col),
                "filler_end": _int_cell(r, cmap.filler_end_col),
                "confidence": _cell(r, cmap.confidence_col, default=""),
                "brand": _cell(r, cmap.brand_col, default=""),
                "year": _cell(r, cmap.year_col, default=""),
            }
        )

    long = pd.DataFrame(rows)
    if long.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty, empty

    documents = (
        long[["document_id", "source_document_id", "brand", "year"]]
        .drop_duplicates()
        .sort_values("document_id")
        .reset_index(drop=True)
    )
    sentences = (
        long[["sentence_id", "document_id", "source_document_id", "sentence"]]
        .drop_duplicates()
        .sort_values(["document_id", "sentence_id"])
        .reset_index(drop=True)
    )
    frame_instances = (
        long[
            [
                "frame_instance_id",
                "sentence_id",
                "document_id",
                "frame_name",
                "framebase_frame_iri",
                "framebase_frame_validated",
                "target_text",
                "target_start",
                "target_end",
                "frame_index",
            ]
        ]
        .drop_duplicates()
        .sort_values(["document_id", "sentence_id", "frame_instance_id"])
        .reset_index(drop=True)
    )
    frame_elements = (
        long[
            [
                "frame_instance_id",
                "sentence_id",
                "document_id",
                "frame_name",
                "fe_name",
                "frame_element_iri",
                "frame_element_validated",
                "filler_id",
                "filler_text",
                "filler_start",
                "filler_end",
                "confidence",
            ]
        ]
        .drop_duplicates()
        .sort_values(["frame_instance_id", "fe_name", "filler_text"])
        .reset_index(drop=True)
    )

    nodes = []
    for _, r in documents.iterrows():
        nodes.append({"node_id": r.document_id, "node_type": "document", "label": r.source_document_id})
    for _, r in sentences.iterrows():
        nodes.append({"node_id": r.sentence_id, "node_type": "sentence", "label": r.sentence[:120]})
    for _, r in frame_instances.iterrows():
        nodes.append({
            "node_id": r.frame_instance_id,
            "node_type": "frame_instance",
            "label": r.frame_name,
            "frame_name": r.frame_name,
            "framebase_frame_iri": r.framebase_frame_iri,
        })
    for _, r in frame_elements[["filler_id", "filler_text"]].drop_duplicates().iterrows():
        nodes.append({"node_id": r.filler_id, "node_type": "filler", "label": r.filler_text})
    nodes_df = pd.DataFrame(nodes).drop_duplicates("node_id")

    edges = []
    for _, r in sentences.iterrows():
        edges.append({
            "source": r.document_id,
            "target": r.sentence_id,
            "edge_type": "has_sentence",
            "label": "has_sentence",
            "layer": "reified",
        })
    for _, r in frame_instances.iterrows():
        edges.append({
            "source": r.sentence_id,
            "target": r.frame_instance_id,
            "edge_type": "evokes_frame",
            "label": "evokes_frame",
            "frame_name": r.frame_name,
            "framebase_frame_iri": r.framebase_frame_iri,
            "layer": "reified",
        })
    for _, r in frame_elements.iterrows():
        edges.append({
            "source": r.frame_instance_id,
            "target": r.filler_id,
            "edge_type": "frame_element",
            "label": r.fe_name,
            "frame_name": r.frame_name,
            "frame_element_iri": r.frame_element_iri,
            "filler_text": r.filler_text,
            "layer": "reified",
            "sentence_id": r.sentence_id,
            "document_id": r.document_id,
        })
    edges_df = pd.DataFrame(edges)

    return documents, sentences, frame_instances, frame_elements, nodes_df, edges_df
