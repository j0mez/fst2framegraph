from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EdgeLayer(str, Enum):
    REIFIED = "reified"
    NESTED = "nested"
    DEREIFIED = "dereified"


class NodeType(str, Enum):
    DOCUMENT = "document"
    SENTENCE = "sentence"
    FRAME_INSTANCE = "frame_instance"
    FILLER = "filler"


class ColumnMap(BaseModel):
    doc_col: str
    sentence_col: str
    frame_col: str
    fe_col: str
    filler_col: str
    sentence_id_col: str | None = None
    frame_index_col: str | None = None
    target_col: str | None = None
    target_start_col: str | None = None
    target_end_col: str | None = None
    filler_start_col: str | None = None
    filler_end_col: str | None = None
    confidence_col: str | None = None
    brand_col: str | None = None
    year_col: str | None = None


class BuildConfig(BaseModel):
    input_path: Path
    out_dir: Path
    column_map: ColumnMap
    framebase_core: Path | None = None
    dbp_labels: Path | None = None
    dered_rules: Path | None = None
    min_filler_len: int = 1
    write_rdf: bool = True
    write_graphml: bool = True


class FrameBaseRule(BaseModel):
    rule_id: str
    frame_iri: str
    frame_name: str | None = None
    subject_fe_iri: str
    object_fe_iri: str
    subject_fe_name: str | None = None
    object_fe_name: str | None = None
    dbp_iri: str
    dbp_label: str | None = None
    raw_rule: str | None = None


class QCReport(BaseModel):
    input_rows: int
    unique_documents: int
    unique_sentences: int
    frame_instances: int
    frame_elements: int
    reified_edges: int
    nested_edges: int
    dereified_edges: int
    framebase_validated_frames: int = 0
    framebase_unmatched_frames: int = 0
    framebase_validated_frame_elements: int = 0
    framebase_unmatched_frame_elements: int = 0
    warnings: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)
