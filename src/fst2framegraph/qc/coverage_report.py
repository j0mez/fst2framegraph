from __future__ import annotations

import pandas as pd

from fst2framegraph.schema import QCReport


def make_qc_report(
    source_rows: int,
    documents: pd.DataFrame,
    sentences: pd.DataFrame,
    frame_instances: pd.DataFrame,
    frame_elements: pd.DataFrame,
    reified_edges: pd.DataFrame,
    nested_edges: pd.DataFrame,
    dereified_edges: pd.DataFrame,
    warnings: list[str] | None = None,
) -> QCReport:
    warnings = warnings or []
    frame_validated = int(frame_instances.get("framebase_frame_validated", pd.Series(dtype=bool)).sum()) if not frame_instances.empty else 0
    fe_validated = int(frame_elements.get("frame_element_validated", pd.Series(dtype=bool)).sum()) if not frame_elements.empty else 0
    return QCReport(
        input_rows=source_rows,
        unique_documents=len(documents),
        unique_sentences=len(sentences),
        frame_instances=len(frame_instances),
        frame_elements=len(frame_elements),
        reified_edges=len(reified_edges),
        nested_edges=0 if nested_edges is None or nested_edges.empty else len(nested_edges),
        dereified_edges=0 if dereified_edges is None or dereified_edges.empty else len(dereified_edges),
        framebase_validated_frames=frame_validated,
        framebase_unmatched_frames=max(len(frame_instances) - frame_validated, 0),
        framebase_validated_frame_elements=fe_validated,
        framebase_unmatched_frame_elements=max(len(frame_elements) - fe_validated, 0),
        warnings=warnings,
    )
