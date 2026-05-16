from __future__ import annotations

import pandas as pd

from fst2framegraph.normalise.text import normalise_for_match


def _has_int(value: object) -> bool:
    try:
        return not pd.isna(value) and int(value) >= 0
    except Exception:
        return False


def _span_contains(parent_start: object, parent_end: object, child_start: object, child_end: object) -> bool:
    if not all(_has_int(v) for v in [parent_start, parent_end, child_start, child_end]):
        return False
    ps, pe, cs, ce = int(parent_start), int(parent_end), int(child_start), int(child_end)
    return ps <= cs and ce <= pe and pe > ps and ce > cs


def build_nested_edges(frame_instances: pd.DataFrame, frame_elements: pd.DataFrame) -> pd.DataFrame:
    """Detect frame-inside-filler nesting.

    Preferred method: character span containment, when the parser output supplies offsets.
    Fallback method: exact normalised target-text containment inside the FE filler text.
    The original FE filler edge is preserved; this adds an event-to-event edge.
    """
    if frame_instances.empty or frame_elements.empty:
        return pd.DataFrame()

    edges = []
    frames_by_sentence = {
        sid: g.to_dict("records") for sid, g in frame_instances.groupby("sentence_id")
    }
    for _, fe in frame_elements.iterrows():
        filler_norm = normalise_for_match(fe.get("filler_text", ""))
        for child in frames_by_sentence.get(fe["sentence_id"], []):
            child_id = child["frame_instance_id"]
            if child_id == fe["frame_instance_id"]:
                continue

            method = None
            confidence = ""
            if _span_contains(
                fe.get("filler_start"),
                fe.get("filler_end"),
                child.get("target_start"),
                child.get("target_end"),
            ):
                method = "span_containment"
                confidence = "high"
            else:
                target = normalise_for_match(child.get("target_text", ""))
                if target and filler_norm and target in filler_norm:
                    method = "target_text_containment"
                    confidence = "medium"

            if method is None:
                continue

            edges.append(
                {
                    "source": fe["frame_instance_id"],
                    "target": child_id,
                    "edge_type": "nested_frame",
                    "label": fe["fe_name"],
                    "parent_fe_name": fe["fe_name"],
                    "parent_frame_name": fe["frame_name"],
                    "parent_filler_text": fe["filler_text"],
                    "child_frame_name": child["frame_name"],
                    "child_target_text": child.get("target_text", ""),
                    "sentence_id": fe["sentence_id"],
                    "document_id": fe["document_id"],
                    "nesting_method": method,
                    "nesting_confidence": confidence,
                    "layer": "nested",
                }
            )
    return pd.DataFrame(edges).drop_duplicates() if edges else pd.DataFrame()
