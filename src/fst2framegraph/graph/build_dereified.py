from __future__ import annotations

from itertools import product

import pandas as pd

from fst2framegraph.framebase.rule_index import RuleIndex
from fst2framegraph.normalise.ids import short_hash


def _rows_for_rule(g: pd.DataFrame, fe_iri: str | None, fe_name: str | None) -> list:
    rows = []
    if fe_iri:
        rows.extend(g[g["frame_element_iri"].astype(str) == fe_iri].itertuples(index=False))
    if fe_name:
        lname = fe_name.lower()
        rows.extend(g[g["fe_name"].astype(str).str.lower() == lname].itertuples(index=False))
    # Preserve order while removing duplicates.
    seen = set()
    unique = []
    for row in rows:
        key = (row.frame_instance_id, row.fe_name, row.filler_id, row.filler_text)
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def build_dereified_edges(frame_elements: pd.DataFrame, rule_index: RuleIndex) -> pd.DataFrame:
    if frame_elements.empty:
        return pd.DataFrame()

    edges = []
    for frame_id, g in frame_elements.groupby("frame_instance_id"):
        frame_name = str(g["frame_name"].iloc[0])
        rules = rule_index.get(frame_name)
        if not rules:
            continue

        for rule in rules:
            subject_rows = _rows_for_rule(g, rule.subject_fe_iri, rule.subject_fe_name)
            object_rows = _rows_for_rule(g, rule.object_fe_iri, rule.object_fe_name)
            if not subject_rows or not object_rows:
                continue

            for sub, obj in product(subject_rows, object_rows):
                if sub.filler_id == obj.filler_id and sub.fe_name == obj.fe_name:
                    continue
                edge_id = "dered_" + short_hash(frame_id, sub.filler_id, rule.dbp_iri, obj.filler_id)
                edges.append(
                    {
                        "edge_id": edge_id,
                        "source": sub.filler_id,
                        "target": obj.filler_id,
                        "edge_type": "framebase_dbp",
                        "label": rule.dbp_label or rule.dbp_iri,
                        "dbp_iri": rule.dbp_iri,
                        "frame_instance_id": frame_id,
                        "frame_name": frame_name,
                        "source_fe": sub.fe_name,
                        "target_fe": obj.fe_name,
                        "source_filler_text": sub.filler_text,
                        "target_filler_text": obj.filler_text,
                        "sentence_id": sub.sentence_id,
                        "document_id": sub.document_id,
                        "rule_id": rule.rule_id,
                        "layer": "dereified",
                    }
                )
    return pd.DataFrame(edges).drop_duplicates("edge_id") if edges else pd.DataFrame()
