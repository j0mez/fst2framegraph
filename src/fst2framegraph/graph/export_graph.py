from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd


def to_networkx(nodes: pd.DataFrame, *edge_tables: pd.DataFrame) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    if not nodes.empty:
        for _, r in nodes.iterrows():
            g.add_node(r["node_id"], **{k: str(v) for k, v in r.items() if pd.notna(v)})
    for edges in edge_tables:
        if edges is None or edges.empty:
            continue
        for _, r in edges.iterrows():
            attrs = {k: str(v) for k, v in r.items() if k not in {"source", "target"} and pd.notna(v)}
            g.add_edge(r["source"], r["target"], **attrs)
    return g


def write_graphml(nodes: pd.DataFrame, out_path: Path, *edge_tables: pd.DataFrame) -> None:
    g = to_networkx(nodes, *edge_tables)
    nx.write_graphml(g, out_path)


def write_turtle(nodes: pd.DataFrame, out_path: Path, *edge_tables: pd.DataFrame) -> None:
    def fg_uri(value: object) -> str:
        return f"<https://w3id.org/fst2framegraph/{value}>"

    def iri(value: object) -> str:
        return f"<{value}>"

    def lit(value: object) -> str:
        text = str(value).replace("\\", "\\\\").replace('"', '\\"')
        return f"\"{text}\""

    lines = [
        "@prefix fg: <https://w3id.org/fst2framegraph/> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "",
    ]

    for _, n in nodes.iterrows():
        node_uri = fg_uri(n["node_id"])
        node_type = fg_uri(n.get("node_type", "node"))
        lines.append(f"{node_uri} rdf:type {node_type} .")
        if "label" in n and pd.notna(n["label"]):
            lines.append(f"{node_uri} rdfs:label {lit(n['label'])} .")

    seen_predicate_labels: set[str] = set()
    for edges in edge_tables:
        if edges is None or edges.empty:
            continue
        for _, e in edges.iterrows():
            source = fg_uri(e["source"])
            target = fg_uri(e["target"])
            if e.get("layer") == "dereified" and pd.notna(e.get("dbp_iri")):
                pred = iri(e["dbp_iri"])
            elif pd.notna(e.get("frame_element_iri")):
                pred = iri(e["frame_element_iri"])
            else:
                pred = fg_uri(e.get("edge_type", "related_to"))
            lines.append(f"{source} {pred} {target} .")
            if "label" in e and pd.notna(e["label"]) and pred not in seen_predicate_labels:
                lines.append(f"{pred} rdfs:label {lit(e['label'])} .")
                seen_predicate_labels.add(pred)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_sentence_graphs(
    sentences: pd.DataFrame,
    frame_instances: pd.DataFrame,
    frame_elements: pd.DataFrame,
    nested_edges: pd.DataFrame,
    dereified_edges: pd.DataFrame,
) -> list[dict]:
    records = []
    frames_by_sent = {sid: g.to_dict("records") for sid, g in frame_instances.groupby("sentence_id")}
    fes_by_frame = {fid: g.to_dict("records") for fid, g in frame_elements.groupby("frame_instance_id")}
    nested_by_sent = {} if nested_edges.empty else {sid: g.to_dict("records") for sid, g in nested_edges.groupby("sentence_id")}
    dered_by_sent = {} if dereified_edges.empty else {sid: g.to_dict("records") for sid, g in dereified_edges.groupby("sentence_id")}
    for _, s in sentences.iterrows():
        sid = s["sentence_id"]
        frames = []
        for fr in frames_by_sent.get(sid, []):
            fr = dict(fr)
            fr["elements"] = fes_by_frame.get(fr["frame_instance_id"], [])
            frames.append(fr)
        records.append(
            {
                "document_id": s["document_id"],
                "sentence_id": sid,
                "sentence": s["sentence"],
                "frames": frames,
                "nested_edges": nested_by_sent.get(sid, []),
                "dereified_edges": dered_by_sent.get(sid, []),
            }
        )
    return records
