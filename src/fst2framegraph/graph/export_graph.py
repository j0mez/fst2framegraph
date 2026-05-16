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
    from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef

    FG = Namespace("https://w3id.org/fst2framegraph/")
    FBFRAME = Namespace("http://framebase.org/frame/")
    FBFE = Namespace("http://framebase.org/fe/")
    FBDBP = Namespace("http://framebase.org/dbp/")

    g = Graph()
    g.bind("fg", FG)
    g.bind("fbframe", FBFRAME)
    g.bind("fbfe", FBFE)
    g.bind("fbdbp", FBDBP)

    for _, n in nodes.iterrows():
        uri = FG[str(n["node_id"])]
        g.add((uri, RDF.type, FG[str(n.get("node_type", "node"))]))
        if "label" in n and pd.notna(n["label"]):
            g.add((uri, RDFS.label, Literal(str(n["label"]))))

    for edges in edge_tables:
        if edges is None or edges.empty:
            continue
        for _, e in edges.iterrows():
            source = FG[str(e["source"])]
            target = FG[str(e["target"])]
            if e.get("layer") == "dereified" and pd.notna(e.get("dbp_iri")):
                pred = URIRef(str(e["dbp_iri"]))
            elif pd.notna(e.get("frame_element_iri")):
                pred = URIRef(str(e["frame_element_iri"]))
            else:
                pred = FG[str(e.get("edge_type", "related_to"))]
            g.add((source, pred, target))
            if "label" in e and pd.notna(e["label"]):
                g.add((pred, RDFS.label, Literal(str(e["label"]))))

    g.serialize(out_path, format="turtle")


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
