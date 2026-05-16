from __future__ import annotations

from pathlib import Path

from fst2framegraph.framebase.iri import dbp_label_from_iri
from fst2framegraph.normalise.text import clean_text


def load_dbp_labels(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    from rdflib import Graph, RDFS

    g = Graph()
    g.parse(path)
    labels: dict[str, str] = {}
    for s, p, o in g.triples((None, RDFS.label, None)):
        s_str = str(s)
        if "/dbp/" in s_str:
            labels[s_str] = clean_text(o)
    return labels


def label_for_dbp(iri: str, labels: dict[str, str]) -> str:
    return labels.get(iri) or dbp_label_from_iri(iri)
