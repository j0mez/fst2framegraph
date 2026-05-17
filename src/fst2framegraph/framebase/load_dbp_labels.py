from __future__ import annotations

import gzip
import re
from pathlib import Path

from fst2framegraph.framebase.iri import dbp_label_from_iri
from fst2framegraph.normalise.text import clean_text


LABEL_RE = re.compile(r"(<[^>]+>)\s+rdfs:label\s+\"([^\"]+)\"")


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def load_dbp_labels(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    labels: dict[str, str] = {}
    with _open_text(path) as fh:
        for line in fh:
            match = LABEL_RE.search(line)
            if not match:
                continue
            iri = match.group(1)[1:-1]
            if "/dbp/" in iri:
                labels[iri] = clean_text(match.group(2))
    return labels


def label_for_dbp(iri: str, labels: dict[str, str]) -> str:
    return labels.get(iri) or dbp_label_from_iri(iri)
