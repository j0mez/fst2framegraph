from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Iterator, Literal, NamedTuple


RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDF_FIRST = "http://www.w3.org/1999/02/22-rdf-syntax-ns#first"
RDF_REST = "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest"
RDF_NIL = "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
XSD_STRING = "http://www.w3.org/2001/XMLSchema#string"
FRAMEBASE_HAS_LEXICAL_FORM = "http://framebase.org/meta/hasLexicalForm"

IRI_OR_BNODE = r"(?:<[^>]+>|_:[^\s]+)"
LITERAL = r'"(?:[^"\\]|\\.)*"(?:@[A-Za-z0-9-]+|\^\^<[^>]+>)?'
TRIPLE_RE = re.compile(
    rf"^\s*(?P<subj>{IRI_OR_BNODE})\s+(?P<pred>{IRI_OR_BNODE})\s+(?P<obj>{IRI_OR_BNODE}|{LITERAL})\s*\.\s*$"
)
LITERAL_VALUE_RE = re.compile(r'^"(?P<value>(?:[^"\\]|\\.)*)"')


class Triple(NamedTuple):
    subject: str
    predicate: str
    object: str
    object_kind: Literal["iri", "bnode", "literal"]


def open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def strip_term(token: str) -> tuple[str, Literal["iri", "bnode", "literal"]]:
    token = token.strip()
    if token.startswith("<") and token.endswith(">"):
        return token[1:-1], "iri"
    if token.startswith("_:"):
        return token, "bnode"
    match = LITERAL_VALUE_RE.match(token)
    if match:
        value = match.group("value")
        value = value.replace('\\"', '"').replace("\\\\", "\\")
        return value, "literal"
    return token, "literal"


def parse_triple_line(line: str) -> Triple | None:
    match = TRIPLE_RE.match(line.strip())
    if not match:
        return None
    subject, subject_kind = strip_term(match.group("subj"))
    predicate, predicate_kind = strip_term(match.group("pred"))
    obj, object_kind = strip_term(match.group("obj"))
    if subject_kind not in {"iri", "bnode"} or predicate_kind != "iri":
        return None
    return Triple(subject=subject, predicate=predicate, object=obj, object_kind=object_kind)


def iter_triples(path: Path) -> Iterator[Triple]:
    with open_text(path) as fh:
        for raw_line in fh:
            triple = parse_triple_line(raw_line)
            if triple is not None:
                yield triple
