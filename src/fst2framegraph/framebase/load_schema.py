from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from fst2framegraph.framebase.iri import fe_iri, frame_iri
from fst2framegraph.normalise.text import clean_text


def _tail(resource: str) -> str:
    resource = resource.rstrip("/>")
    if "#" in resource:
        return resource.rsplit("#", 1)[-1]
    return resource.rsplit("/", 1)[-1]


def _frame_from_resource(resource: str) -> str | None:
    if "/frame/" in resource:
        tail = resource.rsplit("/frame/", 1)[-1]
    else:
        tail = _tail(resource)
    tail = tail.replace("+", " ")
    if tail.startswith("Microframe."):
        parts = tail.split(".")
        return parts[1] if len(parts) > 1 else None
    if tail.startswith("frame."):
        parts = tail.split(".")
        return parts[1] if len(parts) > 1 else None
    return tail.split(".", 1)[0] or None


def _normalise_fe_name(fe: str) -> str:
    fe = fe.removeprefix("has_").replace("+", " ").strip()
    if fe.islower():
        return "_".join(part.capitalize() for part in fe.split("_"))
    return fe


def _fe_from_resource(resource: str) -> tuple[str | None, str | None]:
    if "/fe/" in resource:
        tail = resource.rsplit("/fe/", 1)[-1]
    else:
        tail = _tail(resource)
    tail = tail.replace("+", " ")
    if tail.startswith("fe."):
        parts = tail.split(".")
        if len(parts) >= 3:
            return parts[1], _normalise_fe_name(parts[2])
    if "." in tail:
        frame_part, fe_part = tail.split(".", 1)
        return frame_part, _normalise_fe_name(fe_part)
    return None, _normalise_fe_name(tail) if tail else None


def _lookup_keys(frame_name: str, fe_name: str) -> list[tuple[str, str]]:
    frame_name = clean_text(frame_name)
    fe_name = clean_text(fe_name)
    return [
        (frame_name, fe_name),
        (frame_name, fe_name.lower()),
        (frame_name, fe_name.upper()),
        (frame_name, fe_name.title()),
        (frame_name, "_".join(part.capitalize() for part in fe_name.lower().split("_"))),
    ]


@dataclass
class FrameBaseSchema:
    frame_lookup: dict[str, str] = field(default_factory=dict)
    fe_lookup: dict[tuple[str, str], str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "FrameBaseSchema":
        return cls()

    @classmethod
    def from_turtle(cls, path: Path | None) -> "FrameBaseSchema":
        if path is None:
            return cls.empty()
        from rdflib import Graph, RDFS

        g = Graph()
        g.parse(path)
        schema = cls.empty()

        for s, p, o in g:
            s_str = str(s)
            if p == RDFS.label:
                schema.labels[s_str] = clean_text(o)
            if "/frame/" in s_str or _tail(s_str).startswith(("Microframe.", "frame.")):
                name = _frame_from_resource(s_str)
                if name:
                    schema.frame_lookup.setdefault(name, s_str)
            if "/fe/" in s_str or _tail(s_str).startswith("fe.") or ".has_" in _tail(s_str):
                frame_name, fe_name = _fe_from_resource(s_str)
                if frame_name and fe_name:
                    for key in _lookup_keys(frame_name, fe_name):
                        schema.fe_lookup.setdefault(key, s_str)
        return schema

    def get_frame_iri(self, frame_name: str) -> tuple[str, bool]:
        frame_name = clean_text(frame_name)
        if frame_name in self.frame_lookup:
            return self.frame_lookup[frame_name], True
        return frame_iri(frame_name), False

    def get_fe_iri(self, frame_name: str, fe_name: str) -> tuple[str, bool]:
        for key in _lookup_keys(frame_name, fe_name):
            if key in self.fe_lookup:
                return self.fe_lookup[key], True
        return fe_iri(clean_text(frame_name), clean_text(fe_name)), False
