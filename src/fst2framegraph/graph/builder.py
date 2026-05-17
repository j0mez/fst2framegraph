from __future__ import annotations

import pickle
import re
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import networkx as nx


_WS = re.compile(r"\s+")
_TRAILING_PUNCT = re.compile(r"([.,;:!?])$")


class FrameGraphBuilder:
    """Build the stable semantic event graph from parsed frame-semantic documents.

    The public graph schema is intentionally small and generic:

    - Document nodes: ``doc:{doc_id}``
    - Sentence nodes: ``sent:{doc_id}:{sent_idx}`` when enabled
    - FrameInstance nodes: ``frame:{doc_id}:{frame_index_within_doc}``
    - Filler nodes: ``filler:{sha256(normalized_text)[:12]}``

    Role edges always go from a FrameInstance node to a globally merged Filler
    node. The edge key is the role string, so a ``MultiDiGraph`` can preserve
    multiple role relationships between the same frame and filler.
    """

    def __init__(
        self,
        normalize_filler: Callable[[object], str] | None = None,
        include_sentence_nodes: bool = True,
    ) -> None:
        self.normalize_filler = normalize_filler or self.default_normalize
        self.include_sentence_nodes = include_sentence_nodes

    @staticmethod
    def default_normalize(text: object) -> str:
        """Normalize filler text for global filler-node merging."""
        value = "" if text is None else str(text)
        value = value.strip().lower()
        value = _WS.sub(" ", value)
        value = _TRAILING_PUNCT.sub("", value)
        return value.strip()

    @staticmethod
    def filler_hash(normalized_text: str) -> str:
        import hashlib

        return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()[:12]

    def _filler_id(self, normalized_text: str) -> str:
        return f"filler:{self.filler_hash(normalized_text)}"

    def build_graph(self, documents: list[dict]) -> nx.MultiDiGraph:
        """Build a semantic event graph from parsed frame-semantic documents.

        ``documents`` is a list of dictionaries with ``doc_id``, optional
        ``text``/``metadata``, and a ``frames`` list. Frames should contain
        ``frame_type`` or ``frame_name``, ``trigger`` or ``target_text``, and
        ``frame_elements`` entries with ``role``/``text`` pairs. Unknown extra
        fields are ignored by the graph constructor rather than hard-coded into
        a domain-specific schema.
        """
        graph = nx.MultiDiGraph()
        filler_registry: dict[str, str] = {}

        for document in documents:
            doc_id = str(document.get("doc_id") or "")
            if not doc_id:
                raise ValueError("Every document must include a non-empty doc_id.")
            doc_node = f"doc:{doc_id}"
            metadata = document.get("metadata") or {}
            if not isinstance(metadata, Mapping):
                raise ValueError(f"Document metadata must be a mapping for doc_id={doc_id!r}.")
            doc_attrs = {str(key): value for key, value in metadata.items()}
            doc_attrs["node_type"] = "Document"
            doc_attrs["doc_id"] = doc_id
            if document.get("text") not in (None, ""):
                doc_attrs["text"] = str(document.get("text"))
            graph.add_node(doc_node, **doc_attrs)

            frame_counter = 0
            seen_sentences: set[int] = set()
            for frame in document.get("frames") or []:
                if not isinstance(frame, Mapping):
                    continue
                sent_idx = self._frame_sent_idx(frame)
                if self.include_sentence_nodes and sent_idx not in seen_sentences:
                    sent_node = f"sent:{doc_id}:{sent_idx}"
                    graph.add_node(
                        sent_node,
                        node_type="Sentence",
                        doc_id=doc_id,
                        sent_idx=sent_idx,
                    )
                    graph.add_edge(
                        doc_node,
                        sent_node,
                        key="HAS_FRAME",
                        edge_type="HAS_FRAME",
                    )
                    seen_sentences.add(sent_idx)

                frame_node = f"frame:{doc_id}:{frame_counter}"
                frame_counter += 1
                frame_type = str(frame.get("frame_type") or frame.get("frame_name") or "")
                trigger = str(frame.get("trigger") or frame.get("target_text") or "")
                graph.add_node(
                    frame_node,
                    node_type="FrameInstance",
                    frame_type=frame_type,
                    trigger=trigger,
                    doc_id=doc_id,
                    sent_idx=sent_idx,
                )
                if self.include_sentence_nodes:
                    source = f"sent:{doc_id}:{sent_idx}"
                else:
                    source = doc_node
                graph.add_edge(
                    source,
                    frame_node,
                    key="HAS_FRAME",
                    edge_type="HAS_FRAME",
                )

                for element in frame.get("frame_elements") or []:
                    if not isinstance(element, Mapping):
                        continue
                    role = str(element.get("role") or element.get("element_name") or "")
                    raw_text = element.get("text", element.get("element_filler", ""))
                    normalized = self.normalize_filler(raw_text)
                    if not role or not normalized:
                        continue
                    filler_node = filler_registry.get(normalized)
                    if filler_node is None:
                        filler_node = self._filler_id(normalized)
                        filler_registry[normalized] = filler_node
                        graph.add_node(filler_node, node_type="Filler", text=normalized)
                    graph.add_edge(
                        frame_node,
                        filler_node,
                        key=role,
                        edge_type=role,
                        role=role,
                    )
        return graph

    @staticmethod
    def _frame_sent_idx(frame: Mapping[str, Any]) -> int:
        for key in ("sent_idx", "sentence_index", "sentence_idx"):
            value = frame.get(key)
            if value is not None and value != "":
                try:
                    return int(value)
                except Exception:
                    return 0
        return 0

    def save_graph(self, graph: nx.MultiDiGraph, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        if suffix == ".graphml":
            nx.write_graphml(graph, path)
            return
        if suffix in {".pkl", ".pickle", ".gpickle"}:
            with path.open("wb") as fh:
                pickle.dump(graph, fh, protocol=pickle.HIGHEST_PROTOCOL)
            return
        raise ValueError("Unsupported graph extension. Use .graphml, .pkl, .pickle, or .gpickle.")

    def load_graph(self, path: str | Path) -> nx.MultiDiGraph:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".graphml":
            graph = nx.read_graphml(path, force_multigraph=True)
            if not isinstance(graph, nx.MultiDiGraph):
                graph = nx.MultiDiGraph(graph)
            return graph
        if suffix in {".pkl", ".pickle", ".gpickle"}:
            with path.open("rb") as fh:
                graph = pickle.load(fh)
            if not isinstance(graph, nx.MultiDiGraph):
                raise ValueError(f"Pickle did not contain a networkx.MultiDiGraph: {type(graph)!r}")
            return graph
        raise ValueError("Unsupported graph extension. Use .graphml, .pkl, .pickle, or .gpickle.")


def iter_frame_elements(frame: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for element in frame.get("frame_elements") or []:
        if isinstance(element, Mapping):
            yield element
