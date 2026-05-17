from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx
import pandas as pd

from fst2framegraph import (
    FrameGraphBuilder,
    from_frame_elements_long_csv,
    from_fst_output,
    from_legacy_pickle,
)


class FakeFrameElement:
    def __init__(self, name: str, text: str) -> None:
        self.name = name
        self.text = text


class FakeFrame:
    def __init__(self) -> None:
        self.name = "Investing"
        self.trigger_location = 3
        self.frame_elements = [FakeFrameElement("Agent", "We")]


class FakeResult:
    def __init__(self) -> None:
        self.sentence = "We invest."
        self.frames = [FakeFrame()]


def sample_documents() -> list[dict]:
    return [
        {
            "doc_id": "doc1",
            "text": "We invest. We report.",
            "metadata": {"company": "ExampleCo", "year": 2026},
            "frames": [
                {
                    "frame_type": "Investing",
                    "trigger": "invest",
                    "sent_idx": 0,
                    "frame_elements": [
                        {"role": "Agent", "text": "We"},
                        {"role": "Goal", "text": "clean energy"},
                    ],
                },
                {
                    "frame_type": "Reporting",
                    "trigger": "report",
                    "sent_idx": 1,
                    "frame_elements": [
                        {"role": "Agent", "text": "we."},
                        {"role": "Topic", "text": "progress"},
                    ],
                },
            ],
        }
    ]


def test_frame_graph_builder_schema_and_filler_merging(tmp_path: Path) -> None:
    builder = FrameGraphBuilder()

    graph = builder.build_graph(sample_documents())

    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.nodes["doc:doc1"]["node_type"] == "Document"
    assert graph.nodes["doc:doc1"]["company"] == "ExampleCo"
    assert graph.nodes["sent:doc1:0"]["node_type"] == "Sentence"
    assert graph.nodes["frame:doc1:0"]["frame_type"] == "Investing"
    assert graph.nodes["frame:doc1:1"]["sent_idx"] == 1

    we_id = "filler:" + FrameGraphBuilder.filler_hash("we")
    assert graph.nodes[we_id]["node_type"] == "Filler"
    assert graph.nodes[we_id]["text"] == "we"
    assert len([node for node, data in graph.nodes(data=True) if data["node_type"] == "Filler"]) == 3

    assert graph.has_edge("doc:doc1", "sent:doc1:0", key="HAS_FRAME")
    assert graph.has_edge("sent:doc1:0", "frame:doc1:0", key="HAS_FRAME")
    assert graph.has_edge("frame:doc1:0", we_id, key="Agent")
    assert graph["frame:doc1:0"][we_id]["Agent"]["role"] == "Agent"

    path = tmp_path / "graph.graphml"
    builder.save_graph(graph, path)
    loaded = builder.load_graph(path)
    assert set(loaded.nodes) == set(graph.nodes)
    assert set(loaded.edges(keys=True)) == set(graph.edges(keys=True))


def test_frame_graph_builder_without_sentence_nodes_and_custom_normalizer() -> None:
    builder = FrameGraphBuilder(
        normalize_filler=lambda value: str(value).strip(),
        include_sentence_nodes=False,
    )

    graph = builder.build_graph(sample_documents())

    assert not any(data["node_type"] == "Sentence" for _, data in graph.nodes(data=True))
    assert graph.has_edge("doc:doc1", "frame:doc1:0", key="HAS_FRAME")
    assert "filler:" + FrameGraphBuilder.filler_hash("We") in graph.nodes
    assert "filler:" + FrameGraphBuilder.filler_hash("we") not in graph.nodes


def test_from_frame_elements_long_csv_adapter(tmp_path: Path) -> None:
    csv_path = tmp_path / "frame_elements_long.csv"
    pd.DataFrame(
        {
            "doc_id": ["d1", "d1", "d1"],
            "sentence_id": ["s1", "s1", "s2"],
            "sentence": ["We invest.", "We invest.", "We report."],
            "frame_index": [0, 0, 0],
            "frame_name": ["Investing", "Investing", "Reporting"],
            "target_text": ["invest", "invest", "report"],
            "element_name": ["Agent", "Goal", "Agent"],
            "element_filler": ["We", "clean energy", "We"],
        }
    ).to_csv(csv_path, index=False)

    documents = from_frame_elements_long_csv(csv_path)

    assert documents == [
        {
            "doc_id": "d1",
            "text": "We invest.\nWe report.",
            "metadata": {},
            "frames": [
                {
                    "frame_type": "Investing",
                    "trigger": "invest",
                    "sent_idx": 0,
                    "frame_elements": [
                        {"role": "Agent", "text": "We"},
                        {"role": "Goal", "text": "clean energy"},
                    ],
                },
                {
                    "frame_type": "Reporting",
                    "trigger": "report",
                    "sent_idx": 1,
                    "frame_elements": [{"role": "Agent", "text": "We"}],
                },
            ],
        }
    ]


def test_from_fst_output_directory_and_legacy_pickle_adapters(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    csv_path = run_dir / "frame_elements_long.csv"
    pd.DataFrame(
        {
            "doc_id": ["d1"],
            "sentence_id": ["s1"],
            "sentence": ["We invest."],
            "frame_index": [0],
            "frame_name": ["Investing"],
            "target_text": ["invest"],
            "element_name": ["Agent"],
            "element_filler": ["We"],
        }
    ).to_csv(csv_path, index=False)

    assert from_fst_output(run_dir)[0]["frames"][0]["frame_type"] == "Investing"

    pickle_path = tmp_path / "legacy.pkl"
    pickle_path.write_bytes(
        pickle.dumps(
            {
                "unique_chunk_ids": ["chunk1"],
                "sentences": ["We invest."],
                "errors": [None],
                "raw_results": [FakeResult()],
            }
        )
    )

    try:
        from_legacy_pickle(pickle_path)
    except ValueError as exc:
        assert "allow_pickle=True" in str(exc)
    else:
        raise AssertionError("from_legacy_pickle should require explicit trusted loading")

    docs = from_legacy_pickle(pickle_path, allow_pickle=True)
    assert docs[0]["doc_id"] == "chunk1"
    assert docs[0]["frames"][0]["frame_elements"] == [{"role": "Agent", "text": "We"}]


def test_from_fst_output_jsonl_merges_sentence_records_by_doc_id(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "fst_clean.jsonl"
    records = [
        {
            "doc_id": "d1",
            "sentence_id": "s1",
            "sentence": "We invest.",
            "frames": [
                {
                    "frame_type": "Investing",
                    "trigger": "invest",
                    "frame_elements": [{"role": "Agent", "text": "We"}],
                }
            ],
        },
        {
            "doc_id": "d1",
            "sentence_id": "s2",
            "sentence": "We report.",
            "frames": [
                {
                    "frame_type": "Reporting",
                    "trigger": "report",
                    "frame_elements": [{"role": "Agent", "text": "We"}],
                }
            ],
        },
    ]
    jsonl_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    documents = from_fst_output(jsonl_path)
    graph = FrameGraphBuilder().build_graph(documents)

    assert len(documents) == 1
    assert documents[0]["text"] == "We invest.\nWe report."
    assert [frame["sent_idx"] for frame in documents[0]["frames"]] == [0, 1]
    assert "frame:d1:0" in graph
    assert "frame:d1:1" in graph
