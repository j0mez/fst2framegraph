from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from run_pipeline import run_pipeline


class FakeFrameElement:
    def __init__(self, name: str, text: str) -> None:
        self.name = name
        self.text = text


class FakeFrame:
    def __init__(self, name: str, trigger_location: int, elements: list[FakeFrameElement]) -> None:
        self.name = name
        self.trigger_location = trigger_location
        self.frame_elements = elements


class FakeResult:
    def __init__(self, sentence: str, frames: list[FakeFrame]) -> None:
        self.sentence = sentence
        self.frames = frames


class FakeFST:
    def detect_frames(self, sentence: str) -> FakeResult:
        if "invest" in sentence.lower():
            return FakeResult(
                sentence,
                [FakeFrame("Investing", sentence.lower().find("invest"), [FakeFrameElement("Agent", "We")])],
            )
        if "protect" in sentence.lower():
            return FakeResult(
                sentence,
                [FakeFrame("Protection", sentence.lower().find("protect"), [FakeFrameElement("Agent", "We")])],
            )
        return FakeResult(sentence, [])


class NoFrameFST:
    def detect_frames(self, sentence: str) -> FakeResult:
        return FakeResult(sentence, [])


class FailIfCalledFST:
    def detect_frames(self, sentence: str) -> FakeResult:
        raise AssertionError(f"FST should not run before FrameBase preflight: {sentence}")


def write_tiny_framebase_core(path: Path) -> Path:
    path.write_text(
        """
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        <http://framebase.org/frame/Investing> rdfs:label "Investing" .
        <http://framebase.org/fe/Investing.has_Agent> rdfs:label "Agent" .
        <http://framebase.org/frame/Protection> rdfs:label "Protection" .
        <http://framebase.org/fe/Protection.has_Agent> rdfs:label "Agent" .
        """,
        encoding="utf-8",
    )
    return path


def test_run_pipeline_requires_framebase_before_calling_fst(tmp_path: Path) -> None:
    csv_path = tmp_path / "oxccal_sample.csv"
    pd.DataFrame(
        {
            "Advert ID": ["ad-1"],
            "Transcript (text and audio)": ["[ad text:] We invest in clean power."],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(RuntimeError, match="FrameBase"):
        run_pipeline(
            csv_path,
            output_root=tmp_path / "outputs",
            fst=FailIfCalledFST(),
            timestamp="missing-framebase",
        )


def test_run_pipeline_handles_oxccal_transcripts_with_markers(tmp_path: Path) -> None:
    csv_path = tmp_path / "oxccal_sample.csv"
    pd.DataFrame(
        {
            "Advert ID": ["ad-1", "ad-2", "ad-3"],
            "Transcript (text and audio)": [
                "[ad text:] We invest in clean power. [audio transcript:] music and applause",
                "[ad text:] We protect local jobs.",
                "[audio transcript:] music only",
            ],
            "Party": ["Example", "Example", "Example"],
        }
    ).to_csv(csv_path, index=False)
    framebase_core = write_tiny_framebase_core(tmp_path / "framebase_core.ttl")

    result = run_pipeline(
        csv_path,
        output_root=tmp_path / "outputs",
        fst=FakeFST(),
        timestamp="fixed",
        framebase_core=framebase_core,
    )

    output_dir = Path(result["output_dir"])
    graph = nx.read_graphml(output_dir / "frame_graph.graphml", force_multigraph=True)
    lift = pd.read_csv(output_dir / "agent_frame_lift.csv")
    reified_edges = pd.read_csv(output_dir / "reified" / "graph_edges_reified.csv")
    reified_frames = pd.read_csv(output_dir / "reified" / "frame_instances.csv")
    reified_elements = pd.read_csv(output_dir / "reified" / "frame_elements.csv")

    assert result["rows_in"] == 3
    assert result["rows_skipped_empty"] == 1
    assert result["documents"] == 2
    assert result["reified_edges"] > 0
    assert result["framebase_reified_dir"] == str(output_dir / "reified")
    assert (output_dir / "summary_report.txt").exists()
    assert (output_dir / "reified" / "graph.graphml").exists()
    assert {data["node_type"] for _, data in graph.nodes(data=True)} >= {
        "Document",
        "FrameInstance",
        "Filler",
    }
    assert not lift.empty
    assert set(lift["frame_type"]) == {"Investing", "Protection"}
    assert not reified_edges.empty
    assert {"framebase_frame_iri", "framebase_frame_validated"} <= set(reified_frames.columns)
    assert {"frame_element_iri", "frame_element_validated"} <= set(reified_elements.columns)
    assert set(reified_frames["framebase_frame_validated"]) == {True}
    assert set(reified_elements["frame_element_validated"]) == {True}


def test_run_pipeline_validates_reified_output_against_framebase_schema(tmp_path: Path) -> None:
    csv_path = tmp_path / "oxccal_sample.csv"
    pd.DataFrame(
        {
            "Advert ID": ["ad-1"],
            "Transcript (text and audio)": ["[ad text:] We invest in clean power."],
        }
    ).to_csv(csv_path, index=False)
    framebase_core = write_tiny_framebase_core(tmp_path / "framebase_core.ttl")

    result = run_pipeline(
        csv_path,
        output_root=tmp_path / "outputs",
        fst=FakeFST(),
        timestamp="schema",
        framebase_core=framebase_core,
        require_framebase=True,
    )

    output_dir = Path(result["output_dir"])
    frames = pd.read_csv(output_dir / "reified" / "frame_instances.csv")
    elements = pd.read_csv(output_dir / "reified" / "frame_elements.csv")

    assert result["framebase_validated_frames"] == 1
    assert result["framebase_validated_frame_elements"] == 1
    assert bool(frames.loc[0, "framebase_frame_validated"]) is True
    assert bool(elements.loc[0, "frame_element_validated"]) is True


def test_run_pipeline_writes_empty_framebase_outputs_for_zero_frame_documents(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "oxccal_sample.csv"
    pd.DataFrame(
        {
            "Advert ID": ["ad-1"],
            "Transcript (text and audio)": ["[ad text:] Hello."],
        }
    ).to_csv(csv_path, index=False)
    framebase_core = write_tiny_framebase_core(tmp_path / "framebase_core.ttl")

    result = run_pipeline(
        csv_path,
        output_root=tmp_path / "outputs",
        fst=NoFrameFST(),
        timestamp="zero-frames",
        framebase_core=framebase_core,
    )

    output_dir = Path(result["output_dir"])
    assert result["documents"] == 1
    assert result["reified_edges"] == 0
    assert (output_dir / "reified" / "frame_instances.csv").exists()
    assert (output_dir / "reified" / "frame_elements.csv").exists()
    assert (output_dir / "reified" / "graph_edges_reified.csv").exists()
    assert (output_dir / "reified" / "graph.graphml").exists()
