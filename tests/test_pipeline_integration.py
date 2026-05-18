from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

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

    result = run_pipeline(csv_path, output_root=tmp_path / "outputs", fst=FakeFST(), timestamp="fixed")

    output_dir = Path(result["output_dir"])
    graph = nx.read_graphml(output_dir / "frame_graph.graphml", force_multigraph=True)
    lift = pd.read_csv(output_dir / "agent_frame_lift.csv")

    assert result["rows_in"] == 3
    assert result["rows_skipped_empty"] == 1
    assert result["documents"] == 2
    assert (output_dir / "summary_report.txt").exists()
    assert {data["node_type"] for _, data in graph.nodes(data=True)} >= {
        "Document",
        "FrameInstance",
        "Filler",
    }
    assert not lift.empty
    assert set(lift["frame_type"]) == {"Investing", "Protection"}
