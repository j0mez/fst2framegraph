from __future__ import annotations

import gzip
import os
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


class CapabilityFST:
    def detect_frames(self, sentence: str) -> FakeResult:
        return FakeResult(
            sentence,
            [
                FakeFrame(
                    "Capability",
                    sentence.lower().find("can"),
                    [
                        FakeFrameElement("Entity", "Technology"),
                        FakeFrameElement("Event", "reduce emissions"),
                    ],
                )
            ],
        )


class FailIfCalledFST:
    def detect_frames(self, sentence: str) -> FakeResult:
        raise AssertionError(f"FST should not run before FrameBase preflight: {sentence}")


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def write_gzip_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(text.strip() + "\n")
    return path


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


def write_tiny_framebase_dir_with_spin_rule(path: Path) -> Path:
    write_text(
        path / "FrameBase_schema_core.ttl",
        """
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        <http://framebase.org/frame/Capability.can.verb> rdfs:label "Capability.can.verb" .
        <http://framebase.org/fe/Capability.has_entity> rdfs:label "Entity" .
        <http://framebase.org/fe/Capability.has_event> rdfs:label "Event" .
        """,
    )
    write_text(
        path / "FrameBase_schema_dbps.ttl",
        """
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        <http://framebase.org/dbp/Capability.hasCapabilityForEvent> rdfs:label "hasCapabilityForEvent" .
        """,
    )
    write_gzip_text(
        path / "dereificationRulesSpinFormat.ttl.gz",
        """
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix sp: <http://spinrdf.org/sp#> .

        <http://framebase.org/rule/capability-can>
            rdf:type sp:Construct ;
            sp:templates (
                [
                    sp:subject [ sp:varName "S" ] ;
                    sp:predicate <http://framebase.org/dbp/Capability.hasCapabilityForEvent> ;
                    sp:object [ sp:varName "O" ]
                ]
            ) ;
            sp:where (
                [
                    sp:subject [ sp:varName "R" ] ;
                    sp:predicate rdf:type ;
                    sp:object <http://framebase.org/frame/Capability.can.verb>
                ]
                [
                    sp:subject [ sp:varName "R" ] ;
                    sp:predicate <http://framebase.org/fe/Capability.has_entity> ;
                    sp:object [ sp:varName "S" ]
                ]
                [
                    sp:subject [ sp:varName "R" ] ;
                    sp:predicate <http://framebase.org/fe/Capability.has_event> ;
                    sp:object [ sp:varName "O" ]
                ]
            ) .
        """,
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


def test_run_pipeline_sets_tensorflow_disable_environment_before_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("USE_TF", "TRANSFORMERS_NO_TF", "USE_FLAX", "TOKENIZERS_PARALLELISM"):
        monkeypatch.delenv(name, raising=False)
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

    assert os.environ["USE_TF"] == "0"
    assert os.environ["TRANSFORMERS_NO_TF"] == "1"
    assert os.environ["USE_FLAX"] == "0"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


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


def test_run_pipeline_emits_direct_edges_from_framebase_spin_rules(tmp_path: Path) -> None:
    csv_path = tmp_path / "oxccal_sample.csv"
    pd.DataFrame(
        {
            "Advert ID": ["ad-1"],
            "Transcript (text and audio)": ["[ad text:] Technology can reduce emissions."],
        }
    ).to_csv(csv_path, index=False)
    framebase_dir = write_tiny_framebase_dir_with_spin_rule(tmp_path / "framebase")

    result = run_pipeline(
        csv_path,
        output_root=tmp_path / "outputs",
        fst=CapabilityFST(),
        timestamp="dereified",
        framebase_dir=framebase_dir,
    )

    output_dir = Path(result["output_dir"]) / "reified"
    direct_edges = pd.read_csv(output_dir / "direct_edges.csv")
    diagnostics = pd.read_csv(output_dir / "dereification_diagnostics.csv")
    summary = pd.read_json(output_dir / "summary.json", typ="series")

    assert result["dereified_edges"] == 1
    assert summary["official_framebase_reder_edges"] == 1
    assert summary["dereification_rules_loaded"] == 1
    assert diagnostics.empty
    assert len(direct_edges) == 1
    edge = direct_edges.iloc[0]
    assert edge["subject_filler"] == "Technology"
    assert edge["object_filler"] == "reduce emissions"
    assert edge["predicate_iri"] == "http://framebase.org/dbp/Capability.hasCapabilityForEvent"
    assert edge["match_tier"] == "frame_target_fe_unique"


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
