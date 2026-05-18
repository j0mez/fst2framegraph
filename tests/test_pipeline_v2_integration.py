from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fst2framegraph.pipeline_v2.orchestrator import run_fst2graph


@dataclass
class FakeFE:
    name: str
    text: str


@dataclass
class FakeFrame:
    name: str
    trigger_location: int
    frame_elements: list[FakeFE]


@dataclass
class FakeResult:
    sentence: str
    frames: list[FakeFrame]


class FakeFST:
    def detect_frames(self, sentence: str) -> FakeResult:
        return FakeResult(
            sentence=sentence,
            frames=[
                FakeFrame(
                    name="Using",
                    trigger_location=0,
                    frame_elements=[
                        FakeFE(name="Agent", text="companies"),
                        FakeFE(name="Purpose", text="reduce emissions"),
                    ],
                )
            ],
        )


class EmptyFST:
    def detect_frames(self, sentence: str) -> FakeResult:
        return FakeResult(sentence=sentence, frames=[])


def test_pipeline_v2_end_to_end_outputs_are_created(tmp_path: Path) -> None:
    csv_path = tmp_path / "oxccal_sample.csv"
    pd.DataFrame(
        {
            "Unique ID": ["ad-1", "ad-2"],
            "Years": ["2019", "2020"],
            "Brand": ["A", "B"],
            "Topics": ["Climate", "Energy"],
            "Imagery": ["People", "Factory"],
            "Transcript (text and audio)": [
                "[ad text:] Companies can reduce emissions with new technology.",
                "[audio transcript:] We invest in cleaner fuels.",
            ],
        }
    ).to_csv(csv_path, index=False)

    payload = run_fst2graph(
        input_csv=csv_path,
        out_root=tmp_path / "out",
        text_col="Transcript (text and audio)",
        id_col="Unique ID",
        doc_col="Unique ID",
        fst=FakeFST(),
        chunk_min_words=2,
        chunk_max_words=70,
        top_n_frames=10,
        top_n_agents=10,
        random_seed=42,
    )

    run_root = Path(payload["run_root"])
    assert run_root.exists()
    assert Path(payload["summary_path"]).exists()
    assert (Path(payload["run_dir"]) / "frame_elements_long.csv").exists()
    assert (Path(payload["graph_out_dir"]) / "graph.gpickle").exists()
    assert (Path(payload["analysis_out_dir"]) / "agent_frame_lift.csv").exists()
    assert (Path(payload["analysis_out_dir"]) / "sample_path_traces.csv").exists()

    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["input_rows"] == 2
    assert summary["chunk_rows"] >= 2
    assert "graph_report" in summary
    assert "analysis_report" in summary

    second = run_fst2graph(
        input_csv=csv_path,
        out_root=tmp_path / "out",
        text_col="Transcript (text and audio)",
        id_col="Unique ID",
        doc_col="Unique ID",
        fst=FakeFST(),
        chunk_min_words=2,
        chunk_max_words=70,
        top_n_frames=10,
        top_n_agents=10,
        random_seed=42,
        resume=True,
    )
    assert second["run_root"] == payload["run_root"]
    assert second["extraction_report"]["processed_this_run"] == 0


def test_pipeline_v2_handles_zero_frame_rows_without_crashing(tmp_path: Path) -> None:
    csv_path = tmp_path / "zero_frames.csv"
    pd.DataFrame(
        {
            "Unique ID": ["ad-1"],
            "Transcript (text and audio)": ["[ad text:] very short line."],
        }
    ).to_csv(csv_path, index=False)

    payload = run_fst2graph(
        input_csv=csv_path,
        out_root=tmp_path / "out",
        text_col="Transcript (text and audio)",
        id_col="Unique ID",
        doc_col="Unique ID",
        fst=EmptyFST(),
        resume=False,
    )
    assert Path(payload["summary_path"]).exists()
    assert payload["extraction_report"]["frame_instances"] == 0
