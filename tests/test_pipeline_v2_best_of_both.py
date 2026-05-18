from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner


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
        lower = sentence.lower()
        if "invest" in lower:
            return FakeResult(
                sentence=sentence,
                frames=[
                    FakeFrame(
                        name="Investing",
                        trigger_location=lower.find("invest"),
                        frame_elements=[FakeFE(name="Agent", text="BP")],
                    )
                ],
            )
        if "protect" in lower:
            return FakeResult(
                sentence=sentence,
                frames=[
                    FakeFrame(
                        name="Protection",
                        trigger_location=lower.find("protect"),
                        frame_elements=[FakeFE(name="Agent", text="Chevron")],
                    )
                ],
            )
        return FakeResult(sentence=sentence, frames=[])


def test_pipeline_v2_combines_product_flow_with_strict_oxccal_cleaning(
    tmp_path: Path,
) -> None:
    from fst2framegraph import run_fst2graph

    csv_path = tmp_path / "oxccal.csv"
    pd.DataFrame(
        {
            "Unique ID": ["ad-1", "ad-2", "ad-3"],
            "Brand": ["BP", "Chevron", "Shell"],
            "Transcript (text and audio)": [
                "[ad text:] BP will invest in lower carbon energy. [audio transcript:] upbeat music",
                "[audio transcript:] Shell mentions invest in narration only.",
                "[ad text:] Chevron will protect reliability.",
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
        resume=False,
        top_n_frames=20,
        top_n_agents=30,
        min_count=1,
    )

    run_root = Path(payload["run_root"])
    source_rows = pd.read_csv(run_root / "source_rows.csv", keep_default_na=False)
    sentence_rows = pd.read_csv(run_root / "sentence_rows.csv")
    lift = pd.read_csv(Path(payload["analysis_out_dir"]) / "agent_frame_lift.csv")

    assert payload["input_rows"] == 3
    assert payload["chunk_rows"] == 2
    assert source_rows.loc[source_rows["source_id"] == "ad-2", "raw_text"].item() == ""
    assert "narration only" not in " ".join(sentence_rows["sentence"].astype(str))
    assert set(lift["frame_type"]) == {"Investing", "Protection"}
    assert (Path(payload["graph_out_dir"]) / "graph.graphml").exists()
    assert Path(payload["summary_path"]).exists()


def test_pipeline_v2_preflight_points_colab_to_local_wheel_install() -> None:
    from fst2framegraph.pipeline_v2.preflight import COLAB_INSTALL_HINT

    assert "pip install --find-links=wheels/ -e ." in COLAB_INSTALL_HINT
    assert "sentencepiece==0.2.0" in COLAB_INSTALL_HINT


def test_typer_pipeline_command_wraps_v2_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import fst2framegraph.cli as cli_module

    csv_path = tmp_path / "input.csv"
    csv_path.write_text("id,text\n1,hello world\n", encoding="utf-8")

    def fake_run_fst2graph(**kwargs):
        assert kwargs["input_csv"] == csv_path
        run_root = Path(kwargs["out_root"]) / "run_stub"
        run_root.mkdir(parents=True)
        payload = {
            "run_id": "run_stub",
            "input_rows": 1,
            "chunk_rows": 1,
            "run_root": str(run_root),
            "preflight": {"ok": True},
        }
        summary_path = run_root / "summary.json"
        summary_path.write_text(json.dumps(payload), encoding="utf-8")
        payload["summary_path"] = str(summary_path)
        return payload

    monkeypatch.setattr(cli_module, "run_fst2graph", fake_run_fst2graph)
    result = CliRunner().invoke(
        cli_module.app,
        [
            "pipeline",
            "--input",
            str(csv_path),
            "--out-root",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["run_id"] == "run_stub"
    assert Path(payload["summary_path"]).exists()
