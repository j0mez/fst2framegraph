from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from fst2framegraph import convert_fst_outputs, encode_with_fst, inspect_fst_outputs
from fst2framegraph.cli import app


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
                    name="Capability",
                    trigger_location=11,
                    frame_elements=[FakeFE(name="Entity", text="Technology")],
                )
            ],
        )


def graph_ready_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sentence_id": ["s1"],
            "doc_id": ["d1"],
            "sentence": ["Technology can help consumers reduce emissions."],
            "frame_index": [0],
            "frame_name": ["Assistance"],
            "target_text": ["help"],
            "target_start": [15],
            "target_end": [19],
            "element_name": ["Goal"],
            "element_filler": ["consumers reduce emissions"],
            "filler_start": [20],
            "filler_end": [46],
        }
    )


def test_inspect_clean_v03_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    encode_with_fst(
        fst=FakeFST(),
        data=pd.DataFrame(
            {"sentence_id": ["s1"], "sentence": ["Technology can reduce emissions."]}
        ),
        sentence_id_col="sentence_id",
        out_dir=run_dir,
        resume=False,
    )

    report = inspect_fst_outputs(run_dir)

    assert report["status"] == "ready"
    assert report["detected_format"] == "v0.3_run_directory"
    assert report["graph_ready"] is True
    assert report["pickle_files"] == []


def test_inspect_graph_ready_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "frame_elements_long.csv"
    graph_ready_rows().to_csv(csv_path, index=False)

    report = inspect_fst_outputs(csv_path)

    assert report["detected_format"] == "graph_ready_csv"
    assert report["status"] == "graph_ready"
    assert report["graph_ready"] is True
    assert report["missing_required_columns"] == []
    assert "build --input" in report["recommended_next_command"]


def test_inspect_flattened_csv_missing_spans_is_flat_only(tmp_path: Path) -> None:
    csv_path = tmp_path / "flat.csv"
    graph_ready_rows().drop(
        columns=["frame_index", "target_start", "target_end", "filler_start", "filler_end"]
    ).to_csv(csv_path, index=False)

    report = inspect_fst_outputs(csv_path)

    assert report["status"] == "flat_only"
    assert report["graph_ready"] is False
    assert "target_start" in report["missing_required_columns"]
    assert "not sufficient for reliable nested graph" in " ".join(report["warnings"])


def test_convert_jsonl_frames_to_canonical_run(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "fst.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "sentence_id": "s1",
                "doc_id": "d1",
                "sentence": "Technology can help consumers reduce emissions.",
                "frames": [
                    {
                        "frame_index": 0,
                        "frame_name": "Assistance",
                        "target_text": "help",
                        "target_start": 15,
                        "target_end": 19,
                        "frame_elements": [
                            {
                                "element_index": 0,
                                "element_name": "Goal",
                                "element_filler": "consumers reduce emissions",
                                "filler_start": 20,
                                "filler_end": 46,
                                "span_status": "exact_unique",
                            }
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = convert_fst_outputs(jsonl_path, tmp_path / "clean")

    assert report["sentences"] == 1
    assert (tmp_path / "clean" / "fst_clean.jsonl").exists()
    assert (tmp_path / "clean" / "progress.sqlite").exists()
    elements = pd.read_csv(tmp_path / "clean" / "frame_elements_long.csv")
    assert elements.loc[0, "frame_name"] == "Assistance"


def test_inspect_pickle_folder_does_not_load_pickles_by_default(tmp_path: Path) -> None:
    pickle_dir = tmp_path / "pickles"
    pickle_dir.mkdir()
    for name in [
        "unique_global_0000000_to_0000031_raw_results.pkl",
        "unique_global_0000064_to_0000095_raw_results.pkl",
    ]:
        (pickle_dir / name).write_bytes(b"not loaded")

    report = inspect_fst_outputs(pickle_dir)

    assert report["status"] == "unsafe_without_pickle_permission"
    assert report["unsafe_without_pickle_permission"] is True
    assert len(report["pickle_files"]) == 2
    assert report["missing_pickle_ranges"] == [{"expected_start": 32, "expected_end": 63}]


def test_convert_rejects_pickle_without_allow_pickle(tmp_path: Path) -> None:
    pickle_path = tmp_path / "result.pkl"
    pickle_path.write_bytes(pickle.dumps([]))

    with pytest.raises(ValueError, match="Python pickles can execute code"):
        convert_fst_outputs(pickle_path, tmp_path / "clean")


def test_convert_trusted_fake_pickle_when_allowed(tmp_path: Path) -> None:
    pickle_path = tmp_path / "result.pkl"
    pickle_path.write_bytes(
        pickle.dumps(
            [
                {
                    "sentence_id": "s1",
                    "doc_id": "d1",
                    "sentence": "Technology can reduce emissions.",
                    "result": FakeResult(
                        sentence="Technology can reduce emissions.",
                        frames=[
                            FakeFrame(
                                name="Capability",
                                trigger_location=11,
                                frame_elements=[FakeFE(name="Entity", text="Technology")],
                            )
                        ],
                    ),
                }
            ]
        )
    )

    report = convert_fst_outputs(pickle_path, tmp_path / "clean", allow_pickle=True)

    assert report["sentences"] == 1
    elements = pd.read_csv(tmp_path / "clean" / "frame_elements_long.csv")
    assert elements.loc[0, "element_name"] == "Entity"
    assert not list((tmp_path / "clean").rglob("*.pkl"))


def test_convert_output_can_be_passed_to_build(tmp_path: Path) -> None:
    csv_path = tmp_path / "frame_elements_long.csv"
    graph_ready_rows().to_csv(csv_path, index=False)
    convert_fst_outputs(csv_path, tmp_path / "clean")

    result = CliRunner().invoke(
        app,
        [
            "build",
            "--input",
            str(tmp_path / "clean" / "frame_elements_long.csv"),
            "--out",
            str(tmp_path / "graph"),
            "--framebase-dir",
            str(tmp_path / "empty-framebase"),
            "--no-rdf",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "graph" / "summary.json").exists()


def test_inspect_and_convert_cli(tmp_path: Path) -> None:
    csv_path = tmp_path / "frame_elements_long.csv"
    graph_ready_rows().to_csv(csv_path, index=False)
    runner = CliRunner()

    inspect_result = runner.invoke(app, ["inspect", "--input", str(csv_path)])
    convert_result = runner.invoke(
        app, ["convert", "--input", str(csv_path), "--out", str(tmp_path / "clean")]
    )

    assert inspect_result.exit_code == 0, inspect_result.output
    assert "graph_ready_csv" in inspect_result.output
    assert convert_result.exit_code == 0, convert_result.output
    assert (tmp_path / "clean" / "fst_clean.jsonl").exists()


def test_doctor_cli_checks_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    encode_with_fst(
        fst=FakeFST(),
        data=pd.DataFrame(
            {"sentence_id": ["s1"], "sentence": ["Technology can reduce emissions."]}
        ),
        sentence_id_col="sentence_id",
        out_dir=run_dir,
        resume=False,
    )

    result = CliRunner().invoke(app, ["doctor", "--run-dir", str(run_dir)])

    assert result.exit_code == 0, result.output
    assert '"ok": true' in result.output


def test_core_cli_help_commands() -> None:
    runner = CliRunner()
    for args in [
        ["--help"],
        ["detect", "--help"],
        ["inspect", "--help"],
        ["convert", "--help"],
        ["materialise", "--help"],
        ["build-framebase-index", "--help"],
        ["build", "--help"],
        ["doctor", "--help"],
    ]:
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        assert "Usage" in result.output


def test_public_example_files_are_usable(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]

    flat_report = inspect_fst_outputs(root / "examples" / "flat_only_old_fst.csv")
    assert flat_report["status"] == "flat_only"
    assert "target_start" in flat_report["missing_required_columns"]

    jsonl_report = inspect_fst_outputs(root / "examples" / "fst_like.jsonl")
    assert jsonl_report["status"] == "convertible"

    convert_report = convert_fst_outputs(
        root / "examples" / "fst_like.jsonl",
        tmp_path / "fst_clean",
    )
    assert convert_report["sentences"] == 2
    assert (tmp_path / "fst_clean" / "frame_elements_long.csv").exists()
    assert not list((tmp_path / "fst_clean").rglob("*.pkl"))
