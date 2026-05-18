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


class SyntheticSetstateKeyError:
    def __init__(self) -> None:
        self.frames = []

    def __getstate__(self) -> dict[str, object]:
        return self.__dict__

    def __getattr__(self, name: str):
        raise KeyError(name)


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


def test_synthetic_legacy_getattr_reproduces_pickle_setstate_keyerror() -> None:
    payload = pickle.dumps(SyntheticSetstateKeyError())

    with pytest.raises(KeyError, match="__setstate__"):
        pickle.loads(payload)


def test_inspect_and_convert_trusted_legacy_nltk_framenet_pickle(tmp_path: Path) -> None:
    framenet = pytest.importorskip("nltk.corpus.reader.framenet")
    pickle_path = tmp_path / "legacy_raw_results.pkl"
    sentence = "Technology can reduce emissions."
    payload = {
        "batch_key": "batch-1",
        "first_global_unique_row": 12944,
        "last_global_unique_row": 12944,
        "unique_chunk_ids": ["chunk-12944"],
        "sentences": [sentence],
        "errors": [None],
        "raw_results": [
            FakeResult(
                sentence=sentence,
                frames=[
                    FakeFrame(
                        name="Capability",
                        trigger_location=11,
                        frame_elements=[FakeFE(name="Entity", text="Technology")],
                    )
                ],
            )
        ],
        "framenet_probe": framenet.PrettyDict({"name": "tiny"}),
    }
    pickle_path.write_bytes(pickle.dumps(payload))

    inspect_result = CliRunner().invoke(
        app, ["inspect", "--input", str(pickle_path), "--allow-pickle"]
    )
    assert inspect_result.exit_code == 0, inspect_result.output
    assert '"status": "convertible"' in inspect_result.output
    assert '"records": 1' in inspect_result.output

    report = convert_fst_outputs(pickle_path, tmp_path / "clean", allow_pickle=True)

    assert report["sentences"] == 1
    elements = pd.read_csv(tmp_path / "clean" / "frame_elements_long.csv")
    assert set(elements["sentence_id"]) == {"chunk-12944"}
    assert elements.loc[0, "frame_name"] == "Capability"
    assert elements.loc[0, "element_name"] == "Entity"


def test_convert_trusted_pickle_reports_unsupported_structure(tmp_path: Path) -> None:
    pickle_path = tmp_path / "unsupported.pkl"
    pickle_path.write_bytes(pickle.dumps({"unexpected": "shape"}))

    with pytest.raises(ValueError, match="Unsupported trusted pickle payload"):
        convert_fst_outputs(pickle_path, tmp_path / "clean", allow_pickle=True)


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


def test_prepare_clean_v03_run_directory(tmp_path: Path) -> None:
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

    result = CliRunner().invoke(
        app,
        ["prepare", "--input", str(run_dir), "--out", str(tmp_path / "prepared")],
    )

    assert result.exit_code == 0, result.output
    assert "Prepared canonical run directory" in result.output
    assert '"graph_ready": true' in result.output
    assert (tmp_path / "prepared" / "fst_clean.jsonl").exists()
    assert (tmp_path / "prepared" / "frame_elements_long.csv").exists()
    assert "fst2framegraph build --input" in result.output


def test_prepare_graph_ready_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "frame_elements_long.csv"
    graph_ready_rows().to_csv(csv_path, index=False)

    result = CliRunner().invoke(
        app,
        ["prepare", "--input", str(csv_path), "--out", str(tmp_path / "clean")],
    )

    assert result.exit_code == 0, result.output
    assert '"detected_format": "graph_ready_csv"' in result.output
    assert (tmp_path / "clean" / "fst_clean.jsonl").exists()
    assert (tmp_path / "clean" / "progress.sqlite").exists()
    assert (tmp_path / "clean" / "frame_elements_long.csv").exists()


def test_prepare_convertible_jsonl(tmp_path: Path) -> None:
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

    result = CliRunner().invoke(
        app,
        ["prepare", "--input", str(jsonl_path), "--out", str(tmp_path / "clean")],
    )

    assert result.exit_code == 0, result.output
    assert '"detected_format": "fst_jsonl"' in result.output
    assert (tmp_path / "clean" / "fst_clean.jsonl").exists()
    assert (tmp_path / "clean" / "frame_elements_long.csv").exists()


def test_prepare_flat_only_csv_reports_not_graph_ready(tmp_path: Path) -> None:
    csv_path = tmp_path / "flat.csv"
    graph_ready_rows().drop(
        columns=["frame_index", "target_start", "target_end", "filler_start", "filler_end"]
    ).to_csv(csv_path, index=False)

    result = CliRunner().invoke(
        app,
        ["prepare", "--input", str(csv_path), "--out", str(tmp_path / "clean")],
    )

    assert result.exit_code == 1, result.output
    assert '"graph_ready": false' in result.output
    assert "reliable nested graphs require frame_index and target/filler spans" in result.output
    assert "fst2framegraph detect --input" in result.output
    assert not (tmp_path / "clean" / "fst_clean.jsonl").exists()


def test_prepare_pickle_folder_refuses_without_allow_pickle(tmp_path: Path) -> None:
    pickle_dir = tmp_path / "pickles"
    pickle_dir.mkdir()
    (pickle_dir / "result.pkl").write_bytes(pickle.dumps([]))

    result = CliRunner().invoke(
        app,
        ["prepare", "--input", str(pickle_dir), "--out", str(tmp_path / "clean")],
    )

    assert result.exit_code == 1, result.output
    assert "Python pickles can execute code" in result.output
    assert "fst2framegraph prepare --input" in result.output
    assert "--allow-pickle" in result.output
    assert not (tmp_path / "clean" / "fst_clean.jsonl").exists()


def test_prepare_trusted_fake_pickle_when_allowed(tmp_path: Path) -> None:
    pickle_dir = tmp_path / "pickles"
    pickle_dir.mkdir()
    (pickle_dir / "result.pkl").write_bytes(
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

    result = CliRunner().invoke(
        app,
        [
            "prepare",
            "--input",
            str(pickle_dir),
            "--out",
            str(tmp_path / "clean"),
            "--allow-pickle",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"detected_format": "pickle_folder"' in result.output
    assert (tmp_path / "clean" / "fst_clean.jsonl").exists()
    assert not list((tmp_path / "clean").rglob("*.pkl"))


def test_prepare_output_can_be_passed_to_build(tmp_path: Path) -> None:
    csv_path = tmp_path / "frame_elements_long.csv"
    graph_ready_rows().to_csv(csv_path, index=False)
    runner = CliRunner()

    prepare_result = runner.invoke(
        app,
        ["prepare", "--input", str(csv_path), "--out", str(tmp_path / "clean")],
    )
    build_result = runner.invoke(
        app,
        [
            "build",
            "--input",
            str(tmp_path / "clean"),
            "--out",
            str(tmp_path / "graph"),
            "--framebase-dir",
            str(tmp_path / "empty-framebase"),
            "--no-rdf",
        ],
    )

    assert prepare_result.exit_code == 0, prepare_result.output
    assert build_result.exit_code == 0, build_result.output
    assert (tmp_path / "graph" / "summary.json").exists()


def test_run_plan_on_graph_ready_csv_writes_no_files(tmp_path: Path) -> None:
    csv_path = tmp_path / "frame_elements_long.csv"
    out_dir = tmp_path / "fst_clean"
    graph_ready_rows().to_csv(csv_path, index=False)

    result = CliRunner().invoke(
        app,
        ["run", "--plan", "--input", str(csv_path), "--out", str(out_dir)],
    )

    assert result.exit_code == 0, result.output
    assert "Detected: graph-ready CSV" in result.output
    assert "convert input to canonical run directory" in result.output
    assert "materialise CSV/report outputs" in result.output
    assert "run doctor checks" in result.output
    assert "To execute:" in result.output
    assert not out_dir.exists()


def test_run_on_canonical_run_directory_materialises_and_doctors(tmp_path: Path) -> None:
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
    (run_dir / "frame_elements_long.csv").unlink()

    result = CliRunner().invoke(
        app,
        ["run", "--input", str(run_dir), "--out", str(tmp_path / "unused")],
    )

    assert result.exit_code == 0, result.output
    assert "canonical run directory" in result.output
    assert '"graph_ready": true' in result.output
    assert (run_dir / "frame_elements_long.csv").exists()
    assert "fst2framegraph build --input" in result.output


def test_run_on_graph_ready_csv_creates_canonical_run(tmp_path: Path) -> None:
    csv_path = tmp_path / "frame_elements_long.csv"
    out_dir = tmp_path / "fst_clean"
    graph_ready_rows().to_csv(csv_path, index=False)

    result = CliRunner().invoke(
        app,
        ["run", "--input", str(csv_path), "--out", str(out_dir)],
    )

    assert result.exit_code == 0, result.output
    assert (out_dir / "fst_clean.jsonl").exists()
    assert (out_dir / "frame_elements_long.csv").exists()
    assert '"graph_ready": true' in result.output


def test_run_on_convertible_jsonl_creates_canonical_run(tmp_path: Path) -> None:
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

    result = CliRunner().invoke(
        app,
        ["run", "--input", str(jsonl_path), "--out", str(tmp_path / "fst_clean")],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "fst_clean" / "fst_clean.jsonl").exists()
    assert (tmp_path / "fst_clean" / "frame_elements_long.csv").exists()


def test_run_on_raw_sentence_csv_uses_detect_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "sentences.csv"
    csv_path.write_text(
        "sentence_id,doc_id,sentence\ns1,d1,Technology can reduce emissions.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "fst2framegraph.fst.export._make_default_fst",
        lambda device: FakeFST(),
    )

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--input",
            str(csv_path),
            "--text-col",
            "sentence",
            "--id-col",
            "sentence_id",
            "--doc-col",
            "doc_id",
            "--out",
            str(tmp_path / "fst_clean"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "fst_clean" / "fst_clean.jsonl").exists()
    report = json.loads(
        (tmp_path / "fst_clean" / "extraction_report.json").read_text(encoding="utf-8")
    )
    assert report["dedupe_enabled"] is True
    assert report["unique_texts"] == 1


def test_run_on_text_rows_auto_chunks_and_writes_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "ads.csv"
    csv_path.write_text(
        (
            "ad_id,row_id,text\n"
            "ad1,r1,Technology can reduce emissions. Companies report progress.\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "fst2framegraph.fst.export._make_default_fst",
        lambda device: FakeFST(),
    )

    out_dir = tmp_path / "fst_clean"
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--input",
            str(csv_path),
            "--text-col",
            "text",
            "--id-col",
            "row_id",
            "--doc-col",
            "ad_id",
            "--out",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    chunked = pd.read_csv(out_dir / "text_chunks.csv")
    mapping = pd.read_csv(out_dir / "text_chunk_mapping.csv")
    assert len(chunked) >= 2
    assert {"sentence_id", "doc_id", "sentence"} <= set(chunked.columns)
    assert len(mapping) == len(chunked)


def test_run_on_flat_only_csv_exits_without_graph_ready_claim(tmp_path: Path) -> None:
    csv_path = tmp_path / "flat.csv"
    graph_ready_rows().drop(
        columns=["frame_index", "target_start", "target_end", "filler_start", "filler_end"]
    ).to_csv(csv_path, index=False)

    result = CliRunner().invoke(
        app,
        ["run", "--input", str(csv_path), "--out", str(tmp_path / "fst_clean")],
    )

    assert result.exit_code == 1, result.output
    assert "flat-only" in result.output
    assert "Flat frame/FE counts may be possible" in result.output
    assert '"graph_ready": false' in result.output
    assert not (tmp_path / "fst_clean" / "fst_clean.jsonl").exists()


def test_run_pickle_folder_refuses_without_allow_pickle(tmp_path: Path) -> None:
    pickle_dir = tmp_path / "pickles"
    pickle_dir.mkdir()
    (pickle_dir / "result.pkl").write_bytes(pickle.dumps([]))

    result = CliRunner().invoke(
        app,
        ["run", "--input", str(pickle_dir), "--out", str(tmp_path / "fst_clean")],
    )

    assert result.exit_code == 1, result.output
    assert "Pickles can execute code" in result.output
    assert "fst2framegraph run --input" in result.output
    assert "--allow-pickle" in result.output
    assert not (tmp_path / "fst_clean" / "fst_clean.jsonl").exists()


def test_run_materialises_directory_when_csv_missing(tmp_path: Path) -> None:
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
    (run_dir / "frame_elements_long.csv").unlink()

    result = CliRunner().invoke(app, ["run", "--input", str(run_dir), "--out", str(run_dir)])

    assert result.exit_code == 0, result.output
    assert (run_dir / "frame_elements_long.csv").exists()


def test_run_interactive_defaults_no_for_pickle_loading(tmp_path: Path) -> None:
    pickle_dir = tmp_path / "pickles"
    pickle_dir.mkdir()
    (pickle_dir / "result.pkl").write_bytes(pickle.dumps([]))

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--interactive",
            "--input",
            str(pickle_dir),
            "--out",
            str(tmp_path / "fst_clean"),
        ],
        input="\n",
    )

    assert result.exit_code == 1, result.output
    assert "Load trusted pickles?" in result.output
    assert not (tmp_path / "fst_clean" / "fst_clean.jsonl").exists()


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
    assert "fst2framegraph build --input" in result.output


def test_doctor_suggests_materialise_when_csvs_missing(tmp_path: Path) -> None:
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
    for name in [
        "sentences.csv",
        "frame_instances.csv",
        "frame_elements.csv",
        "frame_elements_long.csv",
        "errors.csv",
    ]:
        (run_dir / name).unlink()

    result = CliRunner().invoke(app, ["doctor", "--run-dir", str(run_dir)])

    assert result.exit_code == 0, result.output
    assert "fst2framegraph materialise --run-dir" in result.output


def test_doctor_suggests_framebase_index_build(tmp_path: Path) -> None:
    index_path = tmp_path / "framebase" / "framebase_index.sqlite"

    result = CliRunner().invoke(app, ["doctor", "--framebase-index", str(index_path)])

    assert result.exit_code == 1, result.output
    assert "fst2framegraph build-framebase-index --framebase-dir" in result.output


def test_core_cli_help_commands() -> None:
    runner = CliRunner()
    for args in [
        ["--help"],
        ["detect", "--help"],
        ["inspect", "--help"],
        ["convert", "--help"],
        ["prepare", "--help"],
        ["run", "--help"],
        ["materialise", "--help"],
        ["build-framebase-index", "--help"],
        ["build", "--help"],
        ["doctor", "--help"],
    ]:
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        assert "Usage" in result.output

    prepare_help = runner.invoke(app, ["prepare", "--help"])
    run_help = runner.invoke(app, ["run", "--help"])
    build_help = runner.invoke(app, ["build", "--help"])
    detect_help = runner.invoke(app, ["detect", "--help"])
    assert "Prepare existing FST-like output" in prepare_help.output
    assert "inspect + plan + execute" in run_help.output
    assert build_help.exit_code == 0
    assert detect_help.exit_code == 0


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
