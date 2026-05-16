from __future__ import annotations

import json
import sqlite3
import builtins
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from fst2framegraph import encode_with_fst, materialise_run
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
    frames: list[FakeFrame]


def fake_result(sentence: str) -> FakeResult:
    return FakeResult(
        frames=[
            FakeFrame(
                name="Capability",
                trigger_location=sentence.index("can"),
                frame_elements=[
                    FakeFE(name="Entity", text=sentence.split(" can ", 1)[0]),
                    FakeFE(name="Event", text=sentence.split(" can ", 1)[1].rstrip(".")),
                ],
            )
        ]
    )


class FakeFST:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def detect_frames(self, sentence: str) -> FakeResult:
        self.calls.append(sentence)
        return fake_result(sentence)


class FakeBatchFST:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def detect_frames(self, sentence: str) -> FakeResult:
        raise AssertionError("batch path should be used")

    def detect_frames_batch(self, sentences: list[str]) -> list[FakeResult]:
        self.batch_sizes.append(len(sentences))
        return [fake_result(sentence) for sentence in sentences]


class SelectiveFailFST:
    def __init__(self, fail_ids: set[str] | None = None) -> None:
        self.calls: list[str] = []
        self.fail_ids = fail_ids or set()

    def detect_frames(self, sentence: str) -> FakeResult:
        self.calls.append(sentence)
        sentence_id = sentence.split(":", 1)[0]
        if sentence_id in self.fail_ids:
            raise RuntimeError(f"planned failure for {sentence_id}")
        return fake_result(sentence.split(": ", 1)[1])


class SpanCaseFST:
    def detect_frames(self, sentence: str) -> FakeResult:
        case, text = sentence.split(": ", 1)
        filler_by_case = {
            "unique": "alpha",
            "ambiguous": "alpha",
            "notfound": "gamma",
            "empty": "",
            "missing": None,
        }
        return FakeResult(
            frames=[
                FakeFrame(
                    name="TestFrame",
                    trigger_location=0,
                    frame_elements=[FakeFE(name="Role", text=filler_by_case[case])],
                )
            ]
        )


def test_jsonl_progress_and_materialise_are_authoritative(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "sentence_id": ["s1", "s2"],
            "doc_id": ["d1", "d1"],
            "sentence": ["Technology can reduce emissions.", "Policy can guide markets."],
        }
    )

    run_dir = tmp_path / "run"
    report = encode_with_fst(
        fst=FakeFST(),
        data=df,
        sentence_id_col="sentence_id",
        doc_col="doc_id",
        out_dir=run_dir,
        checkpoint_every=1,
        resume=False,
    )

    assert report["sentences"] == 2
    assert (run_dir / "fst_clean.jsonl").exists()
    assert (run_dir / "progress.sqlite").exists()
    assert not list(run_dir.rglob("*.pkl"))
    assert not list(run_dir.rglob("*.pickle"))

    records = [
        json.loads(line)
        for line in (run_dir / "fst_clean.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == len(df)
    assert {record["status"] for record in records} == {"completed"}
    assert all("row_index" in record for record in records)
    assert all("doc_id" in record for record in records)
    assert all("sentence" in record for record in records)
    assert all("frames" in record for record in records)

    with sqlite3.connect(run_dir / "progress.sqlite") as conn:
        statuses = dict(conn.execute("SELECT sentence_id, status FROM progress").fetchall())
    assert statuses == {"s1": "completed", "s2": "completed"}

    (run_dir / "frame_elements_long.csv").write_text("broken\n", encoding="utf-8")
    rebuilt = materialise_run(run_dir)
    assert rebuilt["frame_elements"] == 4
    elements = pd.read_csv(run_dir / "frame_elements_long.csv")
    assert set(elements["sentence_id"]) == {"s1", "s2"}
    assert set(elements["element_name"]) == {"Entity", "Event"}


def test_resume_skips_completed_sentence_ids(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "sentence_id": ["s1", "s2"],
            "sentence": ["Technology can reduce emissions.", "Policy can guide markets."],
        }
    )
    run_dir = tmp_path / "run"

    first_fst = FakeFST()
    encode_with_fst(
        fst=first_fst,
        data=df,
        sentence_id_col="sentence_id",
        out_dir=run_dir,
        limit=1,
        checkpoint_every=1,
        resume=False,
    )
    assert first_fst.calls == ["Technology can reduce emissions."]

    second_fst = FakeFST()
    report = encode_with_fst(
        fst=second_fst,
        data=df,
        sentence_id_col="sentence_id",
        out_dir=run_dir,
        checkpoint_every=1,
        resume=True,
    )

    assert second_fst.calls == ["Policy can guide markets."]
    assert report["skipped_existing"] == 1
    assert report["sentences"] == 2


def test_failed_rows_are_checkpointed_and_retry_errors_is_explicit(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "sentence_id": ["s1", "s2"],
            "sentence": ["s1: Technology can reduce emissions.", "s2: Policy can guide markets."],
        }
    )
    run_dir = tmp_path / "run"

    first = SelectiveFailFST(fail_ids={"s1"})
    first_report = encode_with_fst(
        fst=first,
        data=df,
        sentence_id_col="sentence_id",
        out_dir=run_dir,
        checkpoint_every=1,
        resume=False,
    )
    assert first_report["errors"] == 1

    records = [
        json.loads(line)
        for line in (run_dir / "fst_clean.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 2
    errors = [record for record in records if record["status"] == "error"]
    assert len(errors) == 1
    assert errors[0]["sentence_id"] == "s1"
    assert "planned failure for s1" in errors[0]["error_message"]

    no_retry = SelectiveFailFST()
    no_retry_report = encode_with_fst(
        fst=no_retry,
        data=df,
        sentence_id_col="sentence_id",
        out_dir=run_dir,
        checkpoint_every=1,
        resume=True,
        retry_errors=False,
    )
    assert no_retry.calls == []
    assert no_retry_report["errors"] == 1

    retry = SelectiveFailFST()
    retry_report = encode_with_fst(
        fst=retry,
        data=df,
        sentence_id_col="sentence_id",
        out_dir=run_dir,
        checkpoint_every=1,
        resume=True,
        retry_errors=True,
    )
    assert retry.calls == ["s1: Technology can reduce emissions."]
    assert retry_report["errors"] == 0


def test_resume_uses_sentence_id_not_row_position(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    first_df = pd.DataFrame(
        {"sentence_id": ["s1"], "sentence": ["Technology can reduce emissions."]}
    )
    second_df = pd.DataFrame(
        {
            "sentence_id": ["s2", "s1"],
            "sentence": ["Policy can guide markets.", "Technology can reduce emissions."],
        }
    )

    encode_with_fst(
        fst=FakeFST(),
        data=first_df,
        sentence_id_col="sentence_id",
        out_dir=run_dir,
        resume=False,
    )
    second = FakeFST()
    report = encode_with_fst(
        fst=second,
        data=second_df,
        sentence_id_col="sentence_id",
        out_dir=run_dir,
        resume=True,
    )

    assert second.calls == ["Policy can guide markets."]
    assert report["skipped_existing"] == 1


def test_duplicate_sentence_ids_are_rejected(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "sentence_id": ["s1", "s1"],
            "sentence": ["Technology can reduce emissions.", "Policy can guide markets."],
        }
    )

    with pytest.raises(ValueError, match="Duplicate sentence_id"):
        encode_with_fst(
            fst=FakeFST(),
            data=df,
            sentence_id_col="sentence_id",
            out_dir=tmp_path / "run",
            resume=False,
        )


def test_materialise_cli_rebuilds_outputs(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "sentence_id": ["s1"],
            "sentence": ["Technology can reduce emissions."],
        }
    )
    run_dir = tmp_path / "run"
    encode_with_fst(
        fst=FakeFST(),
        data=df,
        sentence_id_col="sentence_id",
        out_dir=run_dir,
        resume=False,
    )
    (run_dir / "frame_instances.csv").write_text("broken\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["materialise", "--run-dir", str(run_dir)])

    assert result.exit_code == 0, result.output
    frames = pd.read_csv(run_dir / "frame_instances.csv")
    assert frames.loc[0, "frame_name"] == "Capability"


def test_materialise_rebuilds_all_csvs_and_is_idempotent(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "sentence_id": ["s1", "s2"],
            "sentence": ["Technology can reduce emissions.", "Policy can guide markets."],
        }
    )
    run_dir = tmp_path / "run"
    encode_with_fst(
        fst=FakeFST(),
        data=df,
        sentence_id_col="sentence_id",
        out_dir=run_dir,
        resume=False,
    )

    for csv_path in run_dir.glob("*.csv"):
        csv_path.unlink()

    first = materialise_run(run_dir)
    first_report = (run_dir / "extraction_report.json").read_bytes()
    first_csvs = {path.name: path.read_bytes() for path in sorted(run_dir.glob("*.csv"))}
    second = materialise_run(run_dir)
    second_report = (run_dir / "extraction_report.json").read_bytes()
    second_csvs = {path.name: path.read_bytes() for path in sorted(run_dir.glob("*.csv"))}

    assert first == second
    assert first_report == second_report
    assert first_csvs == second_csvs
    assert {"sentences.csv", "frame_instances.csv", "frame_elements_long.csv", "errors.csv"} <= set(
        first_csvs
    )
    assert pd.read_csv(run_dir / "frame_elements_long.csv")["sentence"].notna().all()


def test_materialise_repairs_missing_and_truncated_csvs(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "sentence_id": ["s1", "s2"],
            "sentence": ["Technology can reduce emissions.", "Policy can guide markets."],
        }
    )
    run_dir = tmp_path / "run"
    encode_with_fst(
        fst=FakeFST(),
        data=df,
        sentence_id_col="sentence_id",
        out_dir=run_dir,
        resume=False,
    )

    (run_dir / "sentences.csv").unlink()
    (run_dir / "frame_elements_long.csv").write_text("sentence_id\ns1\n", encoding="utf-8")

    materialise_run(run_dir)

    assert len(pd.read_csv(run_dir / "sentences.csv")) == 2
    assert len(pd.read_csv(run_dir / "frame_elements_long.csv")) == 4


def test_materialise_corrupt_jsonl_errors_without_overwriting_csvs(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {"sentence_id": ["s1"], "sentence": ["Technology can reduce emissions."]}
    )
    run_dir = tmp_path / "run"
    encode_with_fst(
        fst=FakeFST(),
        data=df,
        sentence_id_col="sentence_id",
        out_dir=run_dir,
        resume=False,
    )
    before = (run_dir / "frame_elements_long.csv").read_bytes()
    with (run_dir / "fst_clean.jsonl").open("a", encoding="utf-8") as f:
        f.write("{not json}\n")

    with pytest.raises(ValueError, match="Invalid JSONL record"):
        materialise_run(run_dir)

    assert (run_dir / "frame_elements_long.csv").read_bytes() == before


def test_span_statuses_survive_jsonl_materialise_csv_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "sentence_id": ["u", "a", "n", "e", "m"],
            "sentence": [
                "unique: alpha beta",
                "ambiguous: alpha beta alpha",
                "notfound: alpha beta",
                "empty: alpha beta",
                "missing: alpha beta",
            ],
        }
    )
    run_dir = tmp_path / "run"
    encode_with_fst(
        fst=SpanCaseFST(),
        data=df,
        sentence_id_col="sentence_id",
        out_dir=run_dir,
        resume=False,
    )
    materialise_run(run_dir)

    rows = pd.read_csv(run_dir / "frame_elements_long.csv").set_index("sentence_id")
    assert rows.loc["u", "span_status"] == "exact_unique"
    assert rows.loc["a", "span_status"] == "exact_ambiguous_not_used"
    assert rows.loc["n", "span_status"] == "not_found"
    assert rows.loc["e", "span_status"] == "empty_text"
    assert rows.loc["m", "span_status"] == "missing_text"

    ambig_dir = tmp_path / "ambig"
    encode_with_fst(
        fst=SpanCaseFST(),
        data=df[df["sentence_id"] == "a"],
        sentence_id_col="sentence_id",
        out_dir=ambig_dir,
        allow_ambiguous_spans=True,
        resume=False,
    )
    rows = pd.read_csv(ambig_dir / "frame_elements_long.csv").set_index("sentence_id")
    assert rows.loc["a", "span_status"] == "exact_ambiguous_first_used"


def test_batch_detection_is_used_when_available(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "sentence_id": ["s1", "s2", "s3"],
            "sentence": [
                "Technology can reduce emissions.",
                "Policy can guide markets.",
                "Teams can build tools.",
            ],
        }
    )
    fst = FakeBatchFST()

    encode_with_fst(
        fst=fst,
        data=df,
        sentence_id_col="sentence_id",
        out_dir=tmp_path / "run",
        batch_size=2,
        resume=False,
    )

    assert fst.batch_sizes == [2, 1]


def test_batch_size_falls_back_to_single_sentence_api(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "sentence_id": ["s1", "s2"],
            "sentence": ["Technology can reduce emissions.", "Policy can guide markets."],
        }
    )
    fst = FakeFST()

    encode_with_fst(
        fst=fst,
        data=df,
        sentence_id_col="sentence_id",
        out_dir=tmp_path / "run",
        batch_size=99,
        resume=False,
    )

    assert fst.calls == ["Technology can reduce emissions.", "Policy can guide markets."]


def test_device_auto_does_not_require_torch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    df = pd.DataFrame(
        {"sentence_id": ["s1"], "sentence": ["Technology can reduce emissions."]}
    )

    report = encode_with_fst(
        fst=FakeFST(),
        data=df,
        sentence_id_col="sentence_id",
        out_dir=tmp_path / "run",
        device="auto",
        resume=False,
    )

    assert report["device"] == "cpu"
