from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fst2framegraph.pipeline_v2.extract import run_fst_extraction


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


class FlakyFakeFST:
    def detect_frames(self, sentence: str) -> FakeResult:
        if "FAIL" in sentence:
            raise RuntimeError("intentional failure for test")
        return FakeResult(
            sentence=sentence,
            frames=[
                FakeFrame(
                    name="Capability",
                    trigger_location=3,
                    frame_elements=[
                        FakeFE(name="Agent", text="we"),
                        FakeFE(name="Goal", text="reduce emissions"),
                    ],
                )
            ],
        )


def _sentence_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sentence_id": ["s1", "s2", "s3"],
            "doc_id": ["d1", "d1", "d1"],
            "sentence": ["We can help.", "We can help.", "This will FAIL now."],
        }
    )


def test_resumable_extraction_dedupes_and_skips_on_second_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "fst_clean"
    first = run_fst_extraction(
        sentences_df=_sentence_df(),
        run_dir=run_dir,
        fst=FlakyFakeFST(),
        resume=True,
        dedupe=True,
        dedupe_normalise="exact",
    )
    assert first["processed_this_run"] >= 1
    assert (run_dir / "fst_clean.jsonl").exists()
    assert (run_dir / "progress.sqlite").exists()

    second = run_fst_extraction(
        sentences_df=_sentence_df(),
        run_dir=run_dir,
        fst=FlakyFakeFST(),
        resume=True,
        dedupe=True,
        dedupe_normalise="exact",
    )
    assert second["processed_this_run"] == 0
    assert second["skipped_existing"] >= 3


def test_extraction_keeps_failure_rows_without_crashing(tmp_path: Path) -> None:
    run_dir = tmp_path / "fst_clean"
    report = run_fst_extraction(
        sentences_df=_sentence_df(),
        run_dir=run_dir,
        fst=FlakyFakeFST(),
        resume=False,
        dedupe=True,
        dedupe_normalise="exact",
    )
    assert report["errors"] >= 1
    errors = pd.read_csv(run_dir / "errors.csv")
    assert not errors.empty
