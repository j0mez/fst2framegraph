from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fst2framegraph import encode_with_fst


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
    trigger_locations: list[int]
    frames: list[FakeFrame]


class FakeFST:
    def detect_frames(self, sentence: str) -> FakeResult:
        return FakeResult(
            sentence=sentence,
            trigger_locations=[11],
            frames=[
                FakeFrame(
                    name="Capability",
                    trigger_location=11,
                    frame_elements=[
                        FakeFE(name="Entity", text="Technology"),
                        FakeFE(name="Event", text="reduce emissions"),
                    ],
                )
            ],
        )


def test_encode_with_fst_writes_clean_tables(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "sentence_id": ["s1"],
            "doc_id": ["d1"],
            "sentence": ["Technology can reduce emissions."],
            "source": ["unit-test"],
        }
    )

    report = encode_with_fst(
        fst=FakeFST(),
        data=df,
        sentence_col="sentence",
        sentence_id_col="sentence_id",
        doc_col="doc_id",
        metadata_cols=["source"],
        out_dir=tmp_path / "clean",
    )

    assert report["sentences"] == 1
    assert report["frame_instances"] == 1
    assert report["frame_elements"] == 2

    elements = pd.read_csv(tmp_path / "clean" / "frame_elements_long.csv")
    assert set(elements["element_name"]) == {"Entity", "Event"}
    assert "frame_instance_id" in elements.columns
    assert "span_status" in elements.columns
    assert set(elements["span_status"]) == {"exact_unique"}

    frames = pd.read_csv(tmp_path / "clean" / "frame_instances.csv")
    assert frames.loc[0, "frame_name"] == "Capability"
    assert frames.loc[0, "frame_index"] == 0
