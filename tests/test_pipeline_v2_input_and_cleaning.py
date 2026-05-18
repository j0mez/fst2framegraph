from __future__ import annotations

from pathlib import Path

import pandas as pd

from fst2framegraph.pipeline_v2.chunking import build_chunk_table, split_into_chunks
from fst2framegraph.pipeline_v2.input_schema import load_input_csv
from fst2framegraph.pipeline_v2.text_cleaning import clean_text_input


def test_load_input_csv_supports_oxccal_headers(tmp_path: Path) -> None:
    csv_path = tmp_path / "oxccal.csv"
    pd.DataFrame(
        {
            "Unique ID": ["ad-1"],
            "Years": ["2019"],
            "Brand": ["ExampleCo"],
            "Topics": ["Climate"],
            "Imagery": ["People"],
            "Transcript (text and audio)": ["[ad text:] We reduce emissions."],
        }
    ).to_csv(csv_path, index=False)

    df, cols = load_input_csv(
        csv_path,
        text_col="Transcript (text and audio)",
        id_col="Unique ID",
        doc_col="Unique ID",
    )
    assert cols.text_col == "Transcript (text and audio)"
    assert cols.id_col == "Unique ID"
    assert cols.doc_col == "Unique ID"
    assert df.loc[0, "source_id"] == "ad-1"
    assert df.loc[0, "source_doc_id"] == "ad-1"


def test_clean_text_input_removes_markers_urls_and_noise() -> None:
    raw = " [ad text:] Visit https://example.com now.\n\n[audio transcript:] We act. "
    clean = clean_text_input(raw)
    assert "[ad text:]" not in clean.lower()
    assert "http" not in clean.lower()
    assert "We act." in clean


def test_split_into_chunks_handles_empty_short_and_long_text() -> None:
    assert split_into_chunks("", min_words=2, max_words=10) == []
    assert split_into_chunks("Hi.", min_words=2, max_words=10) == []
    chunks = split_into_chunks(
        "We invest in clean energy. We invest in clean energy. This sentence is very long, and it should be split into smaller, cleaner pieces for parsing reliability.",
        min_words=2,
        max_words=12,
    )
    assert chunks
    assert len(chunks) >= 2
    assert len(chunks) == len(set(chunk.lower() for chunk in chunks))


def test_build_chunk_table_does_not_crash_with_noisy_rows() -> None:
    source = pd.DataFrame(
        {
            "source_row_index": [0, 1, 2],
            "source_id": ["a", "b", "c"],
            "source_doc_id": ["a", "b", "c"],
            "raw_text": ["[audio:] We reduce emissions.", "", None],
        }
    )
    sentences, mapping = build_chunk_table(source, min_words=2, max_words=10)
    assert len(sentences) == len(mapping)
    assert set(sentences.columns) >= {"sentence_id", "doc_id", "sentence"}
