from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fst2framegraph.fst import encode_with_fst


def run_fst_extraction(
    *,
    sentences_df: pd.DataFrame,
    run_dir: str | Path,
    fst: Any | None = None,
    resume: bool = True,
    checkpoint_every: int = 100,
    batch_size: int = 16,
    device: str = "auto",
    dedupe: bool = True,
    dedupe_normalise: str = "exact",
) -> dict[str, Any]:
    required = {"sentence_id", "doc_id", "sentence"}
    missing = sorted(required - set(sentences_df.columns))
    if missing:
        raise ValueError(f"sentences_df is missing required columns: {missing}")

    metadata_cols = [
        col
        for col in sentences_df.columns
        if col not in {"sentence_id", "doc_id", "sentence"}
    ]

    return encode_with_fst(
        fst=fst,
        data=sentences_df,
        sentence_col="sentence",
        sentence_id_col="sentence_id",
        doc_col="doc_id",
        metadata_cols=metadata_cols,
        out_dir=Path(run_dir),
        resume=resume,
        checkpoint_every=checkpoint_every,
        batch_size=batch_size,
        device=device,
        dedupe=dedupe,
        dedupe_normalise=dedupe_normalise,
    )
