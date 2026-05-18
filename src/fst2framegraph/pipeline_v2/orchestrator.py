from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .chunking import build_chunk_table
from .extract import run_fst_extraction
from .graph_and_analysis import build_event_graph, run_analysis_outputs
from .input_schema import InputColumns, load_input_csv
from .materialize import materialize_outputs
from .preflight import run_preflight
from .report import utc_timestamp_slug, write_run_summary


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def _build_sentence_rows_without_chunking(source_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sentence_df = pd.DataFrame(
        {
            "sentence_id": source_df["source_id"].astype(str),
            "doc_id": source_df["source_doc_id"].astype(str),
            "sentence": source_df["raw_text"].fillna("").astype(str),
            "source_id": source_df["source_id"].astype(str),
            "source_doc_id": source_df["source_doc_id"].astype(str),
            "source_row_index": source_df["source_row_index"].astype(int),
        }
    )
    mapping_df = pd.DataFrame(
        {
            "source_id": source_df["source_id"].astype(str),
            "source_doc_id": source_df["source_doc_id"].astype(str),
            "source_row_index": source_df["source_row_index"].astype(int),
            "sentence_id": source_df["source_id"].astype(str),
            "unique_chunk_id": source_df["source_id"].astype(str),
            "chunk_index": 0,
            "chunk_text": source_df["raw_text"].fillna("").astype(str),
        }
    )
    return sentence_df, mapping_df


def run_fst2graph(
    *,
    input_csv: str | Path,
    out_root: str | Path = "outputs",
    text_col: str | None = None,
    id_col: str | None = None,
    doc_col: str | None = None,
    metadata_cols: list[str] | None = None,
    framebase_index: str | Path | None = None,
    framebase_dir: str | Path | None = None,
    resume: bool = True,
    batch_size: int = 16,
    dedupe: bool = True,
    dedupe_normalise: str = "exact",
    checkpoint_every: int = 100,
    device: str = "auto",
    chunk_text: bool = True,
    chunk_min_words: int = 2,
    chunk_max_words: int = 70,
    top_n_frames: int = 20,
    top_n_agents: int = 30,
    min_count: int = 2,
    n_communities: int = 5,
    random_seed: int = 42,
    run_name: str | None = None,
    fst: Any | None = None,
) -> dict[str, Any]:
    """Run the end-to-end product pipeline in one call.

    The output root contains one timestamped run directory with canonical FST
    outputs, event graph outputs, analysis tables, and run summaries.
    """
    _set_random_seed(random_seed)
    preflight = run_preflight(fst=fst, require_fst=fst is None, apply_env_guards=True)

    source_df, resolved_cols = load_input_csv(
        input_csv,
        text_col=text_col,
        id_col=id_col,
        doc_col=doc_col,
        metadata_cols=metadata_cols,
    )
    if chunk_text:
        sentence_df, mapping_df = build_chunk_table(
            source_df,
            min_words=chunk_min_words,
            max_words=chunk_max_words,
        )
    else:
        sentence_df, mapping_df = _build_sentence_rows_without_chunking(source_df)

    input_path = Path(input_csv)
    if run_name:
        run_id = run_name
    elif resume:
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", input_path.stem).strip("._-") or "input"
        run_id = f"run_{stem}"
    else:
        run_id = f"run_{utc_timestamp_slug()}"
    run_root = Path(out_root) / run_id
    run_dir = run_root / "fst_clean"
    graph_out_dir = run_root / "graph"
    analysis_out_dir = run_root / "analysis"
    run_dir.mkdir(parents=True, exist_ok=True)
    graph_out_dir.mkdir(parents=True, exist_ok=True)
    analysis_out_dir.mkdir(parents=True, exist_ok=True)

    source_df.to_csv(run_root / "source_rows.csv", index=False)
    sentence_df.to_csv(run_root / "sentence_rows.csv", index=False)
    mapping_df.to_csv(run_root / "chunk_mapping.csv", index=False)

    extraction_report = run_fst_extraction(
        sentences_df=sentence_df,
        run_dir=run_dir,
        fst=fst,
        resume=resume,
        checkpoint_every=checkpoint_every,
        batch_size=batch_size,
        device=device,
        dedupe=dedupe,
        dedupe_normalise=dedupe_normalise,
    )
    materialized_report = materialize_outputs(run_dir=run_dir, chunk_mapping=mapping_df)

    graph, graph_report = build_event_graph(
        run_dir=run_dir,
        graph_out_dir=graph_out_dir,
        include_sentence_nodes=True,
    )
    analysis_report = run_analysis_outputs(
        graph=graph,
        analysis_out_dir=analysis_out_dir,
        top_n_frames=top_n_frames,
        top_n_agents=top_n_agents,
        min_count=min_count,
        n_communities=n_communities,
    )

    payload: dict[str, Any] = {
        "run_id": run_id,
        "input_csv": str(input_path),
        "resolved_columns": {
            "text_col": resolved_cols.text_col,
            "id_col": resolved_cols.id_col,
            "doc_col": resolved_cols.doc_col,
            "metadata_cols": list(resolved_cols.metadata_cols),
        },
        "input_rows": int(len(source_df)),
        "chunk_rows": int(len(sentence_df)),
        "run_root": str(run_root),
        "run_dir": str(run_dir),
        "graph_out_dir": str(graph_out_dir),
        "analysis_out_dir": str(analysis_out_dir),
        "preflight": {
            "ok": preflight.ok,
            "python_version": preflight.python_version,
            "in_colab": preflight.in_colab,
            "fst_available": preflight.fst_available,
            "protobuf_version": preflight.protobuf_version,
            "env_guards_applied": preflight.env_guards_applied,
            "warnings": preflight.warnings,
        },
        "extraction_report": extraction_report,
        "materialized_report": materialized_report,
        "graph_report": graph_report,
        "analysis_report": analysis_report,
        "framebase_index": str(framebase_index) if framebase_index else None,
        "framebase_dir": str(framebase_dir) if framebase_dir else None,
    }
    summary_path = write_run_summary(out_dir=run_root, payload=payload)
    payload["summary_path"] = str(summary_path)
    return payload


__all__ = ["run_fst2graph", "InputColumns"]
