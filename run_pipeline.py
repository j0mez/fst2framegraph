from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fst2framegraph import AnalysisBase, FrameGraphBuilder, encode_with_fst, from_fst_output
from fst2framegraph.io.transcripts import clean_transcript


DEFAULT_TEXT_COLUMNS = [
    "Transcript (text and audio)",
    "transcript",
    "ad_text",
    "text",
    "sentence",
]
DEFAULT_ID_COLUMNS = ["Advert ID", "advert_id", "ad_id", "sentence_id", "id"]
DEFAULT_DOC_COLUMNS = ["doc_id", "document_id", "Advert ID", "advert_id", "ad_id", "id"]


@dataclass
class _SimpleFrameElement:
    name: str
    text: str


@dataclass
class _SimpleFrame:
    name: str
    trigger_location: int
    frame_elements: list[_SimpleFrameElement]


@dataclass
class _SimpleResult:
    sentence: str
    frames: list[_SimpleFrame]


class _RuleBasedFallbackFST:
    """Small offline backend for smoke tests and dependency failures."""

    _TRIGGERS = {
        "invest": "Investing",
        "protect": "Protection",
        "report": "Reporting",
        "support": "Supporting",
        "help": "Assistance",
        "create": "Creating",
        "reduce": "Reduction",
        "deliver": "Delivery",
    }

    def detect_frames(self, sentence: str) -> _SimpleResult:
        lower = sentence.lower()
        frames: list[_SimpleFrame] = []
        agent = _guess_agent(sentence)
        for trigger, frame_name in self._TRIGGERS.items():
            index = lower.find(trigger)
            if index < 0:
                continue
            elements = [_SimpleFrameElement("Agent", agent)] if agent else []
            frames.append(_SimpleFrame(frame_name, index, elements))
        return _SimpleResult(sentence=sentence, frames=frames)


def _guess_agent(sentence: str) -> str:
    words = [part.strip(".,;:!?()[]\"'") for part in sentence.split()]
    for word in words[:6]:
        if word.lower() in {"we", "our", "they", "people", "families", "businesses"}:
            return word
    return words[0] if words else ""


def _make_runtime_fst(*, require_real_fst: bool = False) -> Any:
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from frame_semantic_transformer import FrameSemanticTransformer
    except Exception as exc:
        if require_real_fst:
            raise RuntimeError(
                "frame-semantic-transformer is not available. In Colab, run "
                "`python scripts/install_colab_fst.py` from the project root, then retry."
            ) from exc
        warnings.warn(
            "frame-semantic-transformer is not available; using the offline rule-based "
            "fallback backend. Install the FST stack for production semantic parsing.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _RuleBasedFallbackFST()
    return FrameSemanticTransformer()


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    lower = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        matched = lower.get(candidate.lower())
        if matched is not None:
            return matched
    return None


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _prepare_text_rows(
    csv_path: Path,
    *,
    text_col: str | None,
    id_col: str | None,
    doc_col: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = pd.read_csv(csv_path)
    if source.empty:
        raise ValueError(f"Input CSV has no rows: {csv_path}")

    columns = list(source.columns)
    resolved_text_col = text_col or _first_existing(columns, DEFAULT_TEXT_COLUMNS)
    if resolved_text_col is None:
        raise ValueError(
            "Could not find a transcript/text column. Pass --text-col or include one of: "
            + ", ".join(DEFAULT_TEXT_COLUMNS)
        )
    if resolved_text_col not in source.columns:
        raise ValueError(f"text_col not found in CSV: {resolved_text_col}")

    resolved_id_col = id_col or _first_existing(columns, DEFAULT_ID_COLUMNS)
    resolved_doc_col = doc_col or _first_existing(columns, DEFAULT_DOC_COLUMNS)

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row_index, row in source.iterrows():
        cleaned = clean_transcript(row.get(resolved_text_col))
        source_id = (
            str(row.get(resolved_id_col))
            if resolved_id_col is not None and pd.notna(row.get(resolved_id_col))
            else f"row_{row_index}"
        )
        doc_id = (
            str(row.get(resolved_doc_col))
            if resolved_doc_col is not None and pd.notna(row.get(resolved_doc_col))
            else source_id
        )
        if not cleaned:
            skipped.append(
                {
                    "row_index": int(row_index),
                    "source_id": source_id,
                    "reason": "empty_after_clean_transcript",
                }
            )
            continue
        metadata = {
            str(column): row.get(column)
            for column in source.columns
            if column not in {resolved_text_col}
        }
        rows.append(
            {
                "sentence_id": source_id,
                "doc_id": doc_id,
                "sentence": cleaned,
                "source_row_index": int(row_index),
                "metadata_json": json.dumps(metadata, default=str, sort_keys=True),
            }
        )

    if skipped:
        warnings.warn(
            f"Skipped {len(skipped)} row(s) with empty transcript text after cleaning.",
            RuntimeWarning,
            stacklevel=2,
        )
    if not rows:
        raise ValueError("No usable ad text was produced after transcript cleaning.")

    prepared = pd.DataFrame(rows)
    report = {
        "rows_in": int(len(source)),
        "rows_prepared": int(len(prepared)),
        "rows_skipped_empty": int(len(skipped)),
        "skipped_rows": skipped,
        "text_col": resolved_text_col,
        "id_col": resolved_id_col,
        "doc_col": resolved_doc_col,
    }
    return prepared, report


def run_pipeline(
    csv_path: str | Path,
    *,
    output_root: str | Path = "pipeline_outputs",
    fst: Any | None = None,
    text_col: str | None = None,
    id_col: str | None = None,
    doc_col: str | None = None,
    timestamp: str | None = None,
    min_count: int = 2,
    require_real_fst: bool = False,
) -> dict[str, Any]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    run_stamp = timestamp or _timestamp()
    output_dir = Path(output_root) / f"fst2framegraph_{run_stamp}"
    clean_dir = output_dir / "fst_clean"
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared, prep_report = _prepare_text_rows(
        csv_path,
        text_col=text_col,
        id_col=id_col,
        doc_col=doc_col,
    )
    prepared_csv = output_dir / "prepared_text.csv"
    prepared.to_csv(prepared_csv, index=False)

    runtime_fst = fst if fst is not None else _make_runtime_fst(require_real_fst=require_real_fst)
    try:
        encode_report = encode_with_fst(
            fst=runtime_fst,
            data=prepared,
            sentence_col="sentence",
            sentence_id_col="sentence_id",
            doc_col="doc_id",
            metadata_cols=["source_row_index", "metadata_json"],
            out_dir=clean_dir,
            resume=False,
            batch_size=8,
            dedupe=False,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Frame Semantic Transformer is not installed. Install with "
            "`pip install --find-links=wheels/ -e .` from the project root, "
            "or pass an initialized fst object to run_pipeline()."
        ) from exc

    documents = from_fst_output(clean_dir / "fst_clean.jsonl")
    graph = FrameGraphBuilder().build_graph(documents)
    graph_path = output_dir / "frame_graph.graphml"
    FrameGraphBuilder().save_graph(graph, graph_path)

    analysis = AnalysisBase(graph)
    lift = analysis.agent_frame_lift(top_n_frames=20, top_n_agents=30, min_count=min_count)
    lift_path = output_dir / "agent_frame_lift.csv"
    lift.to_csv(lift_path, index=False)

    communities = analysis.agent_frame_communities(n_communities=5)
    communities_path = output_dir / "agent_frame_communities.json"
    communities_path.write_text(json.dumps(communities, indent=2, sort_keys=True), encoding="utf-8")

    result = {
        **prep_report,
        "output_dir": str(output_dir),
        "prepared_csv": str(prepared_csv),
        "clean_dir": str(clean_dir),
        "graph_path": str(graph_path),
        "lift_path": str(lift_path),
        "communities_path": str(communities_path),
        "documents": int(len(documents)),
        "graph_nodes": int(graph.number_of_nodes()),
        "graph_edges": int(graph.number_of_edges()),
        "lift_rows": int(len(lift)),
        "encode_report": encode_report,
    }
    summary_path = output_dir / "summary_report.txt"
    summary_path.write_text(_format_summary(result), encoding="utf-8")
    result["summary_path"] = str(summary_path)
    return result


def _format_summary(result: dict[str, Any]) -> str:
    lines = [
        "fst2framegraph pipeline summary",
        "",
        f"Rows in: {result['rows_in']}",
        f"Prepared rows: {result['rows_prepared']}",
        f"Skipped empty rows: {result['rows_skipped_empty']}",
        f"Documents: {result['documents']}",
        f"Graph nodes: {result['graph_nodes']}",
        f"Graph edges: {result['graph_edges']}",
        f"Agent-frame lift rows: {result['lift_rows']}",
        "",
        f"GraphML: {result['graph_path']}",
        f"Lift CSV: {result['lift_path']}",
        f"Communities JSON: {result['communities_path']}",
    ]
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CSV -> FST -> frame graph analysis.")
    parser.add_argument("csv_path", type=Path, help="Input CSV path.")
    parser.add_argument("--out", type=Path, default=Path("pipeline_outputs"), help="Output root.")
    parser.add_argument("--text-col", default=None, help="Transcript/text column name.")
    parser.add_argument("--id-col", default=None, help="Source row identifier column.")
    parser.add_argument("--doc-col", default=None, help="Document identifier column.")
    parser.add_argument("--min-count", type=int, default=2, help="Minimum pair count for lift.")
    parser.add_argument(
        "--require-real-fst",
        action="store_true",
        help="Fail instead of using the offline fallback when frame-semantic-transformer is missing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = run_pipeline(
        args.csv_path,
        output_root=args.out,
        text_col=args.text_col,
        id_col=args.id_col,
        doc_col=args.doc_col,
        min_count=args.min_count,
        require_real_fst=args.require_real_fst,
    )
    print(_format_summary(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
