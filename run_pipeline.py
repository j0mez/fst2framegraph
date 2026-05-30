from __future__ import annotations
# ruff: noqa: E402

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fst2framegraph import AnalysisBase, FrameGraphBuilder, encode_with_fst, from_fst_output
from fst2framegraph.framebase.download import find_framebase_files
from fst2framegraph.framebase.index import (
    find_framebase_index,
    load_dbp_labels_from_index,
    load_rules_from_index,
    load_schema_from_index,
)
from fst2framegraph.framebase.load_dbp_labels import load_dbp_labels
from fst2framegraph.framebase.load_schema import FrameBaseSchema
from fst2framegraph.framebase.parse_dered_rules import parse_dered_rules
from fst2framegraph.framebase.parse_spin_rules import parse_spin_dereification_rules
from fst2framegraph.framebase.rule_index import RuleIndex
from fst2framegraph.graph.build_dereified import build_dereified_edges
from fst2framegraph.graph.build_nested import build_nested_edges
from fst2framegraph.graph.build_reified import build_reified_tables
from fst2framegraph.graph.export_graph import build_sentence_graphs, write_graphml
from fst2framegraph.io.column_detection import detect_columns
from fst2framegraph.io.transcripts import clean_transcript
from fst2framegraph.io.write_outputs import ensure_out_dir, write_csv, write_json, write_jsonl
from fst2framegraph.io.web_artifact import write_web_artifact
from fst2framegraph.qc.ambiguity_report import repeated_frame_warnings
from fst2framegraph.qc.coverage_report import make_qc_report


DEFAULT_TEXT_COLUMNS = [
    "Transcript (text and audio)",
    "transcript",
    "ad_text",
    "text",
    "sentence",
]
DEFAULT_ID_COLUMNS = ["Advert ID", "advert_id", "ad_id", "sentence_id", "id"]
DEFAULT_DOC_COLUMNS = ["doc_id", "document_id", "Advert ID", "advert_id", "ad_id", "id"]
DEFAULT_FRAMEBASE_DIR = Path("data") / "framebase"
FST_BACKEND_ENV = {
    "USE_TF": "0",
    "TRANSFORMERS_NO_TF": "1",
    "USE_FLAX": "0",
    "TOKENIZERS_PARALLELISM": "false",
}


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


@dataclass(frozen=True)
class _FrameBaseRuntime:
    schema: FrameBaseSchema
    dbp_labels: dict[str, str]
    dereification_rules: list[Any]
    source: str
    framebase_dir: Path | None
    framebase_core: Path | None
    framebase_index: Path | None
    dbp_labels_path: Path | None
    dereification_rules_path: Path | None


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


def _apply_fst_backend_env() -> None:
    for name, value in FST_BACKEND_ENV.items():
        os.environ.setdefault(name, value)


def _make_runtime_fst(*, require_real_fst: bool = False) -> Any:
    _apply_fst_backend_env()
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


def _optional_path(value: str | Path | None) -> Path | None:
    return Path(value) if value is not None else None


def _schema_has_entries(schema: FrameBaseSchema) -> bool:
    return bool(schema.frame_lookup or schema.fe_lookup)


def _resolve_framebase_runtime(
    *,
    framebase_dir: str | Path | None,
    framebase_core: str | Path | None,
    framebase_index: str | Path | None,
) -> _FrameBaseRuntime:
    """Load FrameBase schema before running any expensive FST work."""
    resolved_dir = _optional_path(framebase_dir)
    explicit_core = _optional_path(framebase_core)
    explicit_index = _optional_path(framebase_index)

    found = find_framebase_files(resolved_dir) if resolved_dir is not None else {}
    if explicit_index is not None:
        resolved_index = explicit_index
    elif explicit_core is not None:
        resolved_index = None
    else:
        resolved_index = find_framebase_index(resolved_dir)
    resolved_core = explicit_core or found.get("core_schema")

    if resolved_index is not None:
        if not resolved_index.exists():
            raise RuntimeError(f"FrameBase index does not exist: {resolved_index}")
        schema = load_schema_from_index(resolved_index)
        dbp_labels = load_dbp_labels_from_index(resolved_index)
        dereification_rules = load_rules_from_index(resolved_index)
        labels_path = None
        rules_path = None
        source = "index"
    elif resolved_core is not None:
        if not resolved_core.exists():
            raise RuntimeError(f"FrameBase core schema does not exist: {resolved_core}")
        schema = FrameBaseSchema.from_turtle(resolved_core)
        labels_path = found.get("dbp_labels")
        rules_path = found.get("dereification_rules_spin") or found.get("dereification_rules_sparql")
        dbp_labels = load_dbp_labels(labels_path)
        if rules_path and "spin" in rules_path.name.lower():
            dereification_rules = list(parse_spin_dereification_rules(rules_path, dbp_labels))
        else:
            dereification_rules = parse_dered_rules(rules_path, dbp_labels)
        source = "core"
    else:
        raise RuntimeError(
            "FrameBase schema is required before running the pipeline. Run "
            "`fst2framegraph setup-framebase --out data/framebase --build-index`, "
            "or pass --framebase-index /path/to/framebase_index.sqlite, "
            "--framebase-core /path/to/FrameBase_schema_core.ttl, or "
            "--framebase-dir /path/to/framebase."
        )

    if not _schema_has_entries(schema):
        raise RuntimeError(
            "FrameBase schema was found but no frames or frame elements could be loaded. "
            "Rebuild the index or pass a valid FrameBase core schema."
        )

    return _FrameBaseRuntime(
        schema=schema,
        source=source,
        framebase_dir=resolved_dir,
        framebase_core=resolved_core,
        framebase_index=resolved_index,
        dbp_labels=dbp_labels,
        dereification_rules=dereification_rules,
        dbp_labels_path=labels_path,
        dereification_rules_path=rules_path,
    )


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


def _write_framebase_outputs(
    *,
    clean_dir: Path,
    out_dir: Path,
    framebase: _FrameBaseRuntime,
    min_filler_len: int,
) -> dict[str, Any]:
    source_csv = clean_dir / "frame_elements_long.csv"
    if not source_csv.exists():
        raise RuntimeError(f"FST output missing required FrameBase input: {source_csv}")

    raw_df = pd.read_csv(source_csv)
    cmap = detect_columns(raw_df)
    ensure_out_dir(out_dir)

    documents, sentences, frame_instances, frame_elements, nodes, reified_edges = build_reified_tables(
        raw_df,
        cmap,
        framebase.schema,
        min_filler_len=min_filler_len,
    )
    nested_edges = build_nested_edges(frame_instances, frame_elements)
    rule_index = RuleIndex.from_rules(framebase.dereification_rules)
    dereified_edges, dereification_diagnostics, dereification_stats = build_dereified_edges(
        frame_instances,
        frame_elements,
        rule_index,
    )
    edges = pd.concat([reified_edges, nested_edges, dereified_edges], ignore_index=True, sort=False)
    warnings_list = repeated_frame_warnings(frame_instances)
    if not framebase.dereification_rules:
        warnings_list.append(
            "FrameBase dereification rules were not available; direct DBP edges were not generated."
        )

    write_csv(documents, out_dir, "documents.csv")
    write_csv(sentences, out_dir, "sentences.csv")
    write_csv(frame_instances, out_dir, "frame_instances.csv")
    write_csv(frame_elements, out_dir, "frame_elements.csv")
    write_csv(
        frame_elements.rename(
            columns={
                "fe_name": "element_name",
                "filler_text": "element_filler",
            }
        ),
        out_dir,
        "frame_elements_long.csv",
    )
    write_csv(nodes, out_dir, "graph_nodes.csv")
    write_csv(reified_edges, out_dir, "graph_edges_reified.csv")
    write_csv(nested_edges, out_dir, "graph_edges_nested.csv")
    write_csv(dereified_edges, out_dir, "graph_edges_dereified.csv")
    write_csv(dereified_edges, out_dir, "direct_edges.csv")
    write_csv(dereification_diagnostics, out_dir, "dereification_diagnostics.csv")
    write_csv(edges, out_dir, "edges.csv")

    if sentences.empty or frame_instances.empty or frame_elements.empty:
        sentence_graphs = []
    else:
        sentence_graphs = build_sentence_graphs(
            sentences,
            frame_instances,
            frame_elements,
            nested_edges,
            dereified_edges,
        )
    write_jsonl(sentence_graphs, out_dir, "sentence_graphs.jsonl")

    qc = make_qc_report(
        source_rows=len(raw_df),
        documents=documents,
        sentences=sentences,
        frame_instances=frame_instances,
        frame_elements=frame_elements,
        reified_edges=reified_edges,
        nested_edges=nested_edges,
        dereified_edges=dereified_edges,
        warnings=warnings_list,
    )
    qc_payload = qc.model_dump()
    write_json(qc_payload, out_dir, "qc_report.json")
    summary_payload = {
        **qc_payload,
        "framebase_source": framebase.source,
        "framebase_index_used": framebase.framebase_index is not None,
        "framebase_index_path": str(framebase.framebase_index) if framebase.framebase_index else None,
        "framebase_core_path": str(framebase.framebase_core) if framebase.framebase_core else None,
        "dbp_label_count": int(len(framebase.dbp_labels)),
        "dereification_rules_loaded": int(len(framebase.dereification_rules)),
        "dereification_rules_matched": int(dereification_stats["dereification_rules_matched"]),
        "dereification_rule_match_ambiguous": int(
            dereification_stats["dereification_rule_match_ambiguous"]
        ),
        "dereification_rule_match_unmatched": int(
            dereification_stats["dereification_rule_match_unmatched"]
        ),
        "dereification_opportunities": int(dereification_stats["dereification_opportunities"]),
        "official_framebase_reder_edges": int(len(dereified_edges)),
    }
    write_json(summary_payload, out_dir, "summary.json")
    manifest_payload = {
        "input": str(source_csv),
        "framebase_source": framebase.source,
        "framebase_dir": str(framebase.framebase_dir) if framebase.framebase_dir else None,
        "framebase_core": str(framebase.framebase_core) if framebase.framebase_core else None,
        "framebase_index": str(framebase.framebase_index) if framebase.framebase_index else None,
        "dbp_labels": str(framebase.dbp_labels_path) if framebase.dbp_labels_path else None,
        "dereification_rules": (
            str(framebase.dereification_rules_path) if framebase.dereification_rules_path else None
        ),
        "columns": cmap.model_dump(),
    }
    write_json(manifest_payload, out_dir, "manifest.json")
    write_web_artifact(
        out=out_dir,
        build_manifest=manifest_payload,
        build_summary=summary_payload,
        documents=documents,
        sentences=sentences,
        frame_instances=frame_instances,
        frame_elements=frame_elements,
        nested_edges=nested_edges,
        direct_edges=dereified_edges,
        dereification_diagnostics=dereification_diagnostics,
    )
    write_graphml(nodes, out_dir / "graph.graphml", reified_edges, nested_edges, dereified_edges)

    return {
        "framebase_reified_dir": str(out_dir),
        "framebase_source": framebase.source,
        "framebase_core": str(framebase.framebase_core) if framebase.framebase_core else None,
        "framebase_index": str(framebase.framebase_index) if framebase.framebase_index else None,
        "reified_documents": int(len(documents)),
        "reified_sentences": int(len(sentences)),
        "reified_frame_instances": int(len(frame_instances)),
        "reified_frame_elements": int(len(frame_elements)),
        "reified_edges": int(len(reified_edges)),
        "nested_edges": int(len(nested_edges)),
        "dereified_edges": int(len(dereified_edges)),
        "framebase_validated_frames": int(qc.framebase_validated_frames),
        "framebase_validated_frame_elements": int(qc.framebase_validated_frame_elements),
        "framebase_unmatched_frames": int(qc.framebase_unmatched_frames),
        "framebase_unmatched_frame_elements": int(qc.framebase_unmatched_frame_elements),
    }


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
    framebase_dir: str | Path | None = DEFAULT_FRAMEBASE_DIR,
    framebase_core: str | Path | None = None,
    framebase_index: str | Path | None = None,
    require_framebase: bool = True,
    min_filler_len: int = 1,
) -> dict[str, Any]:
    _apply_fst_backend_env()
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not require_framebase:
        raise RuntimeError("run_pipeline requires FrameBase; fallback IRIs are not supported.")

    framebase = _resolve_framebase_runtime(
        framebase_dir=framebase_dir,
        framebase_core=framebase_core,
        framebase_index=framebase_index,
    )

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
    framebase_report = _write_framebase_outputs(
        clean_dir=clean_dir,
        out_dir=output_dir / "reified",
        framebase=framebase,
        min_filler_len=min_filler_len,
    )

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
        **framebase_report,
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
        f"FrameBase reified edges: {result['reified_edges']}",
        f"FrameBase dereified/direct edges: {result['dereified_edges']}",
        f"FrameBase validated frames: {result['framebase_validated_frames']}",
        f"FrameBase validated frame elements: {result['framebase_validated_frame_elements']}",
        "",
        f"GraphML: {result['graph_path']}",
        f"Lift CSV: {result['lift_path']}",
        f"Communities JSON: {result['communities_path']}",
        f"FrameBase reified dir: {result['framebase_reified_dir']}",
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
        "--framebase-dir",
        type=Path,
        default=DEFAULT_FRAMEBASE_DIR,
        help="Directory containing FrameBase files or framebase_index.sqlite.",
    )
    parser.add_argument("--framebase-core", type=Path, default=None, help="FrameBase core schema TTL path.")
    parser.add_argument("--framebase-index", type=Path, default=None, help="Prebuilt FrameBase SQLite index path.")
    parser.add_argument("--min-filler-len", type=int, default=1, help="Minimum filler length for reified output.")
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
        framebase_dir=args.framebase_dir,
        framebase_core=args.framebase_core,
        framebase_index=args.framebase_index,
        min_filler_len=args.min_filler_len,
    )
    print(_format_summary(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
