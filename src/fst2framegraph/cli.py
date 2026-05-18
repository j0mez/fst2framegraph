from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console

from fst2framegraph.framebase.download import (
    download_framebase_files,
    find_framebase_files,
    write_framebase_manifest,
)
from fst2framegraph.framebase.index import (
    build_framebase_index as build_framebase_index_file,
    find_framebase_index,
    inspect_rule_candidates,
    load_dbp_labels_from_index,
    load_rules_from_index,
    load_schema_from_index,
)
from fst2framegraph.framebase.load_dbp_labels import load_dbp_labels
from fst2framegraph.framebase.load_schema import FrameBaseSchema
from fst2framegraph.framebase.parse_dered_rules import parse_dered_rules
from fst2framegraph.framebase.parse_spin_rules import parse_spin_dereification_rules
from fst2framegraph.framebase.rule_index import RuleIndex
from fst2framegraph.fst import encode_with_fst, materialise_run
from fst2framegraph.graph.build_dereified import build_dereified_edges
from fst2framegraph.graph.build_nested import build_nested_edges
from fst2framegraph.graph.build_reified import build_reified_tables
from fst2framegraph.graph.export_graph import build_sentence_graphs, write_graphml, write_turtle
from fst2framegraph.io.column_detection import detect_columns
from fst2framegraph.io.inspect_outputs import (
    FLAT_COLUMNS,
    GRAPH_READY_COLUMNS,
    convert_fst_outputs,
    doctor_run,
    inspect_fst_outputs,
)
from fst2framegraph.io.read_fst import read_fst_csv
from fst2framegraph.io.write_outputs import ensure_out_dir, write_csv, write_json, write_jsonl
from fst2framegraph.pipeline_v2 import run_fst2graph
from fst2framegraph.qc.ambiguity_report import repeated_frame_warnings
from fst2framegraph.qc.coverage_report import make_qc_report
from fst2framegraph.qc.validation import require_file
from fst2framegraph.schema import ColumnMap

app = typer.Typer(help="Convert FrameNet-style parser output into FrameBase-compatible graphs.")
console = Console()
WS_RE = re.compile(r"\s+")

CANONICAL_RUN_FILES = [
    "fst_clean.jsonl",
    "progress.sqlite",
    "sentences.csv",
    "frame_instances.csv",
    "frame_elements.csv",
    "frame_elements_long.csv",
    "errors.csv",
    "extraction_report.json",
    "extraction_report.md",
    "manifest.json",
]


def _resolve_framebase_paths(
    framebase_dir: Path | None,
    framebase_core: Path | None,
    dbp_labels: Path | None,
    dered_rules: Path | None,
    framebase_index: Path | None = None,
) -> tuple[Path | None, Path | None, Path | None, dict[str, str | None]]:
    found = find_framebase_files(framebase_dir) if framebase_dir is not None else {}
    core = framebase_core or found.get("core_schema")
    labels = dbp_labels or found.get("dbp_labels")
    rules = (
        dered_rules
        or found.get("dereification_rules_spin")
        or found.get("dereification_rules_sparql")
    )
    return core, labels, rules, {
        "framebase_dir": str(framebase_dir) if framebase_dir else None,
        "framebase_core": str(core) if core else None,
        "dbp_labels": str(labels) if labels else None,
        "dered_rules": str(rules) if rules else None,
        "framebase_index": str(framebase_index) if framebase_index else None,
    }


def _resolve_framebase_index(
    framebase_dir: Path | None,
    framebase_index: Path | None,
) -> Path | None:
    if framebase_index is not None:
        return framebase_index
    return find_framebase_index(framebase_dir)


def _resolve_build_input(input_path: Path) -> Path:
    if input_path.is_dir():
        csv_path = input_path / "frame_elements_long.csv"
        if csv_path.exists():
            return csv_path
        if (input_path / "fst_clean.jsonl").exists():
            materialise_run(input_path)
            if csv_path.exists():
                return csv_path
        raise ValueError(
            "Input directory does not contain frame_elements_long.csv or fst_clean.jsonl. "
            f"Try `fst2framegraph inspect --input {input_path}`."
        )
    return input_path


def _clear_canonical_outputs(out: Path) -> None:
    for name in CANONICAL_RUN_FILES:
        path = out / name
        if path.exists() and path.is_file():
            path.unlink()


def _files_written(out: Path) -> list[str]:
    return [str(out / name) for name in CANONICAL_RUN_FILES if (out / name).exists()]


def _prepare_build_command(out: Path) -> str:
    return (
        "fst2framegraph build "
        f"--input {out} "
        "--out graph_output "
        "--framebase-index PATH/framebase_index.sqlite"
    )


def _run_command(
    *,
    input: Path,
    out: Path,
    graph: bool = False,
    graph_out: Path | None = None,
    framebase_index: Path | None = None,
    framebase_dir: Path | None = None,
    text_col: str = "sentence",
    id_col: str | None = None,
    doc_col: str | None = None,
    allow_pickle: bool = False,
    resume: bool = True,
    checkpoint_every: int = 100,
    batch_size: int = 16,
    device: str = "auto",
    dedupe: bool = True,
    dedupe_normalise: str = "exact",
    chunk_text: bool = True,
    chunk_min_words: int = 2,
    chunk_max_words: int = 70,
    plan: bool = False,
    yes: bool = False,
    interactive: bool = False,
) -> str:
    parts = ["fst2framegraph run", "--input", str(input), "--out", str(out)]
    if graph:
        parts.append("--graph")
    if graph_out is not None:
        parts.extend(["--graph-out", str(graph_out)])
    if framebase_index is not None:
        parts.extend(["--framebase-index", str(framebase_index)])
    if framebase_dir is not None:
        parts.extend(["--framebase-dir", str(framebase_dir)])
    if text_col != "sentence":
        parts.extend(["--text-col", text_col])
    if id_col is not None:
        parts.extend(["--id-col", id_col])
    if doc_col is not None:
        parts.extend(["--doc-col", doc_col])
    if allow_pickle:
        parts.append("--allow-pickle")
    if not resume:
        parts.append("--no-resume")
    if checkpoint_every != 100:
        parts.extend(["--checkpoint-every", str(checkpoint_every)])
    if batch_size != 16:
        parts.extend(["--batch-size", str(batch_size)])
    if device != "auto":
        parts.extend(["--device", device])
    if not dedupe:
        parts.append("--no-dedupe")
    if dedupe_normalise != "exact":
        parts.extend(["--dedupe-normalise", dedupe_normalise])
    if not chunk_text:
        parts.append("--no-chunk-text")
    if chunk_min_words != 2:
        parts.extend(["--chunk-min-words", str(chunk_min_words)])
    if chunk_max_words != 70:
        parts.extend(["--chunk-max-words", str(chunk_max_words)])
    if plan:
        parts.append("--dry-run")
    if yes:
        parts.append("--yes")
    if interactive:
        parts.append("--interactive")
    return " ".join(parts)


def _detect_next_command_for_input(
    input: Path,
    out: Path,
    text_col: str,
    id_col: str | None,
    doc_col: str | None,
) -> str:
    parts = [
        "fst2framegraph detect",
        "--input",
        str(input),
        "--text-col",
        text_col,
    ]
    if id_col is not None:
        parts.extend(["--id-col", id_col])
    if doc_col is not None:
        parts.extend(["--doc-col", doc_col])
    parts.extend(["--out", str(out), "--resume"])
    return " ".join(parts)


def _detect_next_command() -> str:
    return (
        "fst2framegraph detect "
        "--input sentences.csv "
        "--text-col sentence "
        "--id-col sentence_id "
        "--doc-col doc_id "
        "--out fst_clean "
        "--resume"
    )


def _looks_like_raw_text_table(path: Path, text_col: str) -> bool:
    if path.is_dir() or path.suffix.lower() != ".csv":
        return False
    try:
        import pandas as pd

        df = pd.read_csv(path, nrows=5)
    except Exception:
        return False
    columns = set(df.columns)
    if text_col not in columns:
        return False
    graph_signal_columns = set(GRAPH_READY_COLUMNS) - {"sentence_id", "sentence"}
    has_graph_columns = any(col in columns for col in graph_signal_columns)
    has_flat_columns = any(col in columns for col in FLAT_COLUMNS)
    return not has_graph_columns and not has_flat_columns


def _clean_input_text(text: object) -> str:
    value = "" if text is None else str(text)
    if not value:
        return ""
    value = re.sub(r"\[ad text:\]|\[audio transcript:\]|\[video transcript:\]|\[text:\]|\[audio:\]", " ", value, flags=re.I)
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    value = value.replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{2,}", "\n", value)
    return value.strip()


def _normalise_chunk_for_dedupe(text: str) -> str:
    return WS_RE.sub(" ", text).strip().lower()


def _stable_chunk_hash(text: str) -> str:
    key = _normalise_chunk_for_dedupe(text)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def _split_into_chunks(text: str, *, min_words: int, max_words: int) -> list[str]:
    text = _clean_input_text(text)
    if not text:
        return []

    rough_parts: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        rough_parts.extend(re.split(r"(?<=[.!?])\s+", line))

    chunks: list[str] = []
    for part in rough_parts:
        part = WS_RE.sub(" ", part).strip()
        if not part:
            continue
        words = part.split()
        if len(words) < min_words:
            continue
        if len(words) <= max_words:
            chunks.append(part)
            continue
        subparts = re.split(r"(?<=[,;:])\s+", part)
        buffer: list[str] = []
        for sub in subparts:
            sub_words = sub.split()
            if len(buffer) + len(sub_words) <= max_words:
                buffer.extend(sub_words)
            else:
                if len(buffer) >= min_words:
                    chunks.append(" ".join(buffer).strip())
                buffer = sub_words
        if len(buffer) >= min_words:
            chunks.append(" ".join(buffer).strip())

    # De-duplicate only within one source row.
    seen: set[str] = set()
    deduped: list[str] = []
    for chunk in chunks:
        key = _normalise_chunk_for_dedupe(chunk)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)
    return deduped


def _build_chunked_sentence_table(
    *,
    input_path: Path,
    text_col: str,
    id_col: str | None,
    doc_col: str | None,
    min_chunk_words: int,
    max_chunk_words: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(input_path)
    for required in [text_col]:
        if required not in source.columns:
            raise ValueError(f"Missing required column for text chunking: {required}")

    rows: list[dict[str, object]] = []
    mapping_rows: list[dict[str, object]] = []
    global_index = 0
    for row_index, row in source.iterrows():
        raw_text = row.get(text_col)
        chunks = _split_into_chunks(
            raw_text,
            min_words=min_chunk_words,
            max_words=max_chunk_words,
        )
        if not chunks:
            continue
        raw_sentence_id = str(row.get(id_col) if id_col and id_col in source.columns else f"row_{row_index}")
        raw_doc_id = str(row.get(doc_col) if doc_col and doc_col in source.columns else raw_sentence_id)
        for chunk_index, chunk in enumerate(chunks):
            chunk_id = f"{raw_sentence_id}__chunk_{chunk_index:03d}"
            unique_chunk_id = _stable_chunk_hash(chunk)
            rows.append(
                {
                    "sentence_id": chunk_id,
                    "doc_id": raw_doc_id,
                    "sentence": chunk,
                }
            )
            mapping_rows.append(
                {
                    "row_index": int(row_index),
                    "source_sentence_id": raw_sentence_id,
                    "source_doc_id": raw_doc_id,
                    "chunk_index": chunk_index,
                    "sentence_id": chunk_id,
                    "unique_chunk_id": unique_chunk_id,
                    "chunk_text": chunk,
                    "chunk_word_count": len(chunk.split()),
                    "global_chunk_row": global_index,
                }
            )
            global_index += 1

    if not rows:
        return pd.DataFrame(columns=["sentence_id", "doc_id", "sentence"]), pd.DataFrame()
    return pd.DataFrame(rows), pd.DataFrame(mapping_rows)


def _plan_label(inspection: dict[str, object], *, raw_text: bool) -> str:
    if raw_text:
        return "raw sentence CSV"
    detected = str(inspection.get("detected_format"))
    return {
        "graph_ready_csv": "graph-ready CSV",
        "fst_jsonl": "convertible JSONL",
        "fst_json": "convertible JSON",
        "v0.3_run_directory": "canonical run directory",
        "pickle_folder": "pickle folder",
        "pickle_file": "pickle file",
        "flattened_csv": "flat-only CSV" if inspection.get("flat_only") else "flattened CSV",
    }.get(detected, detected.replace("_", " "))


def _print_run_plan(
    *,
    input: Path,
    out: Path,
    graph_out: Path | None,
    framebase_index: Path | None,
    framebase_dir: Path | None,
    inspection: dict[str, object],
    raw_text: bool,
    text_col: str,
    id_col: str | None,
    doc_col: str | None,
    allow_pickle: bool,
    resume: bool,
    checkpoint_every: int,
    batch_size: int,
    device: str,
    dedupe: bool,
    dedupe_normalise: str,
    chunk_text: bool,
    chunk_min_words: int,
    chunk_max_words: int,
) -> None:
    detected = _plan_label(inspection, raw_text=raw_text)
    actions: list[str] = []
    if raw_text:
        actions.extend(
            [
                f"run FrameSemanticTransformer into canonical run directory: {out}",
                (
                    f"chunk long text rows into sentence-like chunks "
                    f"({chunk_min_words}-{chunk_max_words} words)"
                    if chunk_text
                    else "do not split text rows into chunks"
                ),
                "dedupe identical input texts before FST inference" if dedupe else "run FST once per input row",
                "materialise CSV/report outputs",
                "run doctor checks",
            ]
        )
    elif inspection.get("detected_format") == "v0.3_run_directory":
        actions.extend(["materialise CSV/report outputs if needed", "run doctor checks"])
    elif inspection.get("convertible") or (
        allow_pickle and inspection.get("detected_format") in {"pickle_file", "pickle_folder"}
    ):
        actions.extend(
            [
                f"convert input to canonical run directory: {out}",
                "materialise CSV/report outputs",
                "run doctor checks",
            ]
        )
    elif inspection.get("flat_only"):
        actions.append("stop: flat-only data cannot support reliable nested graph building")
    else:
        actions.append("stop: inspect output is insufficient for an automatic workflow")

    console.print(f"Detected: {detected}\n")
    console.print("Planned actions:")
    for i, action in enumerate(actions, start=1):
        console.print(f"  {i}. {action}")

    if graph_out and (framebase_index or framebase_dir):
        console.print(f"\nGraph build: planned into {graph_out}.")
    else:
        console.print(
            "\nGraph build: skipped because --graph-out and --framebase-index were not provided."
        )

    console.print("\nTo execute:")
    console.print(
        "  "
        + _run_command(
            input=input,
            out=out,
            graph=graph_out is not None,
            graph_out=graph_out,
            framebase_index=framebase_index,
            framebase_dir=framebase_dir,
            text_col=text_col,
            id_col=id_col,
            doc_col=doc_col,
            allow_pickle=allow_pickle,
            resume=resume,
            checkpoint_every=checkpoint_every,
            batch_size=batch_size,
            device=device,
            dedupe=dedupe,
            dedupe_normalise=dedupe_normalise,
            chunk_text=chunk_text,
            chunk_min_words=chunk_min_words,
            chunk_max_words=chunk_max_words,
        )
    )


def _graph_build_requested(
    graph_out: Path | None,
    framebase_index: Path | None,
    framebase_dir: Path | None,
) -> bool:
    return graph_out is not None and (framebase_index is not None or framebase_dir is not None)


def _run_graph_build_if_requested(
    *,
    run_dir: Path,
    graph_out: Path | None,
    framebase_index: Path | None,
    framebase_dir: Path | None,
) -> dict[str, object] | None:
    if graph_out is None:
        return None
    index_path = _resolve_framebase_index(framebase_dir, framebase_index)
    if index_path is None and framebase_dir is not None:
        expected = framebase_dir / "framebase_index.sqlite"
        return {
            "status": "skipped",
            "message": "FrameBase index was not found in --framebase-dir.",
            "next_command": (
                "fst2framegraph build-framebase-index "
                f"--framebase-dir {framebase_dir} "
                f"--index {expected}"
            ),
        }
    if index_path is None:
        return {
            "status": "skipped",
            "message": "--graph-out was provided without --framebase-index or --framebase-dir.",
            "next_command": _prepare_build_command(run_dir),
        }
    build(
        input=run_dir,
        out=graph_out,
        framebase_dir=None,
        framebase_core=None,
        dbp_labels=None,
        dered_rules=None,
        framebase_index=index_path,
        require_framebase=False,
        doc_col=None,
        sentence_col=None,
        frame_col=None,
        fe_col=None,
        filler_col=None,
        sentence_id_col=None,
        frame_index_col=None,
        target_col=None,
        target_start_col=None,
        target_end_col=None,
        filler_start_col=None,
        filler_end_col=None,
        confidence_col=None,
        brand_col=None,
        year_col=None,
        min_filler_len=1,
        no_rdf=False,
        no_graphml=False,
    )
    return {"status": "built", "graph_out": str(graph_out), "framebase_index": str(index_path)}


@app.command("setup-framebase")
def setup_framebase(
    out: Path = typer.Option(Path("data/framebase"), "--out", help="Directory for FrameBase files."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Re-download existing files."),
    manifest_only: bool = typer.Option(
        False,
        "--manifest-only",
        help="Do not download; write/refresh manifest and checksums for files already present.",
    ),
    build_index: bool = typer.Option(
        False,
        "--build-index",
        help="Build or reuse a compact FrameBase SQLite index after setup.",
    ),
    index: Optional[Path] = typer.Option(None, "--index", help="Optional path for framebase_index.sqlite."),
) -> None:
    """Download or register the external FrameBase resources used by the converter."""
    if manifest_only:
        manifest = write_framebase_manifest(out)
    else:
        manifest = download_framebase_files(out, overwrite=overwrite)
    console.print(f"[green]FrameBase setup complete:[/green] {out}")
    console.print_json(data=manifest)
    if build_index:
        index_report = build_framebase_index_file(
            framebase_dir=out,
            index_path=index,
            overwrite=overwrite,
        )
        console.print("[green]FrameBase index ready:[/green]")
        console.print_json(data=index_report)


@app.command("build-framebase-index")
def build_framebase_index_command(
    framebase_dir: Path = typer.Option(
        Path("data/framebase"),
        "--framebase-dir",
        help="Directory containing FrameBase source files.",
    ),
    index: Optional[Path] = typer.Option(None, "--index", help="Output SQLite index path."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Rebuild an existing index."),
    framebase_core: Optional[Path] = typer.Option(None, help="FrameBase core schema TTL/Turtle gzip."),
    dbp_labels: Optional[Path] = typer.Option(
        None,
        help="FrameBase direct binary predicate labels TTL/Turtle gzip.",
    ),
    dered_rules: Optional[Path] = typer.Option(
        None,
        help="FrameBase dereification rules, SPARQL zip or text.",
    ),
    spin_limit: Optional[int] = typer.Option(
        None,
        "--spin-limit",
        help="Optional limit for SPIN rules parsed during indexing; useful for small probes.",
    ),
) -> None:
    """Build the compact FrameBase SQLite index used by normal graph builds."""
    def _progress(payload: dict[str, object]) -> None:
        console.print(
            "[cyan]"
            f"{payload.get('phase', 'spin')}[/cyan] "
            f"lines={payload.get('lines_processed', 0)} "
            f"constructs={payload.get('construct_nodes_seen', 0)} "
            f"candidates={payload.get('candidate_rules_extracted', 0)} "
            f"valid={payload.get('valid_rules_inserted', 0)} "
            f"warnings={payload.get('parse_warnings', 0)} "
            f"errors={payload.get('parse_errors', 0)}"
        )

    try:
        report = build_framebase_index_file(
            framebase_dir=framebase_dir,
            index_path=index,
            overwrite=overwrite,
            framebase_core=framebase_core,
            dbp_labels=dbp_labels,
            dered_rules=dered_rules,
            spin_limit=spin_limit,
            progress=_progress,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print_json(data=report)


@app.command()
def build(
    input: Path = typer.Option(
        ...,
        "--input",
        help="CSV file or canonical run directory.",
    ),
    out: Path = typer.Option(..., "--out", help="Output directory."),
    framebase_dir: Optional[Path] = typer.Option(
        None,
        "--framebase-dir",
        help="Directory containing current FrameBase 2.0 schema/rule files. Defaults to data/framebase if present.",
    ),
    framebase_core: Optional[Path] = typer.Option(None, help="FrameBase core schema TTL/Turtle gzip."),
    dbp_labels: Optional[Path] = typer.Option(None, help="FrameBase direct binary predicate labels TTL/Turtle gzip."),
    dered_rules: Optional[Path] = typer.Option(None, help="FrameBase dereification rules, preferably SPIN Turtle(.gz)."),
    framebase_index: Optional[Path] = typer.Option(
        None,
        "--framebase-index",
        help="Compact FrameBase SQLite index. Auto-discovered from --framebase-dir when present.",
    ),
    require_framebase: bool = typer.Option(
        False,
        "--require-framebase",
        help="Fail if neither a FrameBase index nor the three FrameBase source files are available.",
    ),
    doc_col: Optional[str] = typer.Option(None, help="Document/ad ID column."),
    sentence_col: Optional[str] = typer.Option(None, help="Sentence text column."),
    frame_col: Optional[str] = typer.Option(None, help="Frame name column."),
    fe_col: Optional[str] = typer.Option(None, help="Frame element name column."),
    filler_col: Optional[str] = typer.Option(None, help="Frame element filler text column."),
    sentence_id_col: Optional[str] = typer.Option(None, help="Optional sentence/chunk ID column."),
    frame_index_col: Optional[str] = typer.Option(None, help="Optional frame index column."),
    target_col: Optional[str] = typer.Option(None, help="Optional lexical target/trigger text column."),
    target_start_col: Optional[str] = typer.Option(None, help="Optional lexical target start character offset."),
    target_end_col: Optional[str] = typer.Option(None, help="Optional lexical target end character offset."),
    filler_start_col: Optional[str] = typer.Option(None, help="Optional filler start character offset."),
    filler_end_col: Optional[str] = typer.Option(None, help="Optional filler end character offset."),
    confidence_col: Optional[str] = typer.Option(None, help="Optional parser confidence column."),
    brand_col: Optional[str] = typer.Option(None, help="Optional brand/company column."),
    year_col: Optional[str] = typer.Option(None, help="Optional year column."),
    min_filler_len: int = typer.Option(1, help="Minimum filler length to keep."),
    no_rdf: bool = typer.Option(False, help="Do not write Turtle/RDF output."),
    no_graphml: bool = typer.Option(False, help="Do not write GraphML output."),
) -> None:
    try:
        input_csv = _resolve_build_input(input)
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    require_file(input_csv, "input")
    index_path = _resolve_framebase_index(framebase_dir, framebase_index)
    if framebase_index is None and any([framebase_core, dbp_labels, dered_rules]):
        index_path = None
    core_path, labels_path, rules_path, fb_paths = _resolve_framebase_paths(
        framebase_dir, framebase_core, dbp_labels, dered_rules, index_path
    )
    if require_framebase and index_path is None and not all([core_path, labels_path, rules_path]):
        missing = [
            k for k, v in fb_paths.items()
            if k not in {"framebase_dir", "framebase_index"} and v is None
        ]
        raise typer.BadParameter(
            "Missing required FrameBase files: "
            + ", ".join(missing)
            + ". Run `fst2framegraph setup-framebase --out data/framebase --build-index` "
            + "or pass explicit paths."
        )
    require_file(index_path, "FrameBase index")
    if index_path is None:
        require_file(core_path, "FrameBase core schema")
        require_file(labels_path, "DBP labels")
        require_file(rules_path, "dereification rules")
    ensure_out_dir(out)

    console.print(f"[bold]Reading[/bold] {input_csv}")
    raw_df, detected = read_fst_csv(input_csv)
    if any([doc_col, sentence_col, frame_col, fe_col, filler_col]):
        if not all([doc_col, sentence_col, frame_col, fe_col, filler_col]):
            raise typer.BadParameter("If setting explicit required columns, provide all five: doc, sentence, frame, FE, filler.")
        cmap = ColumnMap(
            doc_col=doc_col,
            sentence_col=sentence_col,
            frame_col=frame_col,
            fe_col=fe_col,
            filler_col=filler_col,
            sentence_id_col=sentence_id_col,
            frame_index_col=frame_index_col,
            target_col=target_col,
            target_start_col=target_start_col,
            target_end_col=target_end_col,
            filler_start_col=filler_start_col,
            filler_end_col=filler_end_col,
            confidence_col=confidence_col,
            brand_col=brand_col,
            year_col=year_col,
        )
    else:
        cmap = detected

    console.print(f"[bold]Using columns[/bold] {cmap.model_dump()}")

    console.print("[bold]Loading FrameBase schema[/bold]")
    if index_path:
        console.print(f"  index: {index_path}")
        schema = load_schema_from_index(index_path)
    elif core_path:
        console.print(f"  core: {core_path}")
        schema = FrameBaseSchema.from_turtle(core_path)
    else:
        console.print("  [yellow]no core schema supplied; using generated fallback IRIs[/yellow]")
        schema = FrameBaseSchema.empty()

    console.print("[bold]Building reified graph[/bold]")
    documents, sentences, frame_instances, frame_elements, nodes, reified_edges = build_reified_tables(
        raw_df, cmap, schema, min_filler_len=min_filler_len
    )

    warnings = repeated_frame_warnings(frame_instances)
    if index_path is None and core_path is None:
        warnings.append("FrameBase core schema was not supplied; frame/FE IRIs are generated fallback IRIs.")
    if index_path is None and labels_path is None:
        warnings.append("FrameBase DBP labels were not supplied; DBP labels fall back to IRI-derived labels.")
    if index_path is None and rules_path is None:
        warnings.append("FrameBase dereification rules were not supplied; no dereified DBP edges can be generated.")

    console.print("[bold]Building nested graph[/bold]")
    nested_edges = build_nested_edges(frame_instances, frame_elements)

    console.print("[bold]Loading FrameBase dereification rules[/bold]")
    if index_path:
        console.print(f"  index: {index_path}")
        labels = load_dbp_labels_from_index(index_path)
        rules = load_rules_from_index(index_path)
    else:
        if labels_path:
            console.print(f"  labels: {labels_path}")
        if rules_path:
            console.print(f"  rules: {rules_path}")
        labels = load_dbp_labels(labels_path)
        if rules_path and "spin" in rules_path.name.lower():
            rules = list(parse_spin_dereification_rules(rules_path, labels))
        else:
            rules = parse_dered_rules(rules_path, labels)
    rule_index = RuleIndex.from_rules(rules)
    if index_path and not rules:
        warnings.append(
            "DBP schema/labels are available, but dereification rules are not supplied."
        )
    elif rules_path and not rules:
        warnings.append("No dereification rules were parsed. Check the FrameBase rule file format.")
    if rules_path is None and labels_path is not None:
        warnings.append(
            "DBP schema/labels are available, but dereification rules are not supplied."
        )

    console.print(f"[bold]Parsed rules:[/bold] {len(rules)}")
    console.print("[bold]Building dereified graph[/bold]")
    dereified_edges, dereification_diagnostics, dereification_stats = build_dereified_edges(
        frame_instances, frame_elements, rule_index
    )

    console.print("[bold]Writing outputs[/bold]")
    write_csv(documents, out, "documents.csv")
    write_csv(sentences, out, "sentences.csv")
    write_csv(frame_instances, out, "frame_instances.csv")
    write_csv(frame_elements, out, "frame_elements.csv")
    write_csv(
        frame_elements.rename(
            columns={
                "fe_name": "element_name",
                "filler_text": "element_filler",
            }
        ),
        out,
        "frame_elements_long.csv",
    )
    write_csv(nodes, out, "graph_nodes.csv")
    write_csv(reified_edges, out, "graph_edges_reified.csv")
    write_csv(nested_edges, out, "graph_edges_nested.csv")
    write_csv(dereified_edges, out, "graph_edges_dereified.csv")
    write_csv(dereified_edges, out, "direct_edges.csv")
    write_csv(dereification_diagnostics, out, "dereification_diagnostics.csv")
    edges = pd.concat([reified_edges, nested_edges, dereified_edges], ignore_index=True, sort=False)
    write_csv(edges, out, "edges.csv")

    sentence_graphs = build_sentence_graphs(
        sentences, frame_instances, frame_elements, nested_edges, dereified_edges
    )
    write_jsonl(sentence_graphs, out, "sentence_graphs.jsonl")

    qc = make_qc_report(
        source_rows=len(raw_df),
        documents=documents,
        sentences=sentences,
        frame_instances=frame_instances,
        frame_elements=frame_elements,
        reified_edges=reified_edges,
        nested_edges=nested_edges,
        dereified_edges=dereified_edges,
        warnings=warnings,
    )
    qc_payload = qc.model_dump()
    summary_payload = {
        **qc_payload,
        "nested_edges": int(len(nested_edges)),
        "projected_fe_edges": 0,
        "official_framebase_reder_edges": int(len(dereified_edges)),
        "rule_pack_edges": 0,
        "dereification_rules_loaded": int(len(rules)),
        "dereification_rules_matched": int(dereification_stats["dereification_rules_matched"]),
        "dereification_rule_match_ambiguous": int(dereification_stats["dereification_rule_match_ambiguous"]),
        "dereification_rule_match_unmatched": int(dereification_stats["dereification_rule_match_unmatched"]),
        "dereification_opportunities": int(dereification_stats["dereification_opportunities"]),
        "framebase_index_used": bool(index_path),
        "framebase_index_path": str(index_path) if index_path else None,
        "spin_rules_source": str(rules_path) if rules_path and "spin" in rules_path.name.lower() else None,
        "warnings": warnings,
    }
    write_json(qc_payload, out, "qc_report.json")
    write_json(summary_payload, out, "summary.json")
    write_json(
        {
            "input": str(input),
            "resolved_input": str(input_csv),
            **fb_paths,
            "columns": cmap.model_dump(),
            "rule_count": len(rules),
            "dbp_label_count": len(labels),
        },
        out,
        "manifest.json",
    )

    if not no_graphml:
        write_graphml(nodes, out / "graph.graphml", reified_edges, nested_edges, dereified_edges)
    if not no_rdf:
        write_turtle(nodes, out / "graph.ttl", reified_edges, nested_edges, dereified_edges)

    console.print("\n[green]Done.[/green]")
    console.print(qc.model_dump_json(indent=2))


@app.command()
def detect(
    input: Path = typer.Option(..., "--input", help="Raw sentence CSV to run through FST."),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        help="Canonical fst_clean output directory. If omitted, only column detection is reported.",
    ),
    text_col: str = typer.Option("sentence", "--text-col", help="Sentence text column."),
    id_col: Optional[str] = typer.Option(None, "--id-col", help="Optional sentence ID column."),
    doc_col: Optional[str] = typer.Option(None, "--doc-col", help="Optional document ID column."),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume existing run state."),
    batch_size: int = typer.Option(16, "--batch-size", help="FST batch size."),
    device: str = typer.Option("auto", "--device", help="FST device: auto, cpu, mps, cuda."),
    checkpoint_every: int = typer.Option(100, "--checkpoint-every", help="Checkpoint interval."),
    dedupe: bool = typer.Option(
        True,
        "--dedupe/--no-dedupe",
        help="Run FST once per unique input text before expanding back to all rows.",
    ),
    dedupe_normalise: str = typer.Option(
        "exact",
        "--dedupe-normalise",
        help="Text dedupe mode: exact or normalised.",
    ),
) -> None:
    """Run FrameSemanticTransformer over a raw sentence CSV and write a canonical run."""
    import pandas as pd

    if out is None:
        df = pd.read_csv(input, nrows=50)
        cmap = detect_columns(df)
        console.print_json(
            data={
                "detected_columns": cmap.model_dump(),
                "next_command": (
                    "fst2framegraph detect "
                    f"--input {input} "
                    f"--text-col {text_col} "
                    "--id-col sentence_id "
                    "--doc-col doc_id "
                    "--out fst_clean "
                    "--resume"
                ),
            }
        )
        return

    try:
        report = encode_with_fst(
            data=input,
            sentence_col=text_col,
            sentence_id_col=id_col,
            doc_col=doc_col,
            out_dir=out,
            resume=resume,
            batch_size=batch_size,
            device=device,
            checkpoint_every=checkpoint_every,
            dedupe=dedupe,
            dedupe_normalise=dedupe_normalise,
        )
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print_json(
        data={
            "message": f"Detected frames into canonical run directory: {out}",
            "graph_ready": report.get("frame_elements", 0) > 0,
            "report": report,
            "next_command": _prepare_build_command(out),
        }
    )


@app.command("inspect")
def inspect_outputs(
    input: Path = typer.Option(..., "--input", help="FST output file or directory to inspect."),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Scan directories recursively."),
    allow_pickle: bool = typer.Option(
        False,
        "--allow-pickle",
        help="Allow inspecting trusted Python pickle files. Pickles can execute code.",
    ),
) -> None:
    """Inspect existing FST outputs and report whether they are graph-ready or convertible."""
    try:
        report = inspect_fst_outputs(input, recursive=recursive, allow_pickle=allow_pickle)
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print_json(data=report)


@app.command("convert")
def convert_outputs(
    input: Path = typer.Option(..., "--input", help="Existing FST output file or directory."),
    out: Path = typer.Option(..., "--out", help="Canonical fst_clean output directory."),
    allow_pickle: bool = typer.Option(
        False,
        "--allow-pickle",
        help="Allow loading trusted Python pickle files. Pickles can execute code.",
    ),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Scan directories recursively."),
) -> None:
    """Convert graph-ready CSV, JSON/JSONL, or trusted pickles into canonical v0.3 output."""
    try:
        report = convert_fst_outputs(
            input,
            out,
            allow_pickle=allow_pickle,
            recursive=recursive,
        )
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print_json(data=report)


@app.command("prepare")
def prepare_outputs(
    input: Path = typer.Option(..., "--input", help="Existing FST output file or directory."),
    out: Path = typer.Option(..., "--out", help="Canonical fst_clean output directory."),
    allow_pickle: bool = typer.Option(
        False,
        "--allow-pickle",
        help="Allow loading trusted Python pickle files. Pickles can execute code.",
    ),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Scan directories recursively."),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite known canonical files already present in --out.",
    ),
) -> None:
    """Prepare existing FST-like output for graph building."""
    try:
        inspection = inspect_fst_outputs(input, recursive=recursive)
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    detected = inspection["detected_format"]
    status = inspection["status"]

    if inspection.get("flat_only"):
        console.print_json(
            data={
                "message": (
                    "Input is flat-only. Flat counts may be possible, but reliable nested graphs "
                    "require frame_index and target/filler spans."
                ),
                "detected_format": detected,
                "status": status,
                "graph_ready": False,
                "warnings": inspection.get("warnings", []),
                "missing_required_columns": inspection.get("missing_required_columns", []),
                "next_command": _detect_next_command(),
            }
        )
        raise typer.Exit(1)

    if status == "unsafe_without_pickle_permission" and not allow_pickle:
        console.print_json(
            data={
                "message": (
                    "Python pickles can execute code. Prepare will only load trusted pickles "
                    "when --allow-pickle is passed."
                ),
                "detected_format": detected,
                "status": status,
                "graph_ready": False,
                "pickle_files": inspection.get("pickle_files", []),
                "next_command": (
                    "fst2framegraph prepare "
                    f"--input {input} "
                    f"--out {out} "
                    "--allow-pickle"
                ),
            }
        )
        raise typer.Exit(1)

    if not inspection.get("convertible") and not (
        allow_pickle and detected in {"pickle_file", "pickle_folder"}
    ):
        console.print_json(
            data={
                "message": f"Input is not preparable ({status}).",
                "detected_format": detected,
                "status": status,
                "graph_ready": False,
                "missing_required_columns": inspection.get("missing_required_columns", []),
                "next_command": f"fst2framegraph inspect --input {input}",
            }
        )
        raise typer.Exit(1)

    if overwrite:
        _clear_canonical_outputs(out)

    try:
        convert_report = convert_fst_outputs(
            input,
            out,
            allow_pickle=allow_pickle,
            recursive=recursive,
        )
        materialise_report = materialise_run(out)
        doctor_report = doctor_run(run_dir=out)
        prepared_report = inspect_fst_outputs(out)
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    graph_ready = bool(prepared_report.get("graph_ready"))
    console.print_json(
        data={
            "message": f"Prepared canonical run directory: {out}",
            "detected_format": detected,
            "status": prepared_report.get("status"),
            "graph_ready": graph_ready,
            "files_written": _files_written(out),
            "conversion_report": convert_report,
            "materialise_report": materialise_report,
            "doctor": doctor_report,
            "next_command": _prepare_build_command(out),
        }
    )
    if not graph_ready:
        raise typer.Exit(1)


@app.command("run")
def run_workflow(
    input: Path = typer.Option(..., "--input", help="File or folder to inspect and process."),
    out: Path = typer.Option(..., "--out", help="Canonical fst_clean output directory."),
    graph: bool = typer.Option(False, "--graph/--no-graph", help="Build the graph after prepare/detect when possible."),
    graph_out: Optional[Path] = typer.Option(None, "--graph-out", help="Optional graph output directory."),
    framebase_index: Optional[Path] = typer.Option(None, "--framebase-index", help="FrameBase SQLite index."),
    framebase_dir: Optional[Path] = typer.Option(None, "--framebase-dir", help="Directory containing framebase_index.sqlite."),
    text_col: str = typer.Option("sentence", "--text-col", help="Raw text sentence column."),
    id_col: Optional[str] = typer.Option(None, "--id-col", help="Optional sentence ID column for raw text."),
    doc_col: Optional[str] = typer.Option(None, "--doc-col", help="Optional document ID column for raw text."),
    allow_pickle: bool = typer.Option(False, "--allow-pickle", help="Allow loading trusted Python pickles."),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume existing run state."),
    checkpoint_every: int = typer.Option(100, "--checkpoint-every", help="Checkpoint interval for FST runs."),
    batch_size: int = typer.Option(16, "--batch-size", help="FST batch size for raw text input."),
    device: str = typer.Option("auto", "--device", help="FST device: auto, cpu, mps, cuda."),
    dedupe: bool = typer.Option(
        True,
        "--dedupe/--no-dedupe",
        help="Dedupe identical raw input texts before FST inference.",
    ),
    dedupe_normalise: str = typer.Option(
        "exact",
        "--dedupe-normalise",
        help="Text dedupe mode for raw input: exact or normalised.",
    ),
    chunk_text: bool = typer.Option(
        True,
        "--chunk-text/--no-chunk-text",
        help="Split long text rows into sentence-like chunks before FST.",
    ),
    chunk_min_words: int = typer.Option(
        2,
        "--chunk-min-words",
        help="Minimum words per generated text chunk.",
    ),
    chunk_max_words: int = typer.Option(
        70,
        "--chunk-max-words",
        help="Maximum words per generated text chunk.",
    ),
    plan: bool = typer.Option(False, "--plan", "--dry-run", help="Inspect and print planned actions without writing files."),
    yes: bool = typer.Option(False, "--yes", help="Skip non-risky confirmations."),
    interactive: bool = typer.Option(False, "--interactive", help="Ask guided questions when helpful."),
) -> None:
    """Smart workflow: inspect + plan + execute."""
    try:
        inspection = inspect_fst_outputs(input)
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    raw_text = _looks_like_raw_text_table(input, text_col)
    if graph and graph_out is None:
        graph_out = out / "graph"

    if plan:
        _print_run_plan(
            input=input,
            out=out,
            graph_out=graph_out,
            framebase_index=framebase_index,
            framebase_dir=framebase_dir,
            inspection=inspection,
            raw_text=raw_text,
            text_col=text_col,
            id_col=id_col,
            doc_col=doc_col,
            allow_pickle=allow_pickle,
            resume=resume,
            checkpoint_every=checkpoint_every,
            batch_size=batch_size,
            device=device,
            dedupe=dedupe,
            dedupe_normalise=dedupe_normalise,
            chunk_text=chunk_text,
            chunk_min_words=chunk_min_words,
            chunk_max_words=chunk_max_words,
        )
        return

    detected = inspection.get("detected_format")
    status = inspection.get("status")

    if status == "unsafe_without_pickle_permission" and not allow_pickle:
        if interactive and typer.confirm(
            "This folder contains pickles. Pickles can execute code. Load trusted pickles?",
            default=False,
        ):
            allow_pickle = True
        else:
            console.print_json(
                data={
                    "message": "Pickles can execute code. Run only on trusted files with --allow-pickle.",
                    "detected_format": detected,
                    "status": status,
                    "graph_ready": False,
                    "next_command": (
                        "fst2framegraph run "
                        f"--input {input} "
                        f"--out {out} "
                        "--allow-pickle"
                    ),
                }
            )
            raise typer.Exit(1)

    if inspection.get("flat_only"):
        console.print_json(
            data={
                "message": (
                    "Input is flat-only. Flat frame/FE counts may be possible, but reliable "
                    "nested graphs require frame_index and target/filler spans."
                ),
                "detected_format": detected,
                "status": status,
                "graph_ready": False,
                "missing_required_columns": inspection.get("missing_required_columns", []),
                "next_command": _detect_next_command_for_input(input, out, text_col, id_col, doc_col),
            }
        )
        raise typer.Exit(1)

    graph_report = None
    try:
        if detected == "v0.3_run_directory":
            run_dir = input
            materialise_report = materialise_run(run_dir)
            doctor_report = doctor_run(run_dir=run_dir, framebase_index=framebase_index)
            graph_report = _run_graph_build_if_requested(
                run_dir=run_dir,
                graph_out=graph_out,
                framebase_index=framebase_index,
                framebase_dir=framebase_dir,
            )
            result = {
                "message": f"Checked canonical run directory: {run_dir}",
                "detected_format": detected,
                "status": inspect_fst_outputs(run_dir).get("status"),
                "graph_ready": True,
                "canonical_run_dir": str(run_dir),
                "materialise_report": materialise_report,
                "doctor": doctor_report,
                "graph": graph_report,
                "next_command": _prepare_build_command(run_dir),
            }
        elif raw_text:
            if interactive and not yes and not typer.confirm(
                "This looks like a raw sentence CSV. Run FrameSemanticTransformer now?",
                default=False,
            ):
                console.print_json(
                    data={
                        "message": "Stopped before FST inference.",
                        "detected_format": "raw_sentence_csv",
                        "graph_ready": False,
                        "next_command": _detect_next_command_for_input(
                            input, out, text_col, id_col, doc_col
                        ),
                    }
                )
                raise typer.Exit(1)
            parse_data: Path | pd.DataFrame = input
            chunking_report: dict[str, object] | None = None
            parse_text_col = text_col
            parse_id_col = id_col
            parse_doc_col = doc_col
            if chunk_text:
                source_rows = pd.read_csv(input)
                chunk_df, chunk_map_df = _build_chunked_sentence_table(
                    input_path=input,
                    text_col=text_col,
                    id_col=id_col,
                    doc_col=doc_col,
                    min_chunk_words=chunk_min_words,
                    max_chunk_words=chunk_max_words,
                )
                if len(chunk_df) == 0:
                    raise ValueError("No text chunks were produced from input.")
                out.mkdir(parents=True, exist_ok=True)
                chunk_df.to_csv(out / "text_chunks.csv", index=False)
                chunk_map_df.to_csv(out / "text_chunk_mapping.csv", index=False)
                parse_data = chunk_df
                parse_text_col = "sentence"
                parse_id_col = "sentence_id"
                parse_doc_col = "doc_id"
                chunking_report = {
                    "enabled": True,
                    "rows_in": int(len(source_rows)),
                    "chunks_out": int(len(chunk_df)),
                    "mapping_rows": int(len(chunk_map_df)),
                    "chunk_min_words": chunk_min_words,
                    "chunk_max_words": chunk_max_words,
                    "chunk_table": str(out / "text_chunks.csv"),
                    "chunk_mapping_table": str(out / "text_chunk_mapping.csv"),
                }

            report = encode_with_fst(
                data=parse_data,
                sentence_col=parse_text_col,
                sentence_id_col=parse_id_col,
                doc_col=parse_doc_col,
                out_dir=out,
                resume=resume,
                checkpoint_every=checkpoint_every,
                batch_size=batch_size,
                device=device,
                dedupe=dedupe,
                dedupe_normalise=dedupe_normalise,
            )
            materialise_report = materialise_run(out)
            doctor_report = doctor_run(run_dir=out, framebase_index=framebase_index)
            graph_report = _run_graph_build_if_requested(
                run_dir=out,
                graph_out=graph_out,
                framebase_index=framebase_index,
                framebase_dir=framebase_dir,
            )
            result = {
                "message": f"Ran FST into canonical run directory: {out}",
                "detected_format": "raw_sentence_csv",
                "status": inspect_fst_outputs(out).get("status"),
                "graph_ready": report.get("frame_elements", 0) > 0,
                "canonical_run_dir": str(out),
                "report": report,
                "chunking": chunking_report,
                "materialise_report": materialise_report,
                "doctor": doctor_report,
                "graph": graph_report,
                "next_command": _prepare_build_command(out),
            }
        elif inspection.get("convertible") or (
            allow_pickle and detected in {"pickle_file", "pickle_folder"}
        ):
            convert_report = convert_fst_outputs(
                input,
                out,
                allow_pickle=allow_pickle,
            )
            materialise_report = materialise_run(out)
            doctor_report = doctor_run(run_dir=out, framebase_index=framebase_index)
            prepared = inspect_fst_outputs(out)
            graph_report = _run_graph_build_if_requested(
                run_dir=out,
                graph_out=graph_out,
                framebase_index=framebase_index,
                framebase_dir=framebase_dir,
            )
            result = {
                "message": f"Prepared canonical run directory: {out}",
                "detected_format": detected,
                "status": prepared.get("status"),
                "graph_ready": bool(prepared.get("graph_ready")),
                "canonical_run_dir": str(out),
                "conversion_report": convert_report,
                "materialise_report": materialise_report,
                "doctor": doctor_report,
                "graph": graph_report,
                "next_command": _prepare_build_command(out),
            }
        else:
            console.print_json(
                data={
                    "message": f"Input is not runnable automatically ({status}).",
                    "detected_format": detected,
                    "status": status,
                    "graph_ready": False,
                    "next_command": f"fst2framegraph inspect --input {input}",
                }
            )
            raise typer.Exit(1)
    except ImportError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    console.print_json(data=result)
    if graph_report and graph_report.get("status") == "skipped":
        raise typer.Exit(1)


@app.command("pipeline")
def pipeline_v2(
    input: Path = typer.Option(..., "--input", help="Input CSV path."),
    out_root: Path = typer.Option(Path("outputs"), "--out-root", help="Root output directory."),
    text_col: Optional[str] = typer.Option(None, "--text-col", help="Input text column name."),
    id_col: Optional[str] = typer.Option(None, "--id-col", help="Input ID column name."),
    doc_col: Optional[str] = typer.Option(None, "--doc-col", help="Input document ID column name."),
    framebase_index: Optional[Path] = typer.Option(None, "--framebase-index", help="Optional FrameBase index path."),
    framebase_dir: Optional[Path] = typer.Option(None, "--framebase-dir", help="Optional FrameBase directory."),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Optional stable run directory name."),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume if output state exists."),
    batch_size: int = typer.Option(16, "--batch-size", help="FST batch size."),
    dedupe: bool = typer.Option(True, "--dedupe/--no-dedupe", help="Dedupe input texts before FST."),
    dedupe_normalise: str = typer.Option("exact", "--dedupe-normalise", help="Dedupe mode: exact or normalised."),
    checkpoint_every: int = typer.Option(100, "--checkpoint-every", help="Checkpoint interval."),
    device: str = typer.Option("auto", "--device", help="FST device selection."),
    chunk_text: bool = typer.Option(True, "--chunk-text/--no-chunk-text", help="Chunk long input rows into sentence-like chunks."),
    chunk_min_words: int = typer.Option(2, "--chunk-min-words", help="Minimum words per chunk."),
    chunk_max_words: int = typer.Option(70, "--chunk-max-words", help="Maximum words per chunk."),
    top_n_frames: int = typer.Option(20, "--top-n-frames", help="Top frame types to report."),
    top_n_agents: int = typer.Option(30, "--top-n-agents", help="Top agent fillers to report."),
    min_count: int = typer.Option(2, "--min-count", help="Minimum count for lift rows."),
    n_communities: int = typer.Option(5, "--n-communities", help="Community detection limit."),
    random_seed: int = typer.Option(42, "--random-seed", help="Deterministic random seed."),
) -> None:
    """One-call product workflow: preflight + extract + graph + analysis."""
    try:
        payload = run_fst2graph(
            input_csv=input,
            out_root=out_root,
            text_col=text_col,
            id_col=id_col,
            doc_col=doc_col,
            framebase_index=framebase_index,
            framebase_dir=framebase_dir,
            run_name=run_name,
            resume=resume,
            batch_size=batch_size,
            dedupe=dedupe,
            dedupe_normalise=dedupe_normalise,
            checkpoint_every=checkpoint_every,
            device=device,
            chunk_text=chunk_text,
            chunk_min_words=chunk_min_words,
            chunk_max_words=chunk_max_words,
            top_n_frames=top_n_frames,
            top_n_agents=top_n_agents,
            min_count=min_count,
            n_communities=n_communities,
            random_seed=random_seed,
        )
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print_json(data=payload)


@app.command("framebase-status")
def framebase_status(
    framebase_dir: Path = typer.Option(Path("data/framebase"), "--framebase-dir", help="FrameBase directory."),
    write_manifest: bool = typer.Option(False, "--write-manifest", help="Write checksum manifest for existing files."),
    framebase_index: Optional[Path] = typer.Option(None, "--framebase-index", help="Optional SQLite index to inspect."),
    frame_name: Optional[str] = typer.Option(None, "--frame-name", help="Optional frame name for rule-candidate inspection."),
    subject_fe: Optional[str] = typer.Option(None, "--subject-fe", help="Optional subject FE name for rule-candidate inspection."),
    object_fe: Optional[str] = typer.Option(None, "--object-fe", help="Optional object FE name for rule-candidate inspection."),
    target_text: Optional[str] = typer.Option(None, "--target-text", help="Optional target text for rule-candidate inspection."),
    limit: int = typer.Option(20, "--limit", help="Maximum candidate rules to print for inspection."),
) -> None:
    inspect_requested = any(value is not None for value in [frame_name, subject_fe, object_fe])
    found = find_framebase_files(framebase_dir)
    if write_manifest or not inspect_requested:
        console.print_json(data={k: str(v) if v else None for k, v in found.items()})
    if write_manifest:
        manifest = write_framebase_manifest(framebase_dir)
        console.print_json(data=manifest)
    if inspect_requested:
        if not all([frame_name, subject_fe, object_fe]):
            console.print("[red]Provide --frame-name, --subject-fe, and --object-fe together.[/red]")
            raise typer.Exit(1)
        index = framebase_index or find_framebase_index(framebase_dir)
        if index is None:
            console.print("[red]No FrameBase index found. Build one first or pass --framebase-index.[/red]")
            raise typer.Exit(1)
        console.print_json(
            data=inspect_rule_candidates(
                index,
                frame_name=frame_name,
                subject_fe=subject_fe,
                object_fe=object_fe,
                target_text=target_text,
                limit=limit,
            )
        )


@app.command("materialise")
def materialise(run_dir: Path = typer.Option(..., "--run-dir", help="FST clean run directory.")) -> None:
    """Rebuild CSV/report outputs from fst_clean.jsonl and progress.sqlite."""
    report = materialise_run(run_dir)
    console.print_json(data=report)


@app.command("doctor")
def doctor(
    run_dir: Optional[Path] = typer.Option(None, "--run-dir", help="Canonical fst_clean run directory."),
    framebase_index: Optional[Path] = typer.Option(
        None,
        "--framebase-index",
        help="FrameBase SQLite index to check.",
    ),
) -> None:
    """Check run-directory and FrameBase-index health before graph building."""
    try:
        report = doctor_run(run_dir=run_dir, framebase_index=framebase_index)
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print_json(data=report)
    if not report["ok"]:
        raise typer.Exit(1)
