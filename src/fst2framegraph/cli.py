from __future__ import annotations

from pathlib import Path
from typing import Optional

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
    load_dbp_labels_from_index,
    load_rules_from_index,
    load_schema_from_index,
)
from fst2framegraph.framebase.load_dbp_labels import load_dbp_labels
from fst2framegraph.framebase.load_schema import FrameBaseSchema
from fst2framegraph.framebase.parse_dered_rules import parse_dered_rules
from fst2framegraph.framebase.rule_index import RuleIndex
from fst2framegraph.fst import materialise_run
from fst2framegraph.graph.build_dereified import build_dereified_edges
from fst2framegraph.graph.build_nested import build_nested_edges
from fst2framegraph.graph.build_reified import build_reified_tables
from fst2framegraph.graph.export_graph import build_sentence_graphs, write_graphml, write_turtle
from fst2framegraph.io.column_detection import detect_columns
from fst2framegraph.io.inspect_outputs import (
    convert_fst_outputs,
    doctor_run,
    inspect_fst_outputs,
)
from fst2framegraph.io.read_fst import read_fst_csv
from fst2framegraph.io.write_outputs import ensure_out_dir, write_csv, write_json, write_jsonl
from fst2framegraph.qc.ambiguity_report import repeated_frame_warnings
from fst2framegraph.qc.coverage_report import make_qc_report
from fst2framegraph.qc.validation import require_file
from fst2framegraph.schema import ColumnMap

app = typer.Typer(help="Convert FrameNet-style parser output into FrameBase-compatible graphs.")
console = Console()


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
    rules = dered_rules or found.get("dereification_rules_sparql")
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
) -> None:
    """Build the compact FrameBase SQLite index used by normal graph builds."""
    try:
        report = build_framebase_index_file(
            framebase_dir=framebase_dir,
            index_path=index,
            overwrite=overwrite,
            framebase_core=framebase_core,
            dbp_labels=dbp_labels,
            dered_rules=dered_rules,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print_json(data=report)


@app.command()
def build(
    input: Path = typer.Option(..., "--input", help="FrameNet/FST-style long CSV."),
    out: Path = typer.Option(..., "--out", help="Output directory."),
    framebase_dir: Optional[Path] = typer.Option(
        None,
        "--framebase-dir",
        help="Directory containing FrameBase_schema_core.ttl.gz, FrameBase_schema_dbps.ttl.gz and dereificationRulesSparqlFormat.txt.zip. Defaults to data/framebase if present.",
    ),
    framebase_core: Optional[Path] = typer.Option(None, help="FrameBase core schema TTL/Turtle gzip."),
    dbp_labels: Optional[Path] = typer.Option(None, help="FrameBase direct binary predicate labels TTL/Turtle gzip."),
    dered_rules: Optional[Path] = typer.Option(None, help="FrameBase dereification rules, SPARQL zip or text."),
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
    require_file(input, "input")
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

    console.print(f"[bold]Reading[/bold] {input}")
    raw_df, detected = read_fst_csv(input)
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
        rules = parse_dered_rules(rules_path, labels)
    rule_index = RuleIndex.from_rules(rules)
    if index_path and not rules:
        warnings.append("FrameBase index contains no dereification rules; DBP dereified edges disabled.")
    elif rules_path and not rules:
        warnings.append("No dereification rules were parsed. Check the FrameBase rule file format.")

    console.print(f"[bold]Parsed rules:[/bold] {len(rules)}")
    console.print("[bold]Building dereified graph[/bold]")
    dereified_edges = build_dereified_edges(frame_elements, rule_index)

    console.print("[bold]Writing outputs[/bold]")
    write_csv(documents, out, "documents.csv")
    write_csv(sentences, out, "sentences.csv")
    write_csv(frame_instances, out, "frame_instances.csv")
    write_csv(frame_elements, out, "frame_elements.csv")
    write_csv(nodes, out, "graph_nodes.csv")
    write_csv(reified_edges, out, "graph_edges_reified.csv")
    write_csv(nested_edges, out, "graph_edges_nested.csv")
    write_csv(dereified_edges, out, "graph_edges_dereified.csv")

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
    write_json(qc.model_dump(), out, "qc_report.json")
    write_json(qc.model_dump(), out, "summary.json")
    write_json(
        {
            "input": str(input),
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
def detect(input: Path = typer.Option(..., "--input", help="CSV to inspect.")) -> None:
    import pandas as pd

    df = pd.read_csv(input, nrows=50)
    cmap = detect_columns(df)
    console.print(cmap.model_dump_json(indent=2))


@app.command("inspect")
def inspect_outputs(
    input: Path = typer.Option(..., "--input", help="FST output file or directory to inspect."),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Scan directories recursively."),
) -> None:
    """Inspect existing FST outputs and report whether they are graph-ready or convertible."""
    try:
        report = inspect_fst_outputs(input, recursive=recursive)
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


@app.command("framebase-status")
def framebase_status(
    framebase_dir: Path = typer.Option(Path("data/framebase"), "--framebase-dir", help="FrameBase directory."),
    write_manifest: bool = typer.Option(False, "--write-manifest", help="Write checksum manifest for existing files."),
) -> None:
    found = find_framebase_files(framebase_dir)
    console.print_json(data={k: str(v) if v else None for k, v in found.items()})
    if write_manifest:
        manifest = write_framebase_manifest(framebase_dir)
        console.print_json(data=manifest)


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
