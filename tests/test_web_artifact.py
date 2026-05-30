from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from fst2framegraph.cli import app
from fst2framegraph.framebase.index import build_framebase_index
from test_spin_dereification import write_graph_ready_capability_csv, write_tiny_current_framebase


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_build_writes_static_web_artifact_contract(tmp_path: Path) -> None:
    framebase_dir = tmp_path / "framebase"
    write_tiny_current_framebase(framebase_dir)
    index_report = build_framebase_index(framebase_dir=framebase_dir, overwrite=True)
    input_csv = tmp_path / "input.csv"
    write_graph_ready_capability_csv(input_csv)
    out_dir = tmp_path / "out"

    result = CliRunner().invoke(
        app,
        [
            "build",
            "--input",
            str(input_csv),
            "--out",
            str(out_dir),
            "--framebase-index",
            index_report["index_path"],
            "--no-rdf",
            "--no-graphml",
        ],
    )

    assert result.exit_code == 0, result.output
    artifact_dir = out_dir / "web_artifact"
    expected_files = {
        "manifest.json",
        "summary.json",
        "documents.json",
        "sentences.json",
        "frames.json",
        "frame_elements.json",
        "nested_edges.json",
        "direct_edges.json",
        "dereification_diagnostics.json",
    }
    assert {path.name for path in artifact_dir.glob("*.json")} == expected_files

    manifest = _read_json(artifact_dir / "manifest.json")
    assert manifest["artifact_type"] == "fst2framegraph.web_artifact"
    assert manifest["schema_version"] == 1
    assert set(manifest["files"]) == expected_files - {"manifest.json"}

    summary = _read_json(artifact_dir / "summary.json")
    assert summary["documents"] == 1
    assert summary["sentences"] == 1
    assert summary["frames"] == 1
    assert summary["frame_elements"] == 2
    assert summary["direct_edges"] == 1

    documents = _read_json(artifact_dir / "documents.json")
    assert documents[0]["document_id"] == "doc_d1"
    assert documents[0]["source_document_id"] == "d1"

    frames = _read_json(artifact_dir / "frames.json")
    assert frames[0]["frame_name"] == "Capability"
    assert frames[0]["sentence_id"] == "sent_s1"
    assert frames[0]["target_text"] == "can"

    frame_elements = _read_json(artifact_dir / "frame_elements.json")
    assert {row["fe_name"] for row in frame_elements} == {"Entity", "Event"}
    assert {row["filler_text"] for row in frame_elements} == {"Technology", "reduce emissions"}

    direct_edges = _read_json(artifact_dir / "direct_edges.json")
    assert direct_edges[0]["edge_type"] == "official_framebase_reder_edge"
    assert direct_edges[0]["subject_filler"] == "Technology"
    assert direct_edges[0]["object_filler"] == "reduce emissions"
    assert (
        direct_edges[0]["predicate_iri"]
        == "http://framebase.org/dbp/Capability.hasCapabilityForEvent"
    )
    assert direct_edges[0]["source_rule_id"]

    diagnostics = _read_json(artifact_dir / "dereification_diagnostics.json")
    assert diagnostics == []
