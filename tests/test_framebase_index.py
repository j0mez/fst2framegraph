from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from fst2framegraph.cli import app
from fst2framegraph.framebase.index import (
    build_framebase_index,
    load_dbp_labels_from_index,
    load_rules_from_index,
    load_schema_from_index,
)
from fst2framegraph.framebase.load_schema import FrameBaseSchema


def write_tiny_framebase(framebase_dir: Path, *, with_rules: bool = False) -> None:
    framebase_dir.mkdir(parents=True)
    (framebase_dir / "FrameBase_schema_core.ttl").write_text(
        """
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        <http://framebase.org/frame/Capability> rdfs:label "Capability" .
        <http://framebase.org/fe/Capability.has_entity> rdfs:label "Entity" .
        <http://framebase.org/fe/Capability.has_event> rdfs:label "Event" .
        """,
        encoding="utf-8",
    )
    (framebase_dir / "FrameBase_schema_dbps.ttl").write_text(
        """
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        <http://framebase.org/dbp/Capability.enables> rdfs:label "enables" .
        """,
        encoding="utf-8",
    )
    if with_rules:
        (framebase_dir / "dereificationRulesSparqlFormat.txt").write_text(
            """
            CONSTRUCT { ?s <http://framebase.org/dbp/Capability.enables> ?o }
            WHERE {
              ?f a <http://framebase.org/frame/Capability> .
              ?f <http://framebase.org/fe/Capability.has_entity> ?s .
              ?f <http://framebase.org/fe/Capability.has_event> ?o .
            }
            """,
            encoding="utf-8",
        )


def write_clean_input(path: Path) -> None:
    pd.DataFrame(
        {
            "doc_id": ["d1", "d1"],
            "sentence_id": ["s1", "s1"],
            "sentence": ["Technology can reduce emissions.", "Technology can reduce emissions."],
            "frame_name": ["Capability", "Capability"],
            "frame_index": [0, 0],
            "target_text": ["can", "can"],
            "element_name": ["Entity", "Event"],
            "element_filler": ["Technology", "reduce emissions"],
        }
    ).to_csv(path, index=False)


def test_build_framebase_index_missing_rules_do_not_fail(tmp_path: Path) -> None:
    framebase_dir = tmp_path / "framebase"
    write_tiny_framebase(framebase_dir, with_rules=False)

    report = build_framebase_index(framebase_dir=framebase_dir, overwrite=True)

    assert report["rules"] == 0
    assert report["warnings"]
    schema = load_schema_from_index(report["index_path"])
    frame_iri, frame_valid = schema.get_frame_iri("Capability")
    fe_iri, fe_valid = schema.get_fe_iri("Capability", "Entity")
    assert frame_valid is True
    assert frame_iri == "http://framebase.org/frame/Capability"
    assert fe_valid is True
    assert fe_iri == "http://framebase.org/fe/Capability.has_entity"
    labels = load_dbp_labels_from_index(report["index_path"])
    assert labels["http://framebase.org/dbp/Capability.enables"] == "enables"
    assert load_rules_from_index(report["index_path"]) == []

    with sqlite3.connect(report["index_path"]) as conn:
        manifest_rows = {
            row[0]: row
            for row in conn.execute(
                "SELECT key, path, sha256, size_bytes, status, error FROM manifest"
            )
        }
    assert manifest_rows["core_schema"][2]
    assert manifest_rows["core_schema"][3] > 0
    assert manifest_rows["core_schema"][4] == "indexed"
    assert manifest_rows["dereification_rules_sparql"][4] == "missing"


def test_build_framebase_index_broken_rules_warn_without_breaking_schema(tmp_path: Path) -> None:
    framebase_dir = tmp_path / "framebase"
    write_tiny_framebase(framebase_dir, with_rules=False)
    (framebase_dir / "dereificationRulesSparqlFormat.txt").write_text(
        "this is not a useful SPARQL construct",
        encoding="utf-8",
    )

    report = build_framebase_index(framebase_dir=framebase_dir, overwrite=True)

    assert report["frames"] > 0
    assert report["frame_element_lookup_keys"] > 0
    assert report["rules"] == 0
    assert report["warnings"]
    assert "dereified edges disabled" in " ".join(report["warnings"])


def test_build_uses_framebase_index(tmp_path: Path) -> None:
    framebase_dir = tmp_path / "framebase"
    write_tiny_framebase(framebase_dir, with_rules=True)
    index_report = build_framebase_index(framebase_dir=framebase_dir, overwrite=True)

    input_csv = tmp_path / "input.csv"
    write_clean_input(input_csv)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "build",
            "--input",
            str(input_csv),
            "--out",
            str(tmp_path / "out"),
            "--framebase-index",
            index_report["index_path"],
            "--no-rdf",
        ],
    )

    assert result.exit_code == 0, result.output
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["framebase_index"] == index_report["index_path"]
    frame_elements = pd.read_csv(tmp_path / "out" / "frame_elements.csv")
    assert set(frame_elements["frame_element_validated"]) == {True}
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    for key in [
        "framebase_validated_frames",
        "framebase_unmatched_frames",
        "framebase_validated_frame_elements",
        "framebase_unmatched_frame_elements",
        "nested_edges",
        "dereified_edges",
        "warnings",
    ]:
        assert key in summary


def test_index_and_raw_ttl_validation_match(tmp_path: Path) -> None:
    framebase_dir = tmp_path / "framebase"
    write_tiny_framebase(framebase_dir, with_rules=False)
    index_report = build_framebase_index(framebase_dir=framebase_dir, overwrite=True)

    indexed_schema = load_schema_from_index(index_report["index_path"])
    raw_schema = FrameBaseSchema.from_turtle(framebase_dir / "FrameBase_schema_core.ttl")

    assert indexed_schema.get_frame_iri("Capability") == raw_schema.get_frame_iri("Capability")
    assert indexed_schema.get_fe_iri("Capability", "Entity") == raw_schema.get_fe_iri(
        "Capability", "Entity"
    )


def test_build_falls_back_to_raw_ttl_without_index(tmp_path: Path) -> None:
    framebase_dir = tmp_path / "framebase"
    write_tiny_framebase(framebase_dir, with_rules=False)
    input_csv = tmp_path / "input.csv"
    write_clean_input(input_csv)

    result = CliRunner().invoke(
        app,
        [
            "build",
            "--input",
            str(input_csv),
            "--out",
            str(tmp_path / "out"),
            "--framebase-core",
            str(framebase_dir / "FrameBase_schema_core.ttl"),
            "--dbp-labels",
            str(framebase_dir / "FrameBase_schema_dbps.ttl"),
            "--no-rdf",
        ],
    )

    assert result.exit_code == 0, result.output
    frame_elements = pd.read_csv(tmp_path / "out" / "frame_elements.csv")
    assert set(frame_elements["frame_element_validated"]) == {True}


def test_build_falls_back_to_generated_iris_with_warnings(tmp_path: Path) -> None:
    input_csv = tmp_path / "input.csv"
    write_clean_input(input_csv)

    result = CliRunner().invoke(
        app,
        [
            "build",
            "--input",
            str(input_csv),
            "--out",
            str(tmp_path / "out"),
            "--framebase-dir",
            str(tmp_path / "empty-framebase"),
            "--no-rdf",
        ],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["framebase_validated_frames"] == 0
    assert summary["framebase_validated_frame_elements"] == 0
    assert summary["warnings"]


def test_build_framebase_index_cli(tmp_path: Path) -> None:
    framebase_dir = tmp_path / "framebase"
    write_tiny_framebase(framebase_dir, with_rules=False)

    result = CliRunner().invoke(
        app,
        [
            "build-framebase-index",
            "--framebase-dir",
            str(framebase_dir),
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (framebase_dir / "framebase_index.sqlite").exists()


def test_setup_framebase_build_index_cli_with_existing_files(tmp_path: Path) -> None:
    framebase_dir = tmp_path / "framebase"
    write_tiny_framebase(framebase_dir, with_rules=False)

    result = CliRunner().invoke(
        app,
        [
            "setup-framebase",
            "--out",
            str(framebase_dir),
            "--manifest-only",
            "--build-index",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (framebase_dir / "framebase_manifest.json").exists()
    assert (framebase_dir / "framebase_index.sqlite").exists()


def test_build_framebase_index_cli_missing_core_error_is_clear(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "build-framebase-index",
            "--framebase-dir",
            str(tmp_path / "missing-framebase"),
        ],
    )

    assert result.exit_code != 0
    assert "FrameBase core schema not found" in result.output
