from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from fst2framegraph.cli import app
from fst2framegraph.framebase.index import build_framebase_index
from fst2framegraph.framebase.parse_spin_rules import parse_spin_dereification_rules


SPIN_FIXTURE = """
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix sp: <http://spinrdf.org/sp#> .

<http://framebase.org/rule/capability-can>
    rdf:type sp:Construct ;
    sp:templates (
        [
            sp:subject [ sp:varName "S" ] ;
            sp:predicate <http://framebase.org/dbp/Capability.hasCapabilityForEvent> ;
            sp:object [ sp:varName "O" ]
        ]
    ) ;
    sp:where (
        [
            sp:subject [ sp:varName "R" ] ;
            sp:predicate rdf:type ;
            sp:object <http://framebase.org/frame/Capability.can.verb>
        ]
        [
            sp:subject [ sp:varName "R" ] ;
            sp:predicate <http://framebase.org/fe/Capability.has_entity> ;
            sp:object [ sp:varName "S" ]
        ]
        [
            sp:subject [ sp:varName "R" ] ;
            sp:predicate <http://framebase.org/fe/Capability.has_event> ;
            sp:object [ sp:varName "O" ]
        ]
    ) .
"""


SPIN_FIXTURE_AMBIGUOUS = """
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix sp: <http://spinrdf.org/sp#> .

<http://framebase.org/rule/capability-can>
    rdf:type sp:Construct ;
    sp:templates (
        [
            sp:subject [ sp:varName "S" ] ;
            sp:predicate <http://framebase.org/dbp/Capability.hasCapabilityForEvent> ;
            sp:object [ sp:varName "O" ]
        ]
    ) ;
    sp:where (
        [
            sp:subject [ sp:varName "R" ] ;
            sp:predicate rdf:type ;
            sp:object <http://framebase.org/frame/Capability.can.verb>
        ]
        [
            sp:subject [ sp:varName "R" ] ;
            sp:predicate <http://framebase.org/fe/Capability.has_entity> ;
            sp:object [ sp:varName "S" ]
        ]
        [
            sp:subject [ sp:varName "R" ] ;
            sp:predicate <http://framebase.org/fe/Capability.has_event> ;
            sp:object [ sp:varName "O" ]
        ]
    ) .

<http://framebase.org/rule/capability-enable>
    rdf:type sp:Construct ;
    sp:templates (
        [
            sp:subject [ sp:varName "S" ] ;
            sp:predicate <http://framebase.org/dbp/Capability.enables> ;
            sp:object [ sp:varName "O" ]
        ]
    ) ;
    sp:where (
        [
            sp:subject [ sp:varName "R" ] ;
            sp:predicate rdf:type ;
            sp:object <http://framebase.org/frame/Capability.enable.verb>
        ]
        [
            sp:subject [ sp:varName "R" ] ;
            sp:predicate <http://framebase.org/fe/Capability.has_entity> ;
            sp:object [ sp:varName "S" ]
        ]
        [
            sp:subject [ sp:varName "R" ] ;
            sp:predicate <http://framebase.org/fe/Capability.has_event> ;
            sp:object [ sp:varName "O" ]
        ]
    ) .
"""


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _write_gzip_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(text.strip() + "\n")


def write_tiny_current_framebase(
    framebase_dir: Path,
    *,
    spin_text: str | None = SPIN_FIXTURE,
    include_clusters: bool = True,
) -> None:
    _write_text(
        framebase_dir / "FrameBase_schema_core.ttl",
        """
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        <http://framebase.org/frame/Capability.can.verb> rdfs:label "Capability.can.verb" .
        <http://framebase.org/frame/Capability.enable.verb> rdfs:label "Capability.enable.verb" .
        <http://framebase.org/fe/Capability.has_entity> rdfs:label "Entity" .
        <http://framebase.org/fe/Capability.has_event> rdfs:label "Event" .
        """,
    )
    _write_text(
        framebase_dir / "FrameBase_schema_dbps.ttl",
        """
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        <http://framebase.org/dbp/Capability.hasCapabilityForEvent> rdfs:label "hasCapabilityForEvent" .
        <http://framebase.org/dbp/Capability.enables> rdfs:label "enables" .
        """,
    )
    if spin_text is not None:
        _write_gzip_text(framebase_dir / "dereificationRulesSpinFormat.ttl.gz", spin_text)
    if include_clusters:
        _write_text(
            framebase_dir / "clusters.txt",
            """
            <parentMacroframe>http://framebase.org/frame/Capability</parentMacroframe>
            http://framebase.org/frame/Capability.can.verb
            """,
        )
        _write_text(
            framebase_dir / "clusterPairs.txt",
            ":me/Capability.can.verb   :me/Synset123.can.verb",
        )
        _write_text(
            framebase_dir / "lexicalClusters.txt",
            """
            clusters.size()=1
            lexicalClusters.size()=1
            ### clusterRepresentant: http://framebase.org/frame/Synset123.can.verb
            http://framebase.org/frame/Capability.can.verb
            can
            """,
        )
        _write_text(
            framebase_dir / "manual/FrameBase_schema_manual_extensions.ttl",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n",
        )
        _write_text(
            framebase_dir / "manual/inferenceRulesForSchema.txt",
            "# tiny fixture\n",
        )


def write_graph_ready_capability_csv(path: Path, *, target_text: str = "can") -> None:
    pd.DataFrame(
        {
            "doc_id": ["d1", "d1"],
            "sentence_id": ["s1", "s1"],
            "sentence": ["Technology can reduce emissions.", "Technology can reduce emissions."],
            "frame_name": ["Capability", "Capability"],
            "frame_index": [0, 0],
            "target_text": [target_text, target_text],
            "target_start": [11, 11],
            "target_end": [14, 14],
            "element_name": ["Entity", "Event"],
            "element_filler": ["Technology", "reduce emissions"],
            "filler_start": [0, 15],
            "filler_end": [10, 31],
        }
    ).to_csv(path, index=False)


def test_parse_current_spin_rule_fixture(tmp_path: Path) -> None:
    rules_path = tmp_path / "dereificationRulesSpinFormat.ttl.gz"
    _write_gzip_text(rules_path, SPIN_FIXTURE)

    rules = list(parse_spin_dereification_rules(rules_path))

    assert len(rules) == 1
    rule = rules[0]
    assert rule.frame_name == "Capability"
    assert rule.microframe_name == "can.verb"
    assert rule.target_lemma_or_lu == "can"
    assert rule.subject_fe_name == "Entity"
    assert rule.object_fe_name == "Event"
    assert rule.dbp_predicate_iri == "http://framebase.org/dbp/Capability.hasCapabilityForEvent"
    assert rule.dbp_predicate_name == "hasCapabilityForEvent"
    assert rule.source_format == "spin"
    assert rule.parse_status == "parsed"


def test_build_framebase_index_loads_spin_rules_and_clusters(tmp_path: Path) -> None:
    framebase_dir = tmp_path / "framebase"
    write_tiny_current_framebase(framebase_dir)

    report = build_framebase_index(framebase_dir=framebase_dir, overwrite=True)

    assert report["dereification_rules_loaded"] == 1
    assert report["dereification_rules_source_format"] == "spin"
    assert report["cluster_files_loaded"] is True
    with sqlite3.connect(report["index_path"]) as conn:
        row = conn.execute(
            """
            SELECT frame_name, microframe_name, target_lemma_or_lu,
                   subject_fe_name, object_fe_name, dbp_predicate_name
            FROM dereification_rules
            """
        ).fetchone()
        assert row == (
            "Capability",
            "can.verb",
            "can",
            "Entity",
            "Event",
            "hasCapabilityForEvent",
        )
        assert conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM cluster_pairs").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM lexical_clusters").fetchone()[0] == 2


def test_build_emits_official_direct_edge_from_spin_rule(tmp_path: Path) -> None:
    framebase_dir = tmp_path / "framebase"
    write_tiny_current_framebase(framebase_dir)
    index_report = build_framebase_index(framebase_dir=framebase_dir, overwrite=True)
    input_csv = tmp_path / "input.csv"
    write_graph_ready_capability_csv(input_csv)

    result = CliRunner().invoke(
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
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["official_framebase_reder_edges"] == 1
    assert summary["dereification_rules_matched"] == 1
    direct_edges = pd.read_csv(tmp_path / "out" / "direct_edges.csv")
    assert len(direct_edges) == 1
    edge = direct_edges.iloc[0]
    assert edge["edge_type"] == "official_framebase_reder_edge"
    assert edge["subject_filler"] == "Technology"
    assert edge["object_filler"] == "reduce emissions"
    assert edge["predicate_iri"] == "http://framebase.org/dbp/Capability.hasCapabilityForEvent"
    assert edge["match_tier"] == "frame_target_fe_unique"


def test_build_reports_ambiguous_broad_match_without_emitting_edge(tmp_path: Path) -> None:
    framebase_dir = tmp_path / "framebase"
    write_tiny_current_framebase(framebase_dir, spin_text=SPIN_FIXTURE_AMBIGUOUS)
    index_report = build_framebase_index(framebase_dir=framebase_dir, overwrite=True)
    input_csv = tmp_path / "input.csv"
    write_graph_ready_capability_csv(input_csv, target_text="")

    result = CliRunner().invoke(
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
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["official_framebase_reder_edges"] == 0
    assert summary["dereification_rule_match_ambiguous"] >= 1
    diagnostics = pd.read_csv(tmp_path / "out" / "dereification_diagnostics.csv")
    assert set(diagnostics["status"]) == {"ambiguous"}
    assert diagnostics.iloc[0]["candidate_rule_count"] == 2


def test_build_warns_when_dbp_schema_exists_but_spin_rules_are_missing(tmp_path: Path) -> None:
    framebase_dir = tmp_path / "framebase"
    write_tiny_current_framebase(framebase_dir, spin_text=None)
    index_report = build_framebase_index(framebase_dir=framebase_dir, overwrite=True)
    input_csv = tmp_path / "input.csv"
    write_graph_ready_capability_csv(input_csv)

    result = CliRunner().invoke(
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
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["official_framebase_reder_edges"] == 0
    assert any("DBP schema/labels are available, but dereification rules are not supplied" in warning for warning in summary["warnings"])
