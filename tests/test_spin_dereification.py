from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from fst2framegraph.cli import app
from fst2framegraph.framebase.index import build_framebase_index, inspect_rule_candidates
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


REAL_STYLE_SPIN_FIXTURE = """
_:rule1 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://spinrdf.org/sp#Construct> .
_:rule1 <http://spinrdf.org/sp#templates> _:tmpl_list1 .
_:rule1 <http://spinrdf.org/sp#where> _:where_list1 .
_:tmpl_list1 <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> _:tmpl_row1 .
_:tmpl_list1 <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest> <http://www.w3.org/1999/02/22-rdf-syntax-ns#nil> .
_:tmpl_row1 <http://spinrdf.org/sp#subject> _:var_s .
_:tmpl_row1 <http://spinrdf.org/sp#predicate> <http://framebase.org/dbp/Capability.hasCapabilityForEvent> .
_:tmpl_row1 <http://spinrdf.org/sp#object> _:var_o .
_:var_s <http://spinrdf.org/sp#varName> "S"^^<http://www.w3.org/2001/XMLSchema#string> .
_:var_o <http://spinrdf.org/sp#varName> "O"^^<http://www.w3.org/2001/XMLSchema#string> .
_:where_list1 <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> _:where_row1 .
_:where_list1 <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest> _:where_list2 .
_:where_list2 <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> _:where_row2 .
_:where_list2 <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest> _:where_list3 .
_:where_list3 <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> _:where_row3 .
_:where_list3 <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest> <http://www.w3.org/1999/02/22-rdf-syntax-ns#nil> .
_:where_row1 <http://spinrdf.org/sp#subject> _:var_r1 .
_:where_row1 <http://spinrdf.org/sp#predicate> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> .
_:where_row1 <http://spinrdf.org/sp#object> <http://framebase.org/frame/Capability.can.verb> .
_:where_row2 <http://spinrdf.org/sp#subject> _:var_r2 .
_:where_row2 <http://spinrdf.org/sp#predicate> <http://framebase.org/fe/Capability.has_entity> .
_:where_row2 <http://spinrdf.org/sp#object> _:var_s2 .
_:where_row3 <http://spinrdf.org/sp#subject> _:var_r3 .
_:where_row3 <http://spinrdf.org/sp#predicate> <http://framebase.org/fe/Capability.has_event> .
_:where_row3 <http://spinrdf.org/sp#object> _:var_o2 .
_:var_r1 <http://spinrdf.org/sp#varName> "R"^^<http://www.w3.org/2001/XMLSchema#string> .
_:var_r2 <http://spinrdf.org/sp#varName> "R"^^<http://www.w3.org/2001/XMLSchema#string> .
_:var_r3 <http://spinrdf.org/sp#varName> "R"^^<http://www.w3.org/2001/XMLSchema#string> .
_:var_s2 <http://spinrdf.org/sp#varName> "S"^^<http://www.w3.org/2001/XMLSchema#string> .
_:var_o2 <http://spinrdf.org/sp#varName> "O"^^<http://www.w3.org/2001/XMLSchema#string> .
"""


REAL_STYLE_SPIN_FIXTURE_WITH_SECOND_RULE = REAL_STYLE_SPIN_FIXTURE + """
_:rule2 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://spinrdf.org/sp#Construct> .
_:rule2 <http://spinrdf.org/sp#templates> _:tmpl_list9 .
_:rule2 <http://spinrdf.org/sp#where> _:where_list9 .
_:tmpl_list9 <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> _:tmpl_row9 .
_:tmpl_list9 <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest> <http://www.w3.org/1999/02/22-rdf-syntax-ns#nil> .
_:tmpl_row9 <http://spinrdf.org/sp#subject> _:var_s9 .
_:tmpl_row9 <http://spinrdf.org/sp#predicate> <http://framebase.org/dbp/Capability.enables> .
_:tmpl_row9 <http://spinrdf.org/sp#object> _:var_o9 .
_:var_s9 <http://spinrdf.org/sp#varName> "S"^^<http://www.w3.org/2001/XMLSchema#string> .
_:var_o9 <http://spinrdf.org/sp#varName> "O"^^<http://www.w3.org/2001/XMLSchema#string> .
_:where_list9 <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> _:where_row9a .
_:where_list9 <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest> _:where_list10 .
_:where_list10 <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> _:where_row9b .
_:where_list10 <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest> _:where_list11 .
_:where_list11 <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> _:where_row9c .
_:where_list11 <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest> <http://www.w3.org/1999/02/22-rdf-syntax-ns#nil> .
_:where_row9a <http://spinrdf.org/sp#subject> _:var_r9a .
_:where_row9a <http://spinrdf.org/sp#predicate> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> .
_:where_row9a <http://spinrdf.org/sp#object> <http://framebase.org/frame/Capability.enable.verb> .
_:where_row9b <http://spinrdf.org/sp#subject> _:var_r9b .
_:where_row9b <http://spinrdf.org/sp#predicate> <http://framebase.org/fe/Capability.has_entity> .
_:where_row9b <http://spinrdf.org/sp#object> _:var_s9b .
_:where_row9c <http://spinrdf.org/sp#subject> _:var_r9c .
_:where_row9c <http://spinrdf.org/sp#predicate> <http://framebase.org/fe/Capability.has_event> .
_:where_row9c <http://spinrdf.org/sp#object> _:var_o9c .
_:var_r9a <http://spinrdf.org/sp#varName> "R"^^<http://www.w3.org/2001/XMLSchema#string> .
_:var_r9b <http://spinrdf.org/sp#varName> "R"^^<http://www.w3.org/2001/XMLSchema#string> .
_:var_r9c <http://spinrdf.org/sp#varName> "R"^^<http://www.w3.org/2001/XMLSchema#string> .
_:var_s9b <http://spinrdf.org/sp#varName> "S"^^<http://www.w3.org/2001/XMLSchema#string> .
_:var_o9c <http://spinrdf.org/sp#varName> "O"^^<http://www.w3.org/2001/XMLSchema#string> .
"""


REAL_STYLE_SPIN_FIXTURE_WITH_BROKEN_RULE = REAL_STYLE_SPIN_FIXTURE + """
_:broken_rule <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://spinrdf.org/sp#Construct> .
_:broken_rule <http://spinrdf.org/sp#templates> _:broken_tmpl_list .
_:broken_rule <http://spinrdf.org/sp#where> _:broken_where_list .
_:broken_tmpl_list <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> _:broken_tmpl_row .
_:broken_tmpl_list <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest> <http://www.w3.org/1999/02/22-rdf-syntax-ns#nil> .
_:broken_tmpl_row <http://spinrdf.org/sp#subject> _:broken_var_s .
_:broken_tmpl_row <http://spinrdf.org/sp#predicate> <http://framebase.org/dbp/Capability.enables> .
_:broken_tmpl_row <http://spinrdf.org/sp#object> _:broken_var_o .
_:broken_var_s <http://spinrdf.org/sp#varName> "S"^^<http://www.w3.org/2001/XMLSchema#string> .
_:broken_var_o <http://spinrdf.org/sp#varName> "O"^^<http://www.w3.org/2001/XMLSchema#string> .
_:broken_where_list <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> _:broken_row1 .
_:broken_where_list <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest> <http://www.w3.org/1999/02/22-rdf-syntax-ns#nil> .
_:broken_row1 <http://spinrdf.org/sp#subject> _:broken_var_r .
_:broken_row1 <http://spinrdf.org/sp#predicate> <http://framebase.org/fe/Capability.has_entity> .
_:broken_row1 <http://spinrdf.org/sp#object> _:broken_var_s2 .
_:broken_var_r <http://spinrdf.org/sp#varName> "R"^^<http://www.w3.org/2001/XMLSchema#string> .
_:broken_var_s2 <http://spinrdf.org/sp#varName> "S"^^<http://www.w3.org/2001/XMLSchema#string> .
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


def write_tiny_real_style_framebase(framebase_dir: Path, *, spin_text: str = REAL_STYLE_SPIN_FIXTURE) -> None:
    _write_gzip_text(
        framebase_dir / "FrameBase_schema_core.ttl.gz",
        """
        <http://framebase.org/frame/Capability.can.verb> <http://www.w3.org/2000/01/rdf-schema#subClassOf> <http://www.w3.org/2000/01/rdf-schema#Resource> .
        <http://framebase.org/frame/Capability.enable.verb> <http://www.w3.org/2000/01/rdf-schema#subClassOf> <http://www.w3.org/2000/01/rdf-schema#Resource> .
        <http://framebase.org/fe/Capability.has_entity> <http://www.w3.org/2000/01/rdf-schema#subPropertyOf> <http://framebase.org/fe/Capability.has_entity> .
        <http://framebase.org/fe/Capability.has_event> <http://www.w3.org/2000/01/rdf-schema#subPropertyOf> <http://framebase.org/fe/Capability.has_event> .
        """,
    )
    _write_gzip_text(
        framebase_dir / "FrameBase_schema_dbps.ttl.gz",
        """
        <http://framebase.org/dbp/Capability.hasCapabilityForEvent> <http://framebase.org/meta/hasLexicalForm> "has capability for event"@en .
        <http://framebase.org/dbp/Capability.hasCapabilityForEvent> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://framebase.org/meta/DirectBinaryPredicateClass> .
        <http://framebase.org/dbp/Capability.enables> <http://framebase.org/meta/hasLexicalForm> "enables"@en .
        <http://framebase.org/dbp/Capability.enables> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://framebase.org/meta/DirectBinaryPredicateClass> .
        """,
    )
    _write_gzip_text(framebase_dir / "dereificationRulesSpinFormat.ttl.gz", spin_text)


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


def test_parse_real_style_spin_rule_fixture(tmp_path: Path) -> None:
    rules_path = tmp_path / "dereificationRulesSpinFormat.ttl.gz"
    _write_gzip_text(rules_path, REAL_STYLE_SPIN_FIXTURE)

    rules = list(parse_spin_dereification_rules(rules_path))

    assert len(rules) == 1
    rule = rules[0]
    assert rule.frame_name == "Capability"
    assert rule.microframe_name == "can.verb"
    assert rule.target_lemma_or_lu == "can"
    assert rule.subject_fe_name == "Entity"
    assert rule.object_fe_name == "Event"
    assert rule.dbp_predicate_iri == "http://framebase.org/dbp/Capability.hasCapabilityForEvent"
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


def test_build_framebase_index_loads_real_style_files(tmp_path: Path) -> None:
    framebase_dir = tmp_path / "framebase"
    write_tiny_real_style_framebase(framebase_dir, spin_text=REAL_STYLE_SPIN_FIXTURE_WITH_BROKEN_RULE)

    report = build_framebase_index(framebase_dir=framebase_dir, overwrite=True)

    assert report["frames"] >= 1
    assert report["dbp_labels"] >= 2
    assert report["dereification_rules_loaded"] == 1
    assert report["dereification_rules_parse_errors"] == 1
    assert report["dereification_rules_parse_warnings"] >= 1
    with sqlite3.connect(report["index_path"]) as conn:
        assert conn.execute("SELECT COUNT(*) FROM rules").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM dereification_rules").fetchone()[0] == 2


def test_build_framebase_index_spin_limit_restricts_rules(tmp_path: Path) -> None:
    framebase_dir = tmp_path / "framebase"
    write_tiny_real_style_framebase(framebase_dir, spin_text=REAL_STYLE_SPIN_FIXTURE_WITH_SECOND_RULE)

    report = build_framebase_index(framebase_dir=framebase_dir, overwrite=True, spin_limit=1)

    assert report["dereification_rules_loaded"] == 1
    assert report["spin_limit"] == 1
    with sqlite3.connect(report["index_path"]) as conn:
        assert conn.execute("SELECT COUNT(*) FROM rules").fetchone()[0] == 1


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


def test_inspect_rule_candidates_returns_matching_rules(tmp_path: Path) -> None:
    framebase_dir = tmp_path / "framebase"
    write_tiny_current_framebase(framebase_dir, spin_text=SPIN_FIXTURE_AMBIGUOUS)
    index_report = build_framebase_index(framebase_dir=framebase_dir, overwrite=True)

    payload = inspect_rule_candidates(
        index_report["index_path"],
        frame_name="Capability",
        subject_fe="Entity",
        object_fe="Event",
        target_text="can",
    )

    assert payload["candidate_count"] == 2
    assert payload["target_match_count"] == 1
    assert payload["resolution"] == "unique_target_match"
    assert "official DBP edge" in payload["suggested_next_action"]
    assert payload["candidates"][0]["target_matches"] is True
    assert "matches this rule" in payload["candidates"][0]["target_match_reason"]
    assert "not 'can'" in payload["candidates"][1]["target_match_reason"]


def test_build_framebase_index_uses_atomic_write(tmp_path: Path, monkeypatch) -> None:
    framebase_dir = tmp_path / "framebase"
    write_tiny_current_framebase(framebase_dir)
    index_path = tmp_path / "custom.sqlite"

    from fst2framegraph.framebase import index as index_module

    original = index_module._load_rules

    def broken_load_rules(path, labels, *, spin_limit=None, progress=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(index_module, "_load_rules", broken_load_rules)
    try:
        try:
            build_framebase_index(framebase_dir=framebase_dir, index_path=index_path, overwrite=True)
        except RuntimeError as exc:
            assert str(exc) == "boom"
        else:
            raise AssertionError("expected RuntimeError")
    finally:
        monkeypatch.setattr(index_module, "_load_rules", original)

    assert not index_path.exists()
    assert not any(index_path.parent.glob("custom.sqlite.tmp*"))
