from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable

from fst2framegraph.framebase.download import find_framebase_files, sha256_file
from fst2framegraph.framebase.load_dbp_labels import load_dbp_labels
from fst2framegraph.framebase.load_schema import FrameBaseSchema
from fst2framegraph.framebase.parse_dered_rules import parse_dered_rules
from fst2framegraph.framebase.parse_spin_rules import parse_spin_dereification_rules
from fst2framegraph.framebase.rule_index import normalise_match_text
from fst2framegraph.schema import FrameBaseRule


FRAMEBASE_INDEX_NAME = "framebase_index.sqlite"
INDEX_SCHEMA_VERSION = "2"


def default_framebase_index_path(framebase_dir: str | Path) -> Path:
    return Path(framebase_dir) / FRAMEBASE_INDEX_NAME


def find_framebase_index(framebase_dir: str | Path | None = None) -> Path | None:
    if framebase_dir is None:
        framebase_dir = Path("data") / "framebase"
    path = default_framebase_index_path(framebase_dir)
    return path if path.exists() else None


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA synchronous=FULL")
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        CREATE TABLE IF NOT EXISTS frames (
            frame_name TEXT PRIMARY KEY,
            frame_iri TEXT NOT NULL,
            label TEXT
        );
        CREATE TABLE IF NOT EXISTS frame_elements (
            lookup_frame_name TEXT NOT NULL,
            lookup_fe_name TEXT NOT NULL,
            fe_iri TEXT NOT NULL,
            source_frame_name TEXT,
            source_fe_name TEXT,
            PRIMARY KEY (lookup_frame_name, lookup_fe_name, fe_iri)
        );
        CREATE TABLE IF NOT EXISTS dbp_labels (
            dbp_iri TEXT PRIMARY KEY,
            label TEXT
        );
        CREATE TABLE IF NOT EXISTS dereification_rules (
            rule_id TEXT PRIMARY KEY,
            source_format TEXT,
            source_file TEXT,
            frame_iri TEXT,
            frame_name TEXT,
            microframe_name TEXT,
            target_lemma_or_lu TEXT,
            subject_fe_iri TEXT,
            subject_fe_name TEXT,
            object_fe_iri TEXT,
            object_fe_name TEXT,
            dbp_predicate_iri TEXT,
            dbp_predicate_name TEXT,
            raw_construct_node TEXT,
            parse_status TEXT,
            parse_warning TEXT,
            raw_rule TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_rules_frame_iri ON dereification_rules(frame_iri);
        CREATE INDEX IF NOT EXISTS idx_rules_frame_name ON dereification_rules(frame_name);
        CREATE INDEX IF NOT EXISTS idx_rules_frame_name_target ON dereification_rules(frame_name, target_lemma_or_lu);
        CREATE INDEX IF NOT EXISTS idx_rules_frame_fe_pair ON dereification_rules(frame_name, subject_fe_name, object_fe_name);
        CREATE INDEX IF NOT EXISTS idx_rules_frame_target_fe_pair ON dereification_rules(frame_name, target_lemma_or_lu, subject_fe_name, object_fe_name);
        CREATE INDEX IF NOT EXISTS idx_rules_fe_pair ON dereification_rules(subject_fe_name, object_fe_name);
        CREATE TABLE IF NOT EXISTS rules (
            rule_id TEXT PRIMARY KEY,
            frame_iri TEXT NOT NULL,
            frame_name TEXT,
            subject_fe_iri TEXT NOT NULL,
            object_fe_iri TEXT NOT NULL,
            subject_fe_name TEXT,
            object_fe_name TEXT,
            dbp_iri TEXT NOT NULL,
            dbp_label TEXT,
            raw_rule TEXT
        );
        CREATE TABLE IF NOT EXISTS clusters (
            parent_macroframe_iri TEXT,
            parent_macroframe_name TEXT,
            member_frame_iri TEXT,
            member_frame_name TEXT
        );
        CREATE TABLE IF NOT EXISTS cluster_pairs (
            source_frame TEXT,
            target_frame TEXT
        );
        CREATE TABLE IF NOT EXISTS lexical_clusters (
            cluster_representant_iri TEXT,
            member_frame_iri TEXT,
            lexical_form TEXT
        );
        CREATE TABLE IF NOT EXISTS manifest (
            key TEXT PRIMARY KEY,
            path TEXT,
            sha256 TEXT,
            size_bytes INTEGER,
            mtime_ns INTEGER,
            status TEXT,
            error TEXT
        );
        """
    )


def _clear_schema(conn: sqlite3.Connection) -> None:
    for table in (
        "metadata",
        "frames",
        "frame_elements",
        "dbp_labels",
        "dereification_rules",
        "rules",
        "clusters",
        "cluster_pairs",
        "lexical_clusters",
        "manifest",
    ):
        conn.execute(f"DELETE FROM {table}")


def _file_manifest_row(key: str, path: Path | None, status: str, error: str | None = None) -> tuple:
    if path is None:
        return (key, None, None, None, None, status, error)
    if not path.exists():
        return (key, str(path), None, None, None, "missing", error)
    stat = path.stat()
    return (
        key,
        str(path),
        sha256_file(path),
        stat.st_size,
        stat.st_mtime_ns,
        status,
        error,
    )


def _insert_manifest(
    conn: sqlite3.Connection,
    *,
    paths: dict[str, Path | None],
    statuses: dict[str, tuple[str, str | None]],
) -> None:
    for key, path in paths.items():
        status, error = statuses.get(key, ("available", None))
        conn.execute(
            """
            INSERT OR REPLACE INTO manifest (
                key, path, sha256, size_bytes, mtime_ns, status, error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            _file_manifest_row(key, path, status, error),
        )


def _write_metadata(conn: sqlite3.Connection, metadata: dict[str, Any]) -> None:
    for key, value in metadata.items():
        conn.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            (key, json.dumps(value, ensure_ascii=False)),
        )


def _frame_name_from_iri(frame_iri: str | None) -> str | None:
    if not frame_iri:
        return None
    tail = frame_iri.rsplit("/frame/", 1)[-1] if "/frame/" in frame_iri else frame_iri
    return tail.split(".", 1)[0] or None


def _load_rules(
    path: Path | None,
    labels: dict[str, str],
    *,
    spin_limit: int | None = None,
    progress: Callable[[dict[str, int | float | str]], None] | None = None,
) -> tuple[list[FrameBaseRule], str | None]:
    if path is None:
        return [], None
    lower_name = path.name.lower()
    if "spin" in lower_name or lower_name.endswith(".ttl") or lower_name.endswith(".ttl.gz"):
        return list(parse_spin_dereification_rules(path, labels, limit=spin_limit, progress=progress)), "spin"
    return parse_dered_rules(path, labels), "sparql"


def _parse_clusters_txt(path: Path) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    current_parent_iri: str | None = None
    current_parent_name: str | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("<parentMacroframe>") and line.endswith("</parentMacroframe>"):
            iri = line.removeprefix("<parentMacroframe>").removesuffix("</parentMacroframe>")
            current_parent_iri = iri
            current_parent_name = _frame_name_from_iri(iri)
            continue
        if line.startswith("http://framebase.org/frame/") and current_parent_iri:
            rows.append((current_parent_iri, current_parent_name or "", line, _frame_name_from_iri(line) or ""))
    return rows


def _parse_cluster_pairs(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            rows.append((parts[0], parts[1]))
    return rows


def _parse_lexical_clusters(path: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    current_representant: str | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("clusters.size()") or line.startswith("lexicalClusters.size()"):
            continue
        if line.startswith("### clusterRepresentant:"):
            current_representant = line.split(":", 1)[1].strip()
            continue
        if current_representant is None:
            continue
        if line.startswith("http://framebase.org/frame/"):
            rows.append((current_representant, line, ""))
        else:
            rows.append((current_representant, "", line))
    return rows


def build_framebase_index(
    *,
    framebase_dir: str | Path = Path("data") / "framebase",
    index_path: str | Path | None = None,
    overwrite: bool = False,
    framebase_core: str | Path | None = None,
    dbp_labels: str | Path | None = None,
    dered_rules: str | Path | None = None,
    spin_limit: int | None = None,
    progress: Callable[[dict[str, int | float | str]], None] | None = None,
) -> dict[str, Any]:
    framebase_dir = Path(framebase_dir)
    index_path = Path(index_path) if index_path else default_framebase_index_path(framebase_dir)

    if index_path.exists() and not overwrite:
        return {
            "index_path": str(index_path),
            "reused": True,
            "message": "FrameBase index already exists.",
        }

    found = find_framebase_files(framebase_dir)
    core_path = Path(framebase_core) if framebase_core else found.get("core_schema")
    labels_path = Path(dbp_labels) if dbp_labels else found.get("dbp_labels")
    rules_path = (
        Path(dered_rules)
        if dered_rules
        else found.get("dereification_rules_spin") or found.get("dereification_rules_sparql")
    )

    if core_path is None:
        raise FileNotFoundError(
            "FrameBase core schema not found. Provide --framebase-core or place it in framebase_dir."
        )

    index_path.parent.mkdir(parents=True, exist_ok=True)
    temp_index_path = index_path.with_name(f"{index_path.name}.tmp-{os.getpid()}-{int(time.time())}")
    if temp_index_path.exists():
        temp_index_path.unlink()

    start_time = time.monotonic()
    conn = _connect(temp_index_path)
    statuses: dict[str, tuple[str, str | None]] = {}
    warnings: list[str] = []
    try:
        _create_schema(conn)
        _clear_schema(conn)

        schema = FrameBaseSchema.from_turtle(core_path)
        for frame_name, frame_iri in schema.frame_lookup.items():
            conn.execute(
                "INSERT OR REPLACE INTO frames(frame_name, frame_iri, label) VALUES (?, ?, ?)",
                (frame_name, frame_iri, schema.labels.get(frame_iri)),
            )
        for (frame_name, fe_name), fe_iri in schema.fe_lookup.items():
            conn.execute(
                """
                INSERT OR IGNORE INTO frame_elements(
                    lookup_frame_name, lookup_fe_name, fe_iri, source_frame_name, source_fe_name
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (frame_name, fe_name, fe_iri, frame_name, fe_name),
            )
        statuses["core_schema"] = ("indexed", None)

        labels: dict[str, str] = {}
        if labels_path is not None:
            labels = load_dbp_labels(labels_path)
            for dbp_iri, label in labels.items():
                conn.execute(
                    "INSERT OR REPLACE INTO dbp_labels(dbp_iri, label) VALUES (?, ?)",
                    (dbp_iri, label),
                )
            statuses["dbp_labels"] = ("indexed", None)
        else:
            warning = "DBP labels unavailable; labels fall back to IRIs."
            statuses["dbp_labels"] = ("missing", warning)
            warnings.append(warning)

        spin_progress: dict[str, int | float | str] = {
            "phase": "spin-init",
            "lines_processed": 0,
            "construct_nodes_seen": 0,
            "candidate_rules_extracted": 0,
            "valid_rules_inserted": 0,
            "parse_warnings": 0,
            "parse_errors": 0,
            "elapsed_seconds": 0.0,
        }

        def _progress(payload: dict[str, int | float | str]) -> None:
            spin_progress.update(payload)
            if progress is not None:
                progress(dict(spin_progress))

        rules, rules_source_format = _load_rules(
            rules_path,
            labels,
            spin_limit=spin_limit,
            progress=_progress,
        )
        parsed_rules = 0
        parse_errors = 0
        parse_warnings = 0
        if rules_path is None:
            warning = "Dereification rules unavailable; DBP dereified edges disabled."
            statuses["dereification_rules_spin"] = ("missing", warning)
            statuses["dereification_rules_sparql"] = ("missing", None)
            warnings.append(warning)
        else:
            for rule in rules:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO dereification_rules(
                        rule_id, source_format, source_file, frame_iri, frame_name, microframe_name,
                        target_lemma_or_lu, subject_fe_iri, subject_fe_name, object_fe_iri,
                        object_fe_name, dbp_predicate_iri, dbp_predicate_name, raw_construct_node,
                        parse_status, parse_warning, raw_rule
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rule.rule_id,
                        rule.source_format,
                        rule.source_file,
                        rule.frame_iri,
                        rule.frame_name,
                        rule.microframe_name,
                        rule.target_lemma_or_lu,
                        rule.subject_fe_iri,
                        rule.subject_fe_name,
                        rule.object_fe_iri,
                        rule.object_fe_name,
                        rule.dbp_predicate_iri,
                        rule.dbp_predicate_name,
                        rule.raw_construct_node,
                        rule.parse_status,
                        rule.parse_warning,
                        rule.raw_rule,
                    ),
                )
                if rule.parse_status == "parsed":
                    parsed_rules += 1
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO rules(
                            rule_id, frame_iri, frame_name, subject_fe_iri, object_fe_iri,
                            subject_fe_name, object_fe_name, dbp_iri, dbp_label, raw_rule
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            rule.rule_id,
                            rule.frame_iri,
                            rule.frame_name,
                            rule.subject_fe_iri,
                            rule.object_fe_iri,
                            rule.subject_fe_name,
                            rule.object_fe_name,
                            rule.dbp_predicate_iri,
                            rule.dbp_label or rule.dbp_predicate_name,
                            rule.raw_rule,
                        ),
                    )
                else:
                    parse_errors += 1
                if rule.parse_warning:
                    parse_warnings += 1
                spin_progress["candidate_rules_extracted"] = parsed_rules + parse_errors
                spin_progress["valid_rules_inserted"] = parsed_rules
                spin_progress["parse_warnings"] = parse_warnings
                spin_progress["parse_errors"] = parse_errors
            manifest_key = "dereification_rules_spin" if rules_source_format == "spin" else "dereification_rules_sparql"
            if parsed_rules:
                statuses[manifest_key] = ("indexed", None)
            else:
                warning = "Dereification rules unavailable; DBP dereified edges disabled: no parseable rules found."
                statuses[manifest_key] = ("unavailable", warning)
                warnings.append(warning)
            other_key = "dereification_rules_sparql" if manifest_key == "dereification_rules_spin" else "dereification_rules_spin"
            statuses.setdefault(other_key, ("missing", None))

        cluster_files_loaded = False
        cluster_rows = 0
        cluster_pair_rows = 0
        lexical_cluster_rows = 0
        if found.get("clusters") and found["clusters"] is not None:
            for row in _parse_clusters_txt(found["clusters"]):
                conn.execute("INSERT INTO clusters VALUES (?, ?, ?, ?)", row)
                cluster_rows += 1
            statuses["clusters"] = ("indexed", None)
            cluster_files_loaded = True
        if found.get("cluster_pairs") and found["cluster_pairs"] is not None:
            for row in _parse_cluster_pairs(found["cluster_pairs"]):
                conn.execute("INSERT INTO cluster_pairs VALUES (?, ?)", row)
                cluster_pair_rows += 1
            statuses["cluster_pairs"] = ("indexed", None)
            cluster_files_loaded = True
        if found.get("lexical_clusters") and found["lexical_clusters"] is not None:
            for row in _parse_lexical_clusters(found["lexical_clusters"]):
                conn.execute("INSERT INTO lexical_clusters VALUES (?, ?, ?)", row)
                lexical_cluster_rows += 1
            statuses["lexical_clusters"] = ("indexed", None)
            cluster_files_loaded = True
        for optional_key in (
            "clusters",
            "cluster_pairs",
            "lexical_clusters",
            "manual_schema_extensions",
            "manual_inference_rules",
        ):
            statuses.setdefault(optional_key, ("missing", None))

        paths = {
            "core_schema": core_path,
            "dbp_labels": labels_path,
            "dereification_rules_spin": found.get("dereification_rules_spin"),
            "dereification_rules_sparql": found.get("dereification_rules_sparql"),
            "clusters": found.get("clusters"),
            "cluster_pairs": found.get("cluster_pairs"),
            "lexical_clusters": found.get("lexical_clusters"),
            "manual_schema_extensions": found.get("manual_schema_extensions"),
            "manual_inference_rules": found.get("manual_inference_rules"),
        }
        _insert_manifest(conn, paths=paths, statuses=statuses)

        conn.commit()
        if overwrite and index_path.exists():
            index_path.unlink()
        temp_index_path.replace(index_path)

        elapsed_seconds = round(time.monotonic() - start_time, 2)
        index_size_bytes = index_path.stat().st_size if index_path.exists() else 0
        metadata = {
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "framebase_dir": str(framebase_dir),
            "frames": len(schema.frame_lookup),
            "frame_element_lookup_keys": len(schema.fe_lookup),
            "dbp_labels": len(labels),
            "rules": parsed_rules,
            "dereification_rules_loaded": parsed_rules,
            "dereification_rules_source_format": rules_source_format,
            "dereification_rules_source_file": str(rules_path) if rules_path else None,
            "dereification_rules_sha256": sha256_file(rules_path) if rules_path and rules_path.exists() else None,
            "dereification_rules_parse_errors": parse_errors,
            "dereification_rules_parse_warnings": parse_warnings,
            "cluster_files_loaded": cluster_files_loaded,
            "cluster_rows_loaded": cluster_rows,
            "cluster_pair_rows_loaded": cluster_pair_rows,
            "lexical_cluster_rows_loaded": lexical_cluster_rows,
            "dbp_schema_loaded": labels_path is not None,
            "core_schema_loaded": core_path is not None,
            "spin_limit": spin_limit,
            "spin_lines_processed": spin_progress.get("lines_processed", 0),
            "spin_construct_nodes_seen": spin_progress.get("construct_nodes_seen", 0),
            "spin_candidate_rules_extracted": spin_progress.get("candidate_rules_extracted", 0),
            "elapsed_seconds": elapsed_seconds,
            "index_size_bytes": index_size_bytes,
        }
        post_conn = _connect(index_path)
        try:
            _write_metadata(post_conn, metadata)
            post_conn.commit()
        finally:
            post_conn.close()

        return {
            "index_path": str(index_path),
            "reused": False,
            **metadata,
            "warnings": warnings,
        }
    except Exception:
        temp_index_path.unlink(missing_ok=True)
        raise
    finally:
        conn.close()


def load_schema_from_index(index_path: str | Path) -> FrameBaseSchema:
    conn = _connect(Path(index_path))
    try:
        frame_lookup = {
            str(name): str(iri)
            for name, iri in conn.execute("SELECT frame_name, frame_iri FROM frames")
        }
        fe_lookup = {
            (str(frame_name), str(fe_name)): str(iri)
            for frame_name, fe_name, iri in conn.execute(
                "SELECT lookup_frame_name, lookup_fe_name, fe_iri FROM frame_elements"
            )
        }
        return FrameBaseSchema(frame_lookup=frame_lookup, fe_lookup=fe_lookup)
    finally:
        conn.close()


def load_dbp_labels_from_index(index_path: str | Path) -> dict[str, str]:
    conn = _connect(Path(index_path))
    try:
        return {
            str(iri): str(label)
            for iri, label in conn.execute("SELECT dbp_iri, label FROM dbp_labels")
        }
    finally:
        conn.close()


def load_rules_from_index(index_path: str | Path) -> list[FrameBaseRule]:
    conn = _connect(Path(index_path))
    try:
        try:
            rows = conn.execute(
                """
                SELECT rule_id, source_format, source_file, frame_iri, frame_name, microframe_name,
                       target_lemma_or_lu, subject_fe_iri, subject_fe_name, object_fe_iri,
                       object_fe_name, dbp_predicate_iri, dbp_predicate_name, raw_construct_node,
                       parse_status, parse_warning, raw_rule
                FROM dereification_rules
                WHERE COALESCE(parse_status, 'parsed') = 'parsed'
                """
            ).fetchall()
            return [
                FrameBaseRule(
                    rule_id=row[0],
                    source_format=row[1],
                    source_file=row[2],
                    frame_iri=row[3],
                    frame_name=row[4],
                    microframe_name=row[5],
                    target_lemma_or_lu=row[6],
                    subject_fe_iri=row[7],
                    subject_fe_name=row[8],
                    object_fe_iri=row[9],
                    object_fe_name=row[10],
                    dbp_predicate_iri=row[11],
                    dbp_predicate_name=row[12],
                    dbp_iri=row[11],
                    dbp_label=row[12],
                    raw_construct_node=row[13],
                    parse_status=row[14],
                    parse_warning=row[15],
                    raw_rule=row[16],
                )
                for row in rows
            ]
        except sqlite3.OperationalError:
            rows = conn.execute(
                """
                SELECT rule_id, frame_iri, frame_name, subject_fe_iri, object_fe_iri,
                       subject_fe_name, object_fe_name, dbp_iri, dbp_label, raw_rule
                FROM rules
                """
            ).fetchall()
            return [
                FrameBaseRule(
                    rule_id=row[0],
                    source_format="sparql",
                    frame_iri=row[1],
                    frame_name=row[2],
                    subject_fe_iri=row[3],
                    object_fe_iri=row[4],
                    subject_fe_name=row[5],
                    object_fe_name=row[6],
                    dbp_predicate_iri=row[7],
                    dbp_predicate_name=row[8],
                    dbp_iri=row[7],
                    dbp_label=row[8],
                    parse_status="parsed",
                    raw_rule=row[9],
                )
                for row in rows
            ]
    finally:
        conn.close()


def inspect_rule_candidates(
    index_path: str | Path,
    *,
    frame_name: str,
    subject_fe: str,
    object_fe: str,
    target_text: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    rules = load_rules_from_index(index_path)
    frame_norm = normalise_match_text(frame_name)
    subject_norm = normalise_match_text(subject_fe)
    object_norm = normalise_match_text(object_fe)
    target_norm = normalise_match_text(target_text or "")

    candidates = []
    for rule in rules:
        if normalise_match_text(rule.frame_name or "") != frame_norm:
            continue
        if normalise_match_text(rule.subject_fe_name or "") != subject_norm:
            continue
        if normalise_match_text(rule.object_fe_name or "") != object_norm:
            continue
        rule_target = normalise_match_text(rule.target_lemma_or_lu or rule.microframe_name or "")
        candidates.append(
            {
                "rule_id": rule.rule_id,
                "frame_name": rule.frame_name,
                "microframe_name": rule.microframe_name,
                "target_lemma_or_lu": rule.target_lemma_or_lu,
                "subject_fe_name": rule.subject_fe_name,
                "object_fe_name": rule.object_fe_name,
                "dbp_predicate_iri": rule.dbp_predicate_iri,
                "dbp_predicate_name": rule.dbp_predicate_name,
                "target_matches": bool(target_norm and rule_target == target_norm),
            }
        )

    candidates.sort(
        key=lambda item: (
            not item["target_matches"],
            str(item["target_lemma_or_lu"] or ""),
            str(item["dbp_predicate_name"] or ""),
        )
    )
    return {
        "index_path": str(index_path),
        "frame_name": frame_name,
        "subject_fe": subject_fe,
        "object_fe": object_fe,
        "target_text": target_text,
        "candidate_count": len(candidates),
        "candidates": candidates[: max(limit, 0)],
    }


def setup_framebase(
    framebase_dir: str | Path = Path("data") / "framebase",
    *,
    build_index: bool = True,
    reuse_existing: bool = True,
    index_path: str | Path | None = None,
) -> dict[str, Any]:
    framebase_dir = Path(framebase_dir)
    found = find_framebase_files(framebase_dir)
    report: dict[str, Any] = {
        "framebase_dir": str(framebase_dir),
        "files": {k: str(v) if v else None for k, v in found.items()},
    }
    if build_index:
        index_path = Path(index_path) if index_path else default_framebase_index_path(framebase_dir)
        report["index"] = build_framebase_index(
            framebase_dir=framebase_dir,
            index_path=index_path,
            overwrite=not reuse_existing,
        )
    return report
