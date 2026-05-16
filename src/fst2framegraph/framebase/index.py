from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from fst2framegraph.framebase.download import find_framebase_files, sha256_file
from fst2framegraph.framebase.load_dbp_labels import load_dbp_labels
from fst2framegraph.framebase.load_schema import FrameBaseSchema
from fst2framegraph.framebase.parse_dered_rules import parse_dered_rules
from fst2framegraph.schema import FrameBaseRule


FRAMEBASE_INDEX_NAME = "framebase_index.sqlite"
INDEX_SCHEMA_VERSION = "1"


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
    for table in ("metadata", "frames", "frame_elements", "dbp_labels", "rules", "manifest"):
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


def build_framebase_index(
    *,
    framebase_dir: str | Path = Path("data") / "framebase",
    index_path: str | Path | None = None,
    overwrite: bool = False,
    framebase_core: str | Path | None = None,
    dbp_labels: str | Path | None = None,
    dered_rules: str | Path | None = None,
) -> dict[str, Any]:
    """Build a compact SQLite lookup index from FrameBase source files.

    Dereification rules are optional: missing or unparsable rules never block
    schema indexing.
    """
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
    rules_path = Path(dered_rules) if dered_rules else found.get("dereification_rules_sparql")

    if core_path is None:
        raise FileNotFoundError(
            "FrameBase core schema not found. Provide --framebase-core or place it in framebase_dir."
        )

    index_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and index_path.exists():
        index_path.unlink()

    conn = _connect(index_path)
    statuses: dict[str, tuple[str, str | None]] = {}
    rules: list[FrameBaseRule] = []
    labels: dict[str, str] = {}
    try:
        _create_schema(conn)
        _clear_schema(conn)

        schema = FrameBaseSchema.from_turtle(core_path)
        for frame_name, frame_iri in schema.frame_lookup.items():
            conn.execute(
                """
                INSERT OR REPLACE INTO frames(frame_name, frame_iri, label)
                VALUES (?, ?, ?)
                """,
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

        if labels_path is not None:
            labels = load_dbp_labels(labels_path)
            for dbp_iri, label in labels.items():
                conn.execute(
                    "INSERT OR REPLACE INTO dbp_labels(dbp_iri, label) VALUES (?, ?)",
                    (dbp_iri, label),
                )
            statuses["dbp_labels"] = ("indexed", None)
        else:
            statuses["dbp_labels"] = ("missing", "DBP labels unavailable; labels fall back to IRIs.")

        if rules_path is not None:
            try:
                rules = parse_dered_rules(rules_path, labels)
                if not rules:
                    statuses["dereification_rules_sparql"] = (
                        "unavailable",
                        "Dereification rules unavailable; DBP dereified edges disabled: "
                        "no parseable rules found.",
                    )
                for rule in rules:
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
                            rule.dbp_iri,
                            rule.dbp_label,
                            rule.raw_rule,
                        ),
                    )
                if rules:
                    statuses["dereification_rules_sparql"] = ("indexed", None)
            except Exception as exc:
                statuses["dereification_rules_sparql"] = (
                    "unavailable",
                    f"Dereification rules unavailable; DBP dereified edges disabled: {exc}",
                )
                rules = []
        else:
            statuses["dereification_rules_sparql"] = (
                "missing",
                "Dereification rules unavailable; DBP dereified edges disabled.",
            )

        paths = {
            "core_schema": core_path,
            "dbp_labels": labels_path,
            "dereification_rules_sparql": rules_path,
        }
        _insert_manifest(conn, paths=paths, statuses=statuses)

        metadata = {
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "framebase_dir": str(framebase_dir),
            "frames": len(schema.frame_lookup),
            "frame_element_lookup_keys": len(schema.fe_lookup),
            "dbp_labels": len(labels),
            "rules": len(rules),
        }
        _write_metadata(conn, metadata)
        conn.commit()

        return {
            "index_path": str(index_path),
            "reused": False,
            **metadata,
            "warnings": [error for _, error in statuses.values() if error],
        }
    finally:
        conn.close()


def load_schema_from_index(index_path: str | Path) -> FrameBaseSchema:
    index_path = Path(index_path)
    conn = _connect(index_path)
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
    index_path = Path(index_path)
    conn = _connect(index_path)
    try:
        return {
            str(iri): str(label)
            for iri, label in conn.execute("SELECT dbp_iri, label FROM dbp_labels")
        }
    finally:
        conn.close()


def load_rules_from_index(index_path: str | Path) -> list[FrameBaseRule]:
    index_path = Path(index_path)
    conn = _connect(index_path)
    try:
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
                frame_iri=row[1],
                frame_name=row[2],
                subject_fe_iri=row[3],
                object_fe_iri=row[4],
                subject_fe_name=row[5],
                object_fe_name=row[6],
                dbp_iri=row[7],
                dbp_label=row[8],
                raw_rule=row[9],
            )
            for row in rows
        ]
    finally:
        conn.close()


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
