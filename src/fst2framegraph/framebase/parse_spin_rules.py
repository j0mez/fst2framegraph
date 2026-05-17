from __future__ import annotations

import re
import sqlite3
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from urllib.parse import unquote_plus

from fst2framegraph.framebase.load_dbp_labels import label_for_dbp
from fst2framegraph.framebase.stream_rdf import (
    RDF_FIRST,
    RDF_NIL,
    RDF_REST,
    RDF_TYPE,
    iter_triples,
    open_text,
)
from fst2framegraph.schema import FrameBaseRule


TOKEN = r"(?:<[^>]+>|[A-Za-z_][\w-]*:[^\s;,\[\]()]+)"
PREFIX_RE = re.compile(r"^(?:@prefix|PREFIX)\s+([A-Za-z_][\w-]*):\s*<([^>]+)>\s*\.?", re.I)
RULE_SUBJECT_RE = re.compile(r"^\s*(<[^>]+>|[A-Za-z_][\w-]*:[^\s;,\[\]()]+)")
TEMPLATE_RE = re.compile(
    rf"sp:subject\s+\[\s*sp:varName\s+\"S\"\s*\]\s*;\s*"
    rf"sp:predicate\s+(?P<predicate>{TOKEN})\s*;\s*"
    rf"sp:object\s+\[\s*sp:varName\s+\"O\"\s*\]",
    re.S,
)
WHERE_ROW_RE = re.compile(
    rf"sp:subject\s+\[\s*sp:varName\s+\"(?P<subject>[^\"]+)\"\s*\]\s*;\s*"
    rf"sp:predicate\s+(?P<predicate>{TOKEN})\s*;\s*"
    rf"sp:object\s+(?P<object>{TOKEN}|\[\s*sp:varName\s+\"[^\"]+\"\s*\])",
    re.S,
)

DEFAULT_PREFIXES = {
    "rdf": RDF_TYPE.rsplit("#", 1)[0] + "#",
    "sp": "http://spinrdf.org/sp#",
}
POS_SUFFIXES = (".verb", ".noun", ".adjective", ".adverb", ".v", ".n", ".a")

SP_CONSTRUCT = "http://spinrdf.org/sp#Construct"
SP_TEMPLATES = "http://spinrdf.org/sp#templates"
SP_WHERE = "http://spinrdf.org/sp#where"
SP_SUBJECT = "http://spinrdf.org/sp#subject"
SP_PREDICATE = "http://spinrdf.org/sp#predicate"
SP_OBJECT = "http://spinrdf.org/sp#object"
SP_VAR_NAME = "http://spinrdf.org/sp#varName"

ProgressCallback = Callable[[dict[str, int | float | str]], None]


def _resolve_token(token: str, prefixes: dict[str, str]) -> str:
    token = token.strip().rstrip(";,")
    if token.startswith("<") and token.endswith(">"):
        return token[1:-1]
    if ":" in token:
        prefix, local = token.split(":", 1)
        base = prefixes.get(prefix)
        if base is not None:
            return base + local
    return token


def _decode_iri_tail(value: str) -> str:
    return unquote_plus(value).strip()


def _frame_parts(frame_iri: str) -> tuple[str | None, str | None, str | None]:
    tail = _decode_iri_tail(frame_iri.rsplit("/frame/", 1)[-1] if "/frame/" in frame_iri else frame_iri)
    if not tail:
        return None, None, None
    frame_name, _, rest = tail.partition(".")
    microframe = rest or None
    lemma = None
    if microframe:
        lemma = microframe
        for suffix in POS_SUFFIXES:
            if lemma.endswith(suffix):
                lemma = lemma[: -len(suffix)]
                break
        lemma = lemma.split(".", 1)[0].replace("_", " ").strip() or None
    return frame_name or None, microframe, lemma


def _fe_name(fe_iri: str) -> str | None:
    tail = _decode_iri_tail(fe_iri.rsplit("/fe/", 1)[-1] if "/fe/" in fe_iri else fe_iri)
    if "." in tail:
        _, tail = tail.split(".", 1)
    tail = tail.removeprefix("has_").replace("_", " ").strip()
    return " ".join(part.capitalize() for part in tail.split()) or None


def _dbp_name(dbp_iri: str) -> str | None:
    tail = _decode_iri_tail(dbp_iri.rsplit("/dbp/", 1)[-1] if "/dbp/" in dbp_iri else dbp_iri)
    if "." in tail:
        _, tail = tail.split(".", 1)
    return tail or None


def _parse_prefixes(lines: list[str]) -> dict[str, str]:
    prefixes = dict(DEFAULT_PREFIXES)
    for line in lines:
        match = PREFIX_RE.match(line.strip())
        if match:
            prefixes[match.group(1)] = match.group(2)
    return prefixes


def _count_delta(text: str) -> int:
    delta = 0
    in_string = False
    escaped = False
    for char in text:
        if char == "\\" and not escaped:
            escaped = True
            continue
        if char == '"' and not escaped:
            in_string = not in_string
        elif not in_string and char in "[(":
            delta += 1
        elif not in_string and char in "])":
            delta -= 1
        escaped = False
    return delta


def _iter_rule_blocks(path: Path) -> Iterator[tuple[str, dict[str, str]]]:
    prefix_lines: list[str] = []
    prefixes = dict(DEFAULT_PREFIXES)
    current: list[str] = []
    depth = 0
    collecting = False

    with open_text(path) as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                if collecting:
                    current.append(line)
                continue

            if not collecting:
                prefix_match = PREFIX_RE.match(stripped)
                if prefix_match:
                    prefix_lines.append(stripped)
                    prefixes = _parse_prefixes(prefix_lines)
                    continue
                if stripped.startswith("#"):
                    continue
                if RULE_SUBJECT_RE.match(stripped) is None:
                    continue
                collecting = True
                current = [line]
                depth = _count_delta(stripped)
                if depth <= 0 and stripped.endswith("."):
                    yield ("\n".join(current), dict(prefixes))
                    collecting = False
                    current = []
                    depth = 0
                continue

            current.append(line)
            depth += _count_delta(stripped)
            if depth <= 0 and stripped.endswith("."):
                yield ("\n".join(current), dict(prefixes))
                collecting = False
                current = []
                depth = 0


def _where_rows(block: str, prefixes: dict[str, str]) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for match in WHERE_ROW_RE.finditer(block):
        subject_var = match.group("subject")
        predicate = _resolve_token(match.group("predicate"), prefixes)
        obj_token = match.group("object").strip()
        var_match = re.search(r'sp:varName\s+"([^"]+)"', obj_token)
        if var_match:
            obj_value = f"?{var_match.group(1)}"
        else:
            obj_value = _resolve_token(obj_token, prefixes)
        rows.append((subject_var, predicate, obj_value))
    return rows


def _build_rule(
    *,
    rule_id: str,
    source_file: str,
    raw_construct_node: str,
    dbp_predicate_iri: str,
    frame_iri: str,
    subject_fe_iri: str,
    object_fe_iri: str,
    dbp_labels: dict[str, str],
    parse_status: str = "parsed",
    parse_warning: str | None = None,
    raw_rule: str | None = None,
) -> FrameBaseRule:
    frame_name, microframe_name, target_lemma_or_lu = _frame_parts(frame_iri)
    dbp_predicate_name = _dbp_name(dbp_predicate_iri)
    return FrameBaseRule(
        rule_id=rule_id,
        source_format="spin",
        source_file=source_file,
        frame_iri=frame_iri,
        frame_name=frame_name,
        microframe_name=microframe_name,
        target_lemma_or_lu=target_lemma_or_lu,
        subject_fe_iri=subject_fe_iri,
        object_fe_iri=object_fe_iri,
        subject_fe_name=_fe_name(subject_fe_iri),
        object_fe_name=_fe_name(object_fe_iri),
        dbp_predicate_iri=dbp_predicate_iri,
        dbp_predicate_name=dbp_predicate_name,
        dbp_iri=dbp_predicate_iri,
        dbp_label=label_for_dbp(dbp_predicate_iri, dbp_labels) if dbp_labels else dbp_predicate_name,
        raw_construct_node=raw_construct_node,
        parse_status=parse_status,
        parse_warning=parse_warning,
        raw_rule=raw_rule,
    )


def _parse_block(
    block: str,
    *,
    prefixes: dict[str, str],
    source_file: str,
    dbp_labels: dict[str, str],
    ordinal: int,
) -> FrameBaseRule | None:
    if "sp:Construct" not in block:
        return None
    subject_match = RULE_SUBJECT_RE.match(block)
    if not subject_match:
        return None
    rule_subject = _resolve_token(subject_match.group(1), prefixes)

    template_match = TEMPLATE_RE.search(block)
    if not template_match:
        return _build_rule(
            rule_id=f"spin_{ordinal}",
            source_file=source_file,
            raw_construct_node=rule_subject,
            dbp_predicate_iri="",
            frame_iri="",
            subject_fe_iri="",
            object_fe_iri="",
            dbp_labels=dbp_labels,
            parse_status="skipped",
            parse_warning="No parseable SPIN template found.",
            raw_rule=block[:4000],
        )

    dbp_predicate_iri = _resolve_token(template_match.group("predicate"), prefixes)

    frame_iri = ""
    subject_fe_iri = ""
    object_fe_iri = ""
    for subject_var, predicate, obj_value in _where_rows(block, prefixes):
        if subject_var != "R":
            continue
        if predicate == RDF_TYPE and obj_value.startswith("http://framebase.org/frame/"):
            frame_iri = obj_value
        elif obj_value == "?S" and predicate.startswith("http://framebase.org/fe/"):
            subject_fe_iri = predicate
        elif obj_value == "?O" and predicate.startswith("http://framebase.org/fe/"):
            object_fe_iri = predicate

    if not all([frame_iri, subject_fe_iri, object_fe_iri, dbp_predicate_iri]):
        return _build_rule(
            rule_id=f"spin_{ordinal}",
            source_file=source_file,
            raw_construct_node=rule_subject,
            dbp_predicate_iri=dbp_predicate_iri,
            frame_iri=frame_iri,
            subject_fe_iri=subject_fe_iri,
            object_fe_iri=object_fe_iri,
            dbp_labels=dbp_labels,
            parse_status="skipped",
            parse_warning="Could not extract frame/type plus subject/object FE bindings.",
            raw_rule=block[:4000],
        )

    return _build_rule(
        rule_id=f"spin_{ordinal}",
        source_file=source_file,
        raw_construct_node=rule_subject,
        dbp_predicate_iri=dbp_predicate_iri,
        frame_iri=frame_iri,
        subject_fe_iri=subject_fe_iri,
        object_fe_iri=object_fe_iri,
        dbp_labels=dbp_labels,
        raw_rule=block[:4000],
    )


def _looks_like_ntriples_spin(path: Path) -> bool:
    with open_text(path) as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("@prefix") or stripped.startswith("PREFIX"):
                return False
            if "http://spinrdf.org/sp#" in stripped:
                return True
    return False


def _emit_progress(progress: ProgressCallback | None, payload: dict[str, int | float | str]) -> None:
    if progress is not None:
        progress(payload)


def _ensure_temp_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE constructs (
            id TEXT PRIMARY KEY,
            templates TEXT,
            where_root TEXT
        );
        CREATE TABLE list_nodes (
            id TEXT PRIMARY KEY,
            first_item TEXT,
            rest TEXT
        );
        CREATE TABLE triple_nodes (
            id TEXT PRIMARY KEY,
            subj TEXT,
            pred TEXT,
            obj TEXT
        );
        CREATE TABLE var_nodes (
            id TEXT PRIMARY KEY,
            var_name TEXT
        );
        """
    )


def _upsert_construct(conn: sqlite3.Connection, subject: str, *, templates: str | None = None, where_root: str | None = None) -> None:
    conn.execute(
        """
        INSERT INTO constructs(id, templates, where_root) VALUES (?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            templates = COALESCE(excluded.templates, constructs.templates),
            where_root = COALESCE(excluded.where_root, constructs.where_root)
        """,
        (subject, templates, where_root),
    )


def _upsert_list_node(conn: sqlite3.Connection, subject: str, *, first_item: str | None = None, rest: str | None = None) -> None:
    conn.execute(
        """
        INSERT INTO list_nodes(id, first_item, rest) VALUES (?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            first_item = COALESCE(excluded.first_item, list_nodes.first_item),
            rest = COALESCE(excluded.rest, list_nodes.rest)
        """,
        (subject, first_item, rest),
    )


def _upsert_triple_node(conn: sqlite3.Connection, subject: str, *, subj: str | None = None, pred: str | None = None, obj: str | None = None) -> None:
    conn.execute(
        """
        INSERT INTO triple_nodes(id, subj, pred, obj) VALUES (?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            subj = COALESCE(excluded.subj, triple_nodes.subj),
            pred = COALESCE(excluded.pred, triple_nodes.pred),
            obj = COALESCE(excluded.obj, triple_nodes.obj)
        """,
        (subject, subj, pred, obj),
    )


def _resolve_var(conn: sqlite3.Connection, token: str, cache: dict[str, str]) -> str:
    if not token.startswith("_:"):
        return token
    if token in cache:
        return cache[token]
    row = conn.execute("SELECT var_name FROM var_nodes WHERE id = ?", (token,)).fetchone()
    if row and row[0]:
        cache[token] = f"?{row[0]}"
    else:
        cache[token] = token
    return cache[token]


def _list_items(conn: sqlite3.Connection, root: str, cache: dict[str, tuple[str | None, str | None]]) -> list[str]:
    items: list[str] = []
    node = root
    visited: set[str] = set()
    while node and node != RDF_NIL and node not in visited:
        visited.add(node)
        entry = cache.get(node)
        if entry is None:
            row = conn.execute("SELECT first_item, rest FROM list_nodes WHERE id = ?", (node,)).fetchone()
            entry = (str(row[0]) if row and row[0] is not None else None, str(row[1]) if row and row[1] is not None else None)
            cache[node] = entry
        first_item, rest = entry
        if first_item:
            items.append(first_item)
        if not rest:
            break
        node = rest
    return items


def _triple_node(conn: sqlite3.Connection, node_id: str, cache: dict[str, tuple[str | None, str | None, str | None]]) -> tuple[str | None, str | None, str | None]:
    entry = cache.get(node_id)
    if entry is not None:
        return entry
    row = conn.execute("SELECT subj, pred, obj FROM triple_nodes WHERE id = ?", (node_id,)).fetchone()
    entry = (
        str(row[0]) if row and row[0] is not None else None,
        str(row[1]) if row and row[1] is not None else None,
        str(row[2]) if row and row[2] is not None else None,
    )
    cache[node_id] = entry
    return entry


def _parse_ntriples_rules(
    path: Path,
    *,
    dbp_labels: dict[str, str],
    limit: int | None = None,
    progress: ProgressCallback | None = None,
) -> Iterator[FrameBaseRule]:
    temp_file = tempfile.NamedTemporaryFile(prefix="fst2framegraph_spin_", suffix=".sqlite", delete=False)
    temp_file.close()
    conn = sqlite3.connect(temp_file.name)
    line_count = 0
    construct_nodes_seen = 0
    last_progress = time.monotonic()
    try:
        _ensure_temp_schema(conn)
        for triple in iter_triples(path):
            line_count += 1
            if triple.predicate == RDF_TYPE and triple.object == SP_CONSTRUCT:
                construct_nodes_seen += 1
                _upsert_construct(conn, triple.subject)
            elif triple.predicate == SP_TEMPLATES:
                _upsert_construct(conn, triple.subject, templates=triple.object)
            elif triple.predicate == SP_WHERE:
                _upsert_construct(conn, triple.subject, where_root=triple.object)
            elif triple.predicate == RDF_FIRST:
                _upsert_list_node(conn, triple.subject, first_item=triple.object)
            elif triple.predicate == RDF_REST:
                _upsert_list_node(conn, triple.subject, rest=triple.object)
            elif triple.predicate == SP_SUBJECT:
                _upsert_triple_node(conn, triple.subject, subj=triple.object)
            elif triple.predicate == SP_PREDICATE:
                _upsert_triple_node(conn, triple.subject, pred=triple.object)
            elif triple.predicate == SP_OBJECT:
                _upsert_triple_node(conn, triple.subject, obj=triple.object)
            elif triple.predicate == SP_VAR_NAME and triple.object_kind == "literal":
                conn.execute(
                    "INSERT OR REPLACE INTO var_nodes(id, var_name) VALUES (?, ?)",
                    (triple.subject, triple.object),
                )
            if line_count % 50000 == 0:
                conn.commit()
            now = time.monotonic()
            if now - last_progress >= 5:
                _emit_progress(
                    progress,
                    {
                        "phase": "spin-scan",
                        "lines_processed": line_count,
                        "construct_nodes_seen": construct_nodes_seen,
                        "candidate_rules_extracted": 0,
                        "valid_rules_inserted": 0,
                        "parse_warnings": 0,
                        "parse_errors": 0,
                        "elapsed_seconds": round(now, 2),
                    },
                )
                last_progress = now
        conn.commit()

        list_cache: dict[str, tuple[str | None, str | None]] = {}
        triple_cache: dict[str, tuple[str | None, str | None, str | None]] = {}
        var_cache: dict[str, str] = {}
        parsed = 0
        warnings = 0
        errors = 0
        source_file = str(path)
        rows = conn.execute(
            "SELECT id, templates, where_root FROM constructs WHERE templates IS NOT NULL AND where_root IS NOT NULL ORDER BY id"
        )
        for ordinal, (construct_id, templates_root, where_root) in enumerate(rows, start=1):
            if limit is not None and ordinal > limit:
                break
            template_items = _list_items(conn, str(templates_root), list_cache)
            dbp_predicate_iri = ""
            for template_node in template_items:
                subject_ref, predicate_ref, object_ref = _triple_node(conn, template_node, triple_cache)
                if not predicate_ref:
                    continue
                subject_value = _resolve_var(conn, subject_ref or "", var_cache)
                object_value = _resolve_var(conn, object_ref or "", var_cache)
                if subject_value == "?S" and object_value == "?O":
                    dbp_predicate_iri = predicate_ref
                    break

            frame_iri = ""
            subject_fe_iri = ""
            object_fe_iri = ""
            for where_node in _list_items(conn, str(where_root), list_cache):
                subject_ref, predicate_ref, object_ref = _triple_node(conn, where_node, triple_cache)
                if not predicate_ref:
                    continue
                subject_value = _resolve_var(conn, subject_ref or "", var_cache)
                object_value = _resolve_var(conn, object_ref or "", var_cache)
                if subject_value != "?R":
                    continue
                if predicate_ref == RDF_TYPE and object_value.startswith("http://framebase.org/frame/"):
                    frame_iri = object_value
                elif object_value == "?S" and predicate_ref.startswith("http://framebase.org/fe/"):
                    subject_fe_iri = predicate_ref
                elif object_value == "?O" and predicate_ref.startswith("http://framebase.org/fe/"):
                    object_fe_iri = predicate_ref

            if all([frame_iri, subject_fe_iri, object_fe_iri, dbp_predicate_iri]):
                parsed += 1
                yield _build_rule(
                    rule_id=f"spin_{ordinal}",
                    source_file=source_file,
                    raw_construct_node=str(construct_id),
                    dbp_predicate_iri=dbp_predicate_iri,
                    frame_iri=frame_iri,
                    subject_fe_iri=subject_fe_iri,
                    object_fe_iri=object_fe_iri,
                    dbp_labels=dbp_labels,
                    raw_rule=f"construct={construct_id};templates={templates_root};where={where_root}",
                )
            else:
                errors += 1
                warnings += 1
                yield _build_rule(
                    rule_id=f"spin_{ordinal}",
                    source_file=source_file,
                    raw_construct_node=str(construct_id),
                    dbp_predicate_iri=dbp_predicate_iri,
                    frame_iri=frame_iri,
                    subject_fe_iri=subject_fe_iri,
                    object_fe_iri=object_fe_iri,
                    dbp_labels=dbp_labels,
                    parse_status="skipped",
                    parse_warning="Could not extract frame/type plus subject/object FE bindings.",
                    raw_rule=f"construct={construct_id};templates={templates_root};where={where_root}",
                )
            now = time.monotonic()
            if now - last_progress >= 5:
                _emit_progress(
                    progress,
                    {
                        "phase": "spin-extract",
                        "lines_processed": line_count,
                        "construct_nodes_seen": construct_nodes_seen,
                        "candidate_rules_extracted": ordinal,
                        "valid_rules_inserted": parsed,
                        "parse_warnings": warnings,
                        "parse_errors": errors,
                        "elapsed_seconds": round(now, 2),
                    },
                )
                last_progress = now
    finally:
        conn.close()
        Path(temp_file.name).unlink(missing_ok=True)


def parse_spin_dereification_rules(
    path: Path | str | None,
    dbp_labels: dict[str, str] | None = None,
    *,
    limit: int | None = None,
    progress: ProgressCallback | None = None,
) -> Iterator[FrameBaseRule]:
    if path is None:
        return
    resolved = Path(path)
    labels = dbp_labels or {}
    if _looks_like_ntriples_spin(resolved):
        yield from _parse_ntriples_rules(resolved, dbp_labels=labels, limit=limit, progress=progress)
        return

    for ordinal, (block, prefixes) in enumerate(_iter_rule_blocks(resolved), start=1):
        if limit is not None and ordinal > limit:
            break
        rule = _parse_block(
            block,
            prefixes=prefixes,
            source_file=str(resolved),
            dbp_labels=labels,
            ordinal=ordinal,
        )
        if rule is not None:
            yield rule
