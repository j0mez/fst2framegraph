from __future__ import annotations

import gzip
import re
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import unquote_plus

from fst2framegraph.framebase.load_dbp_labels import label_for_dbp
from fst2framegraph.schema import FrameBaseRule


TOKEN = r"(?:<[^>]+>|[A-Za-z_][\w-]*:[^\s;,\[\]()]+)"
PREFIX_RE = re.compile(r"^(?:@prefix|PREFIX)\s+([A-Za-z_][\w-]*):\s*<([^>]+)>\s*\.?", re.I)
RULE_SUBJECT_RE = re.compile(r"^\s*(<[^>]+>|[A-Za-z_][\w-]*:[^\s;,\[\]()]+)")
VAR_NODE_RE = r"\[\s*sp:varName\s+\"(?P<var>[^\"]+)\"\s*\]"
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
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "sp": "http://spinrdf.org/sp#",
}
POS_SUFFIXES = (".verb", ".noun", ".adjective", ".adverb", ".v", ".n", ".a")


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


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

    with _open_text(path) as fh:
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
        return FrameBaseRule(
            rule_id=f"spin_{ordinal}",
            source_format="spin",
            source_file=source_file,
            frame_iri="",
            subject_fe_iri="",
            object_fe_iri="",
            dbp_predicate_iri="",
            raw_construct_node=rule_subject,
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
        if predicate == DEFAULT_PREFIXES["rdf"] + "type" and obj_value.startswith("http://framebase.org/frame/"):
            frame_iri = obj_value
        elif obj_value == "?S" and predicate.startswith("http://framebase.org/fe/"):
            subject_fe_iri = predicate
        elif obj_value == "?O" and predicate.startswith("http://framebase.org/fe/"):
            object_fe_iri = predicate

    if not all([frame_iri, subject_fe_iri, object_fe_iri, dbp_predicate_iri]):
        return FrameBaseRule(
            rule_id=f"spin_{ordinal}",
            source_format="spin",
            source_file=source_file,
            frame_iri=frame_iri,
            subject_fe_iri=subject_fe_iri,
            object_fe_iri=object_fe_iri,
            dbp_predicate_iri=dbp_predicate_iri,
            raw_construct_node=rule_subject,
            parse_status="skipped",
            parse_warning="Could not extract frame/type plus subject/object FE bindings.",
            raw_rule=block[:4000],
        )

    frame_name, microframe_name, target_lemma_or_lu = _frame_parts(frame_iri)
    dbp_predicate_name = _dbp_name(dbp_predicate_iri)
    return FrameBaseRule(
        rule_id=f"spin_{ordinal}",
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
        raw_construct_node=rule_subject,
        parse_status="parsed",
        parse_warning=None,
        raw_rule=block[:4000],
    )


def parse_spin_dereification_rules(
    path: Path | str | None,
    dbp_labels: dict[str, str] | None = None,
) -> Iterator[FrameBaseRule]:
    if path is None:
        return
    resolved = Path(path)
    labels = dbp_labels or {}
    for ordinal, (block, prefixes) in enumerate(_iter_rule_blocks(resolved), start=1):
        rule = _parse_block(
            block,
            prefixes=prefixes,
            source_file=str(resolved),
            dbp_labels=labels,
            ordinal=ordinal,
        )
        if rule is not None:
            yield rule
