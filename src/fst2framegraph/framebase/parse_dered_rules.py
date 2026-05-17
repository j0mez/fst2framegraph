from __future__ import annotations

import re
import zipfile
from pathlib import Path

from fst2framegraph.framebase.load_dbp_labels import label_for_dbp
from fst2framegraph.schema import FrameBaseRule

# Accept full IRIs, prefixed names and the older FrameBase colon-local style.
TOKEN = r"(?:<[^>]+>|[A-Za-z_][\w-]*:[^\s;{}()]+|:[^\s;{}()]+)"
PREFIX_RE = re.compile(r"PREFIX\s+([A-Za-z_][\w-]*|):\s*<([^>]+)>", re.I)
CONSTRUCT_RE = re.compile(r"(?=\bCONSTRUCT\s*\{)", re.I)
TYPE_RE = re.compile(
    r"\?f\s+(?:a|rdf:type)(?:\s*/\s*rdfs:subClassOf\s*\*)?\s+(" + TOKEN + r")",
    re.I,
)
FE_RE = re.compile(r"\?f\s+(" + TOKEN + r")\s+\?(\w+)\b", re.I)
DBP_RE = re.compile(r"\?(\w+)\s+(" + TOKEN + r")\s+\?(\w+)\b", re.I)

DEFAULT_PREFIXES = {
    "": "http://framebase.org/",
    "fb": "http://framebase.org/",
    "fbframe": "http://framebase.org/frame/",
    "fbfe": "http://framebase.org/fe/",
    "fbdbp": "http://framebase.org/dbp/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
}


def _read_rule_files(path: Path) -> list[tuple[str, str]]:
    if path.suffix.lower() == ".zip":
        out = []
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                try:
                    text = zf.read(name).decode("utf-8", errors="replace")
                except UnicodeDecodeError:
                    text = zf.read(name).decode("latin-1", errors="replace")
                out.append((name, text))
        return out
    return [(path.name, path.read_text(encoding="utf-8", errors="replace"))]


def _prefixes_from_text(text: str) -> dict[str, str]:
    prefixes = dict(DEFAULT_PREFIXES)
    for prefix, iri in PREFIX_RE.findall(text):
        prefixes[prefix] = iri
    return prefixes


def _split_sparql_constructs(text: str) -> list[str]:
    chunks = CONSTRUCT_RE.split(text)
    return [c.strip() for c in chunks if c.strip().lower().startswith("construct")]


def _resolve_token(token: str, prefixes: dict[str, str]) -> str:
    token = token.strip().rstrip(".;,")
    if token.startswith("<") and token.endswith(">"):
        return token[1:-1]
    if ":" in token:
        prefix, local = token.split(":", 1)
        base = prefixes.get(prefix)
        if base is not None:
            return base + local
    return token


def _tail(resource: str) -> str:
    resource = resource.rstrip("/>")
    if "#" in resource:
        return resource.rsplit("#", 1)[-1]
    return resource.rsplit("/", 1)[-1]


def _clean_resource_name(value: str) -> str:
    return value.replace("+", " ").replace("%20", " ").strip()


def _frame_name_from_resource(resource: str) -> str | None:
    resource = _clean_resource_name(resource)
    if "/frame/" in resource:
        tail = resource.rsplit("/frame/", 1)[-1]
    else:
        tail = _tail(resource)
    # Old style: Microframe.Separating.verb.partition
    if tail.startswith("Microframe."):
        parts = tail.split(".")
        return parts[1] if len(parts) > 1 else None
    if tail.startswith("frame."):
        parts = tail.split(".")
        return parts[1] if len(parts) > 1 else None
    # Current paths often use FrameName or FrameName.m.lexeme.pos.
    return tail.split(".", 1)[0] or None


def _fe_parts_from_resource(resource: str) -> tuple[str | None, str | None]:
    resource = _clean_resource_name(resource)
    if "/fe/" in resource:
        tail = resource.rsplit("/fe/", 1)[-1]
    else:
        tail = _tail(resource)
    # Old style: fe.Separating.Whole
    if tail.startswith("fe."):
        parts = tail.split(".")
        if len(parts) >= 3:
            return parts[1], _normalise_fe_name(parts[2])
    if "." in tail:
        frame, fe = tail.split(".", 1)
        return frame, _normalise_fe_name(fe)
    return None, _normalise_fe_name(tail) if tail else None


def _normalise_fe_name(fe: str) -> str:
    fe = fe.removeprefix("has_").replace("+", " ")
    # FrameBase often lowercases FE names in property names; FST usually has title-case FEs.
    if fe.islower():
        return "_".join(part.capitalize() for part in fe.split("_"))
    return fe


def _is_fe_resource(resource: str) -> bool:
    tail = _tail(resource)
    return "/fe/" in resource or tail.startswith("fe.") or ".has_" in tail


def _is_dbp_resource(resource: str) -> bool:
    tail = _tail(resource)
    return "/dbp/" in resource or tail.startswith("dbp.") or ".is" in tail or ".has" in tail


def _parse_construct(chunk: str, prefixes: dict[str, str], file_name: str, i: int, labels: dict[str, str]) -> FrameBaseRule | None:
    dbp_match = None
    for match in DBP_RE.finditer(chunk):
        subj_var, predicate_token, obj_var = match.groups()
        pred = _resolve_token(predicate_token, prefixes)
        if _is_dbp_resource(pred):
            dbp_match = (subj_var, pred, obj_var)
            break
    if dbp_match is None:
        return None

    type_match = TYPE_RE.search(chunk)
    if not type_match:
        return None
    frame_res = _resolve_token(type_match.group(1), prefixes)
    frame_name = _frame_name_from_resource(frame_res)

    fe_by_var: dict[str, str] = {}
    for token, var in FE_RE.findall(chunk):
        res = _resolve_token(token, prefixes)
        if not _is_fe_resource(res):
            continue
        fe_by_var[var] = res

    subj_var, dbp_res, obj_var = dbp_match
    sub_fe = fe_by_var.get(subj_var)
    obj_fe = fe_by_var.get(obj_var)
    if sub_fe is None or obj_fe is None:
        return None

    sub_frame, sub_fe_name = _fe_parts_from_resource(sub_fe)
    obj_frame, obj_fe_name = _fe_parts_from_resource(obj_fe)
    # If type parsing failed on a microframe, recover frame from the FE resources.
    if frame_name is None:
        frame_name = sub_frame or obj_frame

    return FrameBaseRule(
        rule_id=f"{Path(file_name).stem}_{i}",
        source_format="sparql",
        source_file=file_name,
        frame_iri=frame_res,
        frame_name=frame_name,
        subject_fe_iri=sub_fe,
        object_fe_iri=obj_fe,
        subject_fe_name=sub_fe_name,
        object_fe_name=obj_fe_name,
        dbp_predicate_iri=dbp_res,
        dbp_predicate_name=_tail(dbp_res).split(".", 1)[-1] if "." in _tail(dbp_res) else _tail(dbp_res),
        dbp_iri=dbp_res,
        dbp_label=label_for_dbp(dbp_res, labels),
        raw_construct_node=f"{Path(file_name).stem}_{i}",
        parse_status="parsed",
        raw_rule=chunk[:4000],
    )


def parse_dered_rules(path: Path | None, dbp_labels: dict[str, str] | None = None) -> list[FrameBaseRule]:
    """Parse FrameBase SPARQL ReDer rules into a compact binary-rule index.

    The official FrameBase bundle contains many SPARQL CONSTRUCT queries. This parser extracts the
    practical binary shape needed for FST output conversion:

        ?f rdf:type/a FRAME .
        ?f FE_subject ?subject .
        ?f FE_object  ?object .
        CONSTRUCT { ?subject DBP ?object . }

    It accepts both modern path IRIs and older prefixed FrameBase names. Rules that do not expose a
    clear subject FE, object FE and DBP are skipped rather than guessed.
    """
    if path is None:
        return []
    labels = dbp_labels or {}
    rules: list[FrameBaseRule] = []
    for file_name, text in _read_rule_files(path):
        prefixes = _prefixes_from_text(text)
        for i, chunk in enumerate(_split_sparql_constructs(text)):
            rule = _parse_construct(chunk, prefixes, file_name, i, labels)
            if rule is not None:
                rules.append(rule)
    return rules
