from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from fst2framegraph.io.inspect_outputs import (
    _iter_pickle_payloads,
    _records_from_pickle_payload,
    materialise_run,
)


def _clean(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def _record_to_document(record: Mapping[str, Any]) -> dict:
    doc_id = _clean(record.get("doc_id") or record.get("document_id") or record.get("sentence_id"))
    sentence = _clean(record.get("text") or record.get("sentence"))
    metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    frames = []
    for frame in record.get("frames") or []:
        if not isinstance(frame, Mapping):
            continue
        elements = [
            {
                "role": _clean(element.get("role") or element.get("element_name") or element.get("name")),
                "text": _clean(element.get("text") or element.get("element_filler")),
            }
            for element in frame.get("frame_elements") or []
            if isinstance(element, Mapping)
        ]
        frames.append(
            {
                "frame_type": _clean(frame.get("frame_type") or frame.get("frame_name") or frame.get("name")),
                "trigger": _clean(frame.get("trigger") or frame.get("target_text")),
                "sent_idx": int(frame.get("sent_idx") or frame.get("sentence_index") or 0),
                "frame_elements": elements,
            }
        )
    return {"doc_id": doc_id, "text": sentence, "metadata": dict(metadata), "frames": frames}


def _records_to_documents(records: list[Mapping[str, Any]]) -> list[dict]:
    """Merge sentence-shaped records into document-shaped records.

    Canonical run JSONL is often one sentence per row. Returning each row as a
    separate document would collide on ``frame:{doc_id}:{i}``, so this adapter
    restores document-level grouping before graph construction.
    """
    documents: OrderedDict[str, dict] = OrderedDict()
    sentence_maps: dict[str, OrderedDict[str, int]] = {}
    text_parts: dict[str, list[str]] = {}

    for index, record in enumerate(records):
        doc = _record_to_document(record)
        doc_id = doc["doc_id"] or f"doc{index}"
        if doc_id not in documents:
            documents[doc_id] = {
                "doc_id": doc_id,
                "text": "",
                "metadata": doc["metadata"],
                "frames": [],
            }
            sentence_maps[doc_id] = OrderedDict()
            text_parts[doc_id] = []
        documents[doc_id]["metadata"].update(doc["metadata"])

        is_sentence_record = "sentence_id" in record or ("sentence" in record and "text" not in record)
        if is_sentence_record:
            sentence_key = _clean(record.get("sentence_id") or record.get("sentence") or index)
            if sentence_key not in sentence_maps[doc_id]:
                sentence_maps[doc_id][sentence_key] = len(sentence_maps[doc_id])
                if doc["text"]:
                    text_parts[doc_id].append(doc["text"])
            sent_idx = sentence_maps[doc_id][sentence_key]
        else:
            if doc["text"] and not documents[doc_id]["text"]:
                text_parts[doc_id] = [doc["text"]]
            sent_idx = None

        for frame in doc["frames"]:
            frame_copy = dict(frame)
            if sent_idx is not None:
                frame_copy["sent_idx"] = sent_idx
            documents[doc_id]["frames"].append(frame_copy)

    for doc_id, doc in documents.items():
        doc["text"] = "\n".join(text_parts[doc_id])
    return list(documents.values())


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        records = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    value = json.loads(line)
                    if isinstance(value, Mapping):
                        records.append(dict(value))
        return records
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    if isinstance(value, Mapping):
        return [dict(value)]
    return []


def from_frame_elements_long_csv(path: str | Path) -> list[dict]:
    """Convert the canonical long FE CSV into the public document schema."""
    df = pd.read_csv(path)
    if "doc_id" not in df.columns:
        df["doc_id"] = df.get("document_id", df.get("sentence_id", "doc"))
    required = {"doc_id", "sentence_id", "sentence", "frame_name", "element_name", "element_filler"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"frame_elements_long CSV is missing required columns: {', '.join(missing)}")

    documents: list[dict] = []
    for doc_id, doc_df in df.groupby("doc_id", sort=False):
        sentence_order = OrderedDict()
        for sentence_id, sent_df in doc_df.groupby("sentence_id", sort=False):
            sentence_order[str(sentence_id)] = _clean(sent_df.iloc[0].get("sentence"))
        sent_idx_by_id = {sentence_id: idx for idx, sentence_id in enumerate(sentence_order)}
        frames = []
        group_cols = [
            col
            for col in ["sentence_id", "frame_index", "frame_name", "target_text"]
            if col in doc_df.columns
        ]
        for _, frame_df in doc_df.groupby(group_cols, sort=False, dropna=False):
            first = frame_df.iloc[0]
            elements = []
            for _, row in frame_df.iterrows():
                role = _clean(row.get("element_name"))
                text = _clean(row.get("element_filler"))
                if role and text:
                    elements.append({"role": role, "text": text})
            frames.append(
                {
                    "frame_type": _clean(first.get("frame_name")),
                    "trigger": _clean(first.get("target_text")),
                    "sent_idx": sent_idx_by_id.get(str(first.get("sentence_id")), 0),
                    "frame_elements": elements,
                }
            )
        documents.append(
            {
                "doc_id": str(doc_id),
                "text": "\n".join(sentence_order.values()),
                "metadata": {},
                "frames": frames,
            }
        )
    return documents


def from_legacy_pickle(path: str | Path, *, allow_pickle: bool = False) -> list[dict]:
    """Convert trusted legacy pickle payloads into the public document schema.

    Pickle loading can execute code, so callers must opt in explicitly with
    ``allow_pickle=True``.
    """
    if not allow_pickle:
        raise ValueError("Pickle loading is unsafe. Re-run with allow_pickle=True for trusted files.")
    records: list[dict[str, Any]] = []
    for payload in _iter_pickle_payloads(Path(path), recursive=True):
        records.extend(_records_from_pickle_payload(payload, len(records)))
    return _records_to_documents(records)


def from_fst_output(path: str | Path, *, allow_pickle: bool = False) -> list[dict]:
    """Adapt common fst2framegraph/FST outputs into the public document schema."""
    path = Path(path)
    if path.is_dir():
        csv_path = path / "frame_elements_long.csv"
        if csv_path.exists():
            return from_frame_elements_long_csv(csv_path)
        if (path / "fst_clean.jsonl").exists():
            materialise_run(path)
            if csv_path.exists():
                return from_frame_elements_long_csv(csv_path)
        raise ValueError(f"No frame_elements_long.csv or materialisable fst_clean.jsonl found in {path}.")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return from_frame_elements_long_csv(path)
    if suffix in {".pkl", ".pickle"}:
        return from_legacy_pickle(path, allow_pickle=allow_pickle)
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return _records_to_documents(_read_json_records(path))
    raise ValueError(f"Unsupported FST output path: {path}")


__all__ = ["from_fst_output", "from_frame_elements_long_csv", "from_legacy_pickle"]
