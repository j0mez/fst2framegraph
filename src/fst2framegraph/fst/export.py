from __future__ import annotations

import csv
import hashlib
import inspect
import json
import re
import sqlite3
import subprocess
import sys
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


TOKEN_RE = re.compile(r"[A-Za-z0-9_\u2019'\-/]+")

SENTENCE_FIELDS = ["sentence_id", "doc_id", "sentence_index", "row_index", "sentence"]
FRAME_INSTANCE_FIELDS = [
    "frame_instance_id",
    "sentence_id",
    "doc_id",
    "sentence_index",
    "row_index",
    "sentence",
    "frame_index",
    "frame_name",
    "target_text",
    "target_start",
    "target_end",
]
FRAME_ELEMENT_FIELDS = [
    "frame_instance_id",
    "sentence_id",
    "doc_id",
    "sentence_index",
    "row_index",
    "sentence",
    "frame_index",
    "frame_name",
    "target_text",
    "target_start",
    "target_end",
    "element_index",
    "element_name",
    "element_filler",
    "filler_start",
    "filler_end",
    "span_status",
    "filler_span_candidates_json",
]
ERROR_FIELDS = [
    "sentence_id",
    "doc_id",
    "sentence_index",
    "row_index",
    "sentence",
    "status",
    "error",
    "error_message",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _plain(obj: Any, depth: int = 0, max_depth: int = 8) -> Any:
    if depth > max_depth:
        return repr(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if is_dataclass(obj):
        return {k: _plain(v, depth + 1, max_depth) for k, v in asdict(obj).items()}
    if isinstance(obj, Mapping):
        return {str(k): _plain(v, depth + 1, max_depth) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v, depth + 1, max_depth) for v in obj]
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        return {str(k): _plain(v, depth + 1, max_depth) for k, v in d.items()}
    return repr(obj)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _read_data(data: Any, sentence_col: str) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()

    if isinstance(data, (str, Path)):
        path = Path(data)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".jsonl", ".ndjson"}:
            return pd.read_json(path, lines=True)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        raise ValueError(f"Unsupported input file type: {path}")

    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        if len(data) == 0:
            return pd.DataFrame(columns=[sentence_col])
        first = data[0]
        if isinstance(first, str):
            return pd.DataFrame({sentence_col: list(data)})
        if isinstance(first, Mapping):
            return pd.DataFrame(list(data))

    raise TypeError(
        "data must be a DataFrame, CSV/JSONL/Excel path, list of strings, or list of dicts"
    )


def _stable_sentence_id(sentence: str, row_index: int) -> str:
    raw = f"{row_index}::{sentence}".encode("utf-8", errors="replace")
    return hashlib.sha1(raw).hexdigest()


def _stable_frame_instance_id(
    sentence_id: str,
    frame_index: int,
    frame_name: str | None,
    trigger_start: int | None,
) -> str:
    safe_frame = (frame_name or "UNKNOWN_FRAME").replace(" ", "_")
    safe_trigger = "NA" if trigger_start is None else str(trigger_start)
    return f"{sentence_id}::f{frame_index}::{safe_frame}::{safe_trigger}"


def _coerce_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return None


def infer_target_text(sentence: str, trigger_start: int | None) -> tuple[str | None, int | None]:
    if trigger_start is None or trigger_start < 0 or trigger_start >= len(sentence):
        return None, None
    match = TOKEN_RE.match(sentence[trigger_start:])
    if not match:
        return None, None
    target_text = match.group(0)
    return target_text, trigger_start + len(target_text)


def reconstruct_filler_span(
    sentence: str,
    filler_text: Any,
    *,
    allow_ambiguous_spans: bool = False,
) -> tuple[int | None, int | None, str, list[dict[str, int]]]:
    if filler_text is None:
        return None, None, "missing_text", []

    text = str(filler_text).strip()
    if not text:
        return None, None, "empty_text", []

    lower_sentence = sentence.lower()
    lower_text = text.lower()

    starts = []
    start = 0
    while True:
        idx = lower_sentence.find(lower_text, start)
        if idx == -1:
            break
        starts.append(idx)
        start = idx + 1

    candidates = [{"start": s, "end": s + len(text)} for s in starts]

    if len(starts) == 1:
        s = starts[0]
        return s, s + len(text), "exact_unique", candidates

    if len(starts) > 1:
        if allow_ambiguous_spans:
            s = starts[0]
            return s, s + len(text), "exact_ambiguous_first_used", candidates
        return None, None, "exact_ambiguous_not_used", candidates

    return None, None, "not_found", []


def _frame_elements_from_frame(
    *,
    frame: Any,
    sentence: str,
    allow_ambiguous_spans: bool,
) -> list[dict[str, Any]]:
    elements = []
    frame_elements = _get(frame, "frame_elements", []) or []
    for element_index, fe in enumerate(frame_elements):
        fe_name = _get(fe, "name")
        fe_text = _get(fe, "text")
        filler_start, filler_end, span_status, candidates = reconstruct_filler_span(
            sentence,
            fe_text,
            allow_ambiguous_spans=allow_ambiguous_spans,
        )
        elements.append(
            {
                "element_index": element_index,
                "element_name": fe_name,
                "element_filler": fe_text,
                "filler_start": filler_start,
                "filler_end": filler_end,
                "span_status": span_status,
                "filler_span_candidates": candidates,
            }
        )
    return elements


def _normalise_frames(
    *,
    result: Any,
    sentence: str,
    sentence_id: str,
    allow_ambiguous_spans: bool,
) -> list[dict[str, Any]]:
    normalised = []
    frames = _get(result, "frames", []) or []
    for frame_index, frame in enumerate(frames):
        frame_name = _get(frame, "name")
        trigger_start = _coerce_int(_get(frame, "trigger_location"))
        target_text, target_end = infer_target_text(sentence, trigger_start)
        normalised.append(
            {
                "frame_instance_id": _stable_frame_instance_id(
                    sentence_id, frame_index, frame_name, trigger_start
                ),
                "frame_index": frame_index,
                "frame_name": frame_name,
                "target_text": target_text,
                "target_start": trigger_start,
                "target_end": target_end,
                "frame_elements": _frame_elements_from_frame(
                    frame=frame,
                    sentence=sentence,
                    allow_ambiguous_spans=allow_ambiguous_spans,
                ),
            }
        )
    return normalised


def _make_success_record(
    *,
    result: Any,
    sentence: str,
    sentence_id: str,
    doc_id: str | None,
    row_index: int | None,
    metadata: Mapping[str, Any] | None,
    allow_ambiguous_spans: bool,
) -> dict[str, Any]:
    metadata = dict(metadata or {})
    doc_id = doc_id or sentence_id
    return {
        "sentence_id": sentence_id,
        "doc_id": doc_id,
        "row_index": row_index,
        "sentence_index": row_index,
        "sentence": sentence,
        "status": "completed",
        "metadata": _json_safe(metadata),
        "processed_at": _utc_now(),
        "frames": _json_safe(
            _normalise_frames(
                result=result,
                sentence=sentence,
                sentence_id=sentence_id,
                allow_ambiguous_spans=allow_ambiguous_spans,
            )
        ),
        "result": _json_safe(_plain(result)),
    }


def _make_error_record(
    *,
    sentence: str,
    sentence_id: str,
    error: Exception | str,
    doc_id: str | None,
    row_index: int | None,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metadata = dict(metadata or {})
    doc_id = doc_id or sentence_id
    error_text = repr(error)
    return {
        "sentence_id": sentence_id,
        "doc_id": doc_id,
        "row_index": row_index,
        "sentence_index": row_index,
        "sentence": sentence,
        "status": "error",
        "metadata": _json_safe(metadata),
        "processed_at": _utc_now(),
        "frames": [],
        "error": error_text,
        "error_message": error_text,
    }


def _metadata_columns(records: Sequence[dict[str, Any]]) -> list[str]:
    cols: list[str] = []
    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        for key in metadata:
            if key not in cols:
                cols.append(str(key))
    return cols


def _row_base(record: Mapping[str, Any]) -> dict[str, Any]:
    row_index = record.get("row_index", record.get("sentence_index"))
    return {
        "sentence_id": record.get("sentence_id"),
        "doc_id": record.get("doc_id") or record.get("sentence_id"),
        "sentence_index": row_index,
        "row_index": row_index,
        "sentence": record.get("sentence"),
    }


def _record_to_rows(record: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    base = {**_row_base(record), **dict(metadata)}
    rows = {
        "sentences": [base],
        "frame_instances": [],
        "frame_elements": [],
        "errors": [],
    }

    is_error = record.get("status") == "error" or (
        "error" in record and "result" not in record and "frames" not in record
    )
    if is_error:
        error = record.get("error") or record.get("error_message")
        rows["errors"].append(
            {
                **base,
                "status": "error",
                "error": error,
                "error_message": error,
            }
        )
        return rows

    frames = record.get("frames")
    if frames is None and "result" in record:
        frames = _normalise_frames(
            result=record.get("result"),
            sentence=str(record.get("sentence") or ""),
            sentence_id=str(record.get("sentence_id") or ""),
            allow_ambiguous_spans=False,
        )
    frames = frames or []

    for frame in frames:
        frame_row = {
            **base,
            "frame_instance_id": frame.get("frame_instance_id"),
            "frame_index": frame.get("frame_index"),
            "frame_name": frame.get("frame_name"),
            "target_text": frame.get("target_text"),
            "target_start": frame.get("target_start"),
            "target_end": frame.get("target_end"),
        }
        rows["frame_instances"].append(frame_row)

        for element in frame.get("frame_elements", []) or []:
            candidates = element.get("filler_span_candidates", [])
            rows["frame_elements"].append(
                {
                    **frame_row,
                    "element_index": element.get("element_index"),
                    "element_name": element.get("element_name"),
                    "element_filler": element.get("element_filler"),
                    "filler_start": element.get("filler_start"),
                    "filler_end": element.get("filler_end"),
                    "span_status": element.get("span_status"),
                    "filler_span_candidates_json": json.dumps(
                        _json_safe(candidates), ensure_ascii=False
                    ),
                }
            )

    return rows


def _write_dataframe(path: Path, rows: list[dict[str, Any]], base_fields: list[str]) -> None:
    extra_fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in base_fields and key not in extra_fields:
                extra_fields.append(key)
    df = pd.DataFrame(rows, columns=base_fields + extra_fields)
    df.to_csv(path, index=False)


def _append_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: _json_safe(row.get(k)) for k in fieldnames})


def _initialise_csv(path: Path, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=list(fieldnames)).writeheader()


def _read_jsonl_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    records_by_sentence: OrderedDict[str, dict[str, Any]] = OrderedDict()
    duplicates = 0
    if not path.exists():
        return [], 0
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL record at {path}:{line_number}: {exc}") from exc
            sentence_id = str(record.get("sentence_id") or f"__line_{line_number}")
            if sentence_id in records_by_sentence:
                duplicates += 1
            records_by_sentence[sentence_id] = record
    return list(records_by_sentence.values()), duplicates


def materialise_run(run_dir: str | Path, *, write_csv: bool = True) -> dict[str, Any]:
    """Rebuild convenience CSV/report outputs from authoritative JSONL run state."""
    run_dir = Path(run_dir)
    jsonl_path = run_dir / "fst_clean.jsonl"
    records, duplicate_sentence_ids = _read_jsonl_records(jsonl_path)

    sentence_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    element_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    for record in records:
        rows = _record_to_rows(record)
        sentence_rows.extend(rows["sentences"])
        frame_rows.extend(rows["frame_instances"])
        element_rows.extend(rows["frame_elements"])
        error_rows.extend(rows["errors"])

    if write_csv:
        _write_dataframe(run_dir / "sentences.csv", sentence_rows, SENTENCE_FIELDS)
        _write_dataframe(run_dir / "frame_instances.csv", frame_rows, FRAME_INSTANCE_FIELDS)
        _write_dataframe(run_dir / "frame_elements.csv", element_rows, FRAME_ELEMENT_FIELDS)
        _write_dataframe(run_dir / "frame_elements_long.csv", element_rows, FRAME_ELEMENT_FIELDS)
        _write_dataframe(run_dir / "errors.csv", error_rows, ERROR_FIELDS)

    span_counts = {}
    if element_rows:
        span_counts = {
            str(k): int(v)
            for k, v in pd.Series([r.get("span_status") for r in element_rows])
            .value_counts(dropna=False)
            .to_dict()
            .items()
        }

    status_counts = {
        str(k): int(v)
        for k, v in pd.Series([r.get("status", "completed") for r in records])
        .value_counts(dropna=False)
        .to_dict()
        .items()
    }
    report = {
        "sentences": int(len(sentence_rows)),
        "frame_instances": int(len(frame_rows)),
        "frame_elements": int(len(element_rows)),
        "errors": int(len(error_rows)),
        "jsonl_records": int(len(records)),
        "duplicate_sentence_ids": int(duplicate_sentence_ids),
        "status_counts": status_counts,
        "span_status_counts": span_counts,
        "source_of_truth": {
            "jsonl": str(jsonl_path),
            "progress": str(run_dir / "progress.sqlite"),
        },
        "outputs": {
            "sentences": str(run_dir / "sentences.csv"),
            "frame_instances": str(run_dir / "frame_instances.csv"),
            "frame_elements": str(run_dir / "frame_elements.csv"),
            "frame_elements_long": str(run_dir / "frame_elements_long.csv"),
            "errors": str(run_dir / "errors.csv"),
            "jsonl": str(jsonl_path),
        },
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "extraction_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    md = [
        "# FST extraction report",
        "",
        f"- Sentences: {report['sentences']}",
        f"- Frame instances: {report['frame_instances']}",
        f"- Frame elements: {report['frame_elements']}",
        f"- Errors: {report['errors']}",
        f"- JSONL records: {report['jsonl_records']}",
        "",
        "## Span status counts",
        "",
    ]
    for key, value in span_counts.items():
        md.append(f"- {key}: {value}")
    (run_dir / "extraction_report.md").write_text("\n".join(md), encoding="utf-8")
    return report


class _ProgressStore:
    def __init__(self, path: Path, *, reset: bool = False) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if reset and path.exists():
            path.unlink()
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=DELETE")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS progress (
                sentence_id TEXT PRIMARY KEY,
                row_index INTEGER,
                doc_id TEXT,
                status TEXT NOT NULL,
                processed_at TEXT NOT NULL,
                error_message TEXT
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_progress_status ON progress(status)"
        )
        self.conn.commit()

    def completed_sentence_ids(self, *, retry_errors: bool = False) -> set[str]:
        statuses = ("completed", "ok") if retry_errors else ("completed", "ok", "error")
        placeholders = ",".join("?" for _ in statuses)
        rows = self.conn.execute(
            f"SELECT sentence_id FROM progress WHERE status IN ({placeholders})",
            statuses,
        ).fetchall()
        return {str(row[0]) for row in rows}

    def mark(self, record: Mapping[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO progress (
                sentence_id, row_index, doc_id, status, processed_at, error_message
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("sentence_id"),
                record.get("row_index", record.get("sentence_index")),
                record.get("doc_id"),
                record.get("status") or "completed",
                record.get("processed_at") or _utc_now(),
                record.get("error_message") or record.get("error"),
            ),
        )

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()


class FSTGraphWriter:
    def __init__(
        self,
        out_dir: str | Path,
        *,
        allow_ambiguous_spans: bool = False,
        metadata_cols: Sequence[str] | None = None,
        resume: bool = False,
        checkpoint_every: int = 100,
        save_jsonl: bool = True,
        save_csv: bool = True,
        save_raw_results: bool = False,
        stream_csv: bool = True,
    ) -> None:
        if resume and not save_jsonl:
            raise ValueError("resume=True requires save_jsonl=True")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.allow_ambiguous_spans = allow_ambiguous_spans
        self.checkpoint_every = max(1, int(checkpoint_every or 1))
        self.save_jsonl = save_jsonl
        self.save_csv = save_csv
        self.save_raw_results = save_raw_results
        self.stream_csv = stream_csv and save_csv
        self._records_since_checkpoint = 0

        metadata_cols = list(metadata_cols or [])
        self.sentence_fields = SENTENCE_FIELDS + [c for c in metadata_cols if c not in SENTENCE_FIELDS]
        self.frame_fields = FRAME_INSTANCE_FIELDS + [
            c for c in metadata_cols if c not in FRAME_INSTANCE_FIELDS
        ]
        self.element_fields = FRAME_ELEMENT_FIELDS + [
            c for c in metadata_cols if c not in FRAME_ELEMENT_FIELDS
        ]
        self.error_fields = ERROR_FIELDS + [c for c in metadata_cols if c not in ERROR_FIELDS]

        self.progress = _ProgressStore(self.out_dir / "progress.sqlite", reset=not resume)
        self.jsonl_path = self.out_dir / "fst_clean.jsonl"
        self._jsonl = None
        if save_jsonl:
            mode = "a" if resume else "w"
            self._jsonl = self.jsonl_path.open(mode, encoding="utf-8", buffering=1)

        if save_csv:
            if not resume:
                self._reset_csv_outputs()
            else:
                self._ensure_csv_headers()

        self._write_manifest(resume=resume)

    def _write_manifest(self, *, resume: bool) -> None:
        manifest = {
            "created_or_updated_at": _utc_now(),
            "resume": bool(resume),
            "canonical_state": ["fst_clean.jsonl", "progress.sqlite"],
            "materialised_outputs": [
                "sentences.csv",
                "frame_instances.csv",
                "frame_elements.csv",
                "frame_elements_long.csv",
                "errors.csv",
                "extraction_report.json",
            ],
            "raw_results": (
                "Raw Python/FST objects are never saved; portable JSONL records are saved instead."
            ),
        }
        (self.out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _reset_csv_outputs(self) -> None:
        _initialise_csv(self.out_dir / "sentences.csv", self.sentence_fields)
        _initialise_csv(self.out_dir / "frame_instances.csv", self.frame_fields)
        _initialise_csv(self.out_dir / "frame_elements.csv", self.element_fields)
        _initialise_csv(self.out_dir / "frame_elements_long.csv", self.element_fields)
        _initialise_csv(self.out_dir / "errors.csv", self.error_fields)

    def _ensure_csv_headers(self) -> None:
        for name, fields in [
            ("sentences.csv", self.sentence_fields),
            ("frame_instances.csv", self.frame_fields),
            ("frame_elements.csv", self.element_fields),
            ("frame_elements_long.csv", self.element_fields),
            ("errors.csv", self.error_fields),
        ]:
            path = self.out_dir / name
            if not path.exists() or path.stat().st_size == 0:
                _initialise_csv(path, fields)

    def completed_sentence_ids(self, *, retry_errors: bool = False) -> set[str]:
        return self.progress.completed_sentence_ids(retry_errors=retry_errors)

    def add_result(
        self,
        *,
        result: Any,
        sentence: str,
        sentence_id: str,
        doc_id: str | None = None,
        sentence_index: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        record = _make_success_record(
            result=result,
            sentence=sentence,
            sentence_id=sentence_id,
            doc_id=doc_id,
            row_index=sentence_index,
            metadata=metadata,
            allow_ambiguous_spans=self.allow_ambiguous_spans,
        )
        self.add_record(record)

    def add_error(
        self,
        *,
        sentence: str,
        sentence_id: str,
        error: Exception | str,
        doc_id: str | None = None,
        sentence_index: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        record = _make_error_record(
            sentence=sentence,
            sentence_id=sentence_id,
            error=error,
            doc_id=doc_id,
            row_index=sentence_index,
            metadata=metadata,
        )
        self.add_record(record)

    def add_record(self, record: dict[str, Any]) -> None:
        if self._jsonl is not None:
            self._jsonl.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")

        if self.stream_csv:
            rows = _record_to_rows(record)
            _append_csv(self.out_dir / "sentences.csv", rows["sentences"], self.sentence_fields)
            _append_csv(
                self.out_dir / "frame_instances.csv", rows["frame_instances"], self.frame_fields
            )
            _append_csv(
                self.out_dir / "frame_elements.csv", rows["frame_elements"], self.element_fields
            )
            _append_csv(
                self.out_dir / "frame_elements_long.csv",
                rows["frame_elements"],
                self.element_fields,
            )
            _append_csv(self.out_dir / "errors.csv", rows["errors"], self.error_fields)

        self.progress.mark(record)
        self._records_since_checkpoint += 1
        if self._records_since_checkpoint >= self.checkpoint_every:
            self.checkpoint()

    def checkpoint(self) -> None:
        if self._jsonl is not None:
            self._jsonl.flush()
        self.progress.commit()
        self._records_since_checkpoint = 0

    def close(self) -> dict[str, Any]:
        self.checkpoint()
        if self._jsonl is not None:
            self._jsonl.close()
        self.progress.close()
        if self.save_jsonl:
            return materialise_run(self.out_dir, write_csv=self.save_csv)
        return {
            "sentences": 0,
            "frame_instances": 0,
            "frame_elements": 0,
            "errors": 0,
            "source_of_truth": {"progress": str(self.out_dir / "progress.sqlite")},
        }


def build_graph_from_clean(
    *,
    clean_dir: str | Path,
    graph_out_dir: str | Path,
    framebase_core: str | Path | None = None,
    dbp_labels: str | Path | None = None,
    framebase_index: str | Path | None = None,
    extra_args: Sequence[str] | None = None,
) -> None:
    clean_dir = Path(clean_dir)
    materialise_run(clean_dir)

    cmd = [
        "fst2framegraph",
        "build",
        "--input",
        str(clean_dir / "frame_elements_long.csv"),
        "--out",
        str(graph_out_dir),
        "--doc-col",
        "doc_id",
        "--sentence-col",
        "sentence",
        "--sentence-id-col",
        "sentence_id",
        "--frame-col",
        "frame_name",
        "--frame-index-col",
        "frame_index",
        "--target-col",
        "target_text",
        "--target-start-col",
        "target_start",
        "--target-end-col",
        "target_end",
        "--fe-col",
        "element_name",
        "--filler-col",
        "element_filler",
        "--filler-start-col",
        "filler_start",
        "--filler-end-col",
        "filler_end",
    ]

    if framebase_index:
        cmd.extend(["--framebase-index", str(framebase_index)])
    if framebase_core:
        cmd.extend(["--framebase-core", str(framebase_core)])
    if dbp_labels:
        cmd.extend(["--dbp-labels", str(dbp_labels)])
    if extra_args:
        cmd.extend(list(extra_args))

    subprocess.run(cmd, check=True)


def _resolve_device(device: str | None) -> str | None:
    if device is None or device != "auto":
        return device
    try:
        import torch
    except Exception:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    try:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _supports_keyword(callable_obj: Any, keyword: str) -> bool:
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.kind == param.VAR_KEYWORD:
            return True
        if param.name == keyword:
            return True
    return False


def _make_default_fst(device: str | None) -> Any:
    try:
        from frame_semantic_transformer import FrameSemanticTransformer
    except Exception as exc:
        raise ImportError(
            "frame-semantic-transformer is required when fst=None, "
            "or pass an existing FST object."
        ) from exc

    if device and _supports_keyword(FrameSemanticTransformer, "device"):
        return FrameSemanticTransformer(device=device)
    return FrameSemanticTransformer()


def _batch_method(fst: Any) -> Any | None:
    for name in (
        "detect_frames_batch",
        "detect_frame_batch",
        "detect_frames_bulk",
        "detect_frames_many",
        "detect_frames_for_sentences",
    ):
        method = getattr(fst, name, None)
        if callable(method):
            return method
    return None


def _detect_one(fst: Any, sentence: str) -> tuple[bool, Any]:
    try:
        return True, fst.detect_frames(sentence)
    except Exception as exc:
        return False, exc


def _detect_batch(fst: Any, sentences: list[str]) -> list[tuple[bool, Any]]:
    if not sentences:
        return []

    method = _batch_method(fst)
    if method is not None:
        try:
            results = list(method(sentences))
            if len(results) == len(sentences):
                return [(True, result) for result in results]
        except Exception:
            pass

    return [_detect_one(fst, sentence) for sentence in sentences]


def _row_payload(
    row: pd.Series,
    *,
    row_index: int,
    sentence_col: str,
    sentence_id_col: str | None,
    doc_col: str | None,
    metadata_cols: Sequence[str],
) -> dict[str, Any]:
    sentence = str(row[sentence_col])
    sentence_id = (
        str(row[sentence_id_col]) if sentence_id_col else _stable_sentence_id(sentence, row_index)
    )
    doc_id = str(row[doc_col]) if doc_col else sentence_id
    return {
        "sentence": sentence,
        "sentence_id": sentence_id,
        "doc_id": doc_id,
        "row_index": row_index,
        "metadata": {col: row[col] for col in metadata_cols},
    }


def _check_unique_sentence_ids(payloads: Sequence[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for payload in payloads:
        sentence_id = str(payload["sentence_id"])
        if sentence_id in seen and sentence_id not in duplicates:
            duplicates.append(sentence_id)
        seen.add(sentence_id)
    if duplicates:
        shown = ", ".join(duplicates[:5])
        more = "" if len(duplicates) <= 5 else f" (+{len(duplicates) - 5} more)"
        raise ValueError(
            "Duplicate sentence_id values are not supported because resume is keyed by "
            f"sentence_id: {shown}{more}"
        )


def encode_with_fst(
    *,
    fst: Any | None = None,
    data: Any,
    sentence_col: str = "sentence",
    sentence_id_col: str | None = None,
    doc_col: str | None = None,
    metadata_cols: Sequence[str] | None = None,
    out_dir: str | Path = "fst_clean",
    allow_ambiguous_spans: bool = False,
    build_graph: bool = False,
    graph_out_dir: str | Path | None = None,
    framebase_core: str | Path | None = None,
    dbp_labels: str | Path | None = None,
    framebase_index: str | Path | None = None,
    limit: int | None = None,
    progress_every: int = 100,
    resume: bool = True,
    checkpoint_every: int = 100,
    stream: bool = True,
    save_jsonl: bool = True,
    save_csv: bool = True,
    save_raw_results: bool = False,
    batch_size: int = 16,
    device: str | None = "auto",
    retry_errors: bool = False,
) -> dict[str, Any]:
    if resume and not save_jsonl:
        raise ValueError("resume=True requires save_jsonl=True")

    resolved_device = _resolve_device(device)
    if fst is None:
        fst = _make_default_fst(resolved_device)

    detect_frames = getattr(fst, "detect_frames", None)
    if not callable(detect_frames):
        raise TypeError("fst must provide a callable .detect_frames(sentence) method")

    df = _read_data(data, sentence_col=sentence_col)

    if sentence_col not in df.columns:
        raise ValueError(f"sentence_col not found in data: {sentence_col}")

    metadata_cols = list(metadata_cols or [])

    missing_meta = [c for c in metadata_cols if c not in df.columns]
    if missing_meta:
        raise ValueError(f"metadata_cols missing from data: {missing_meta}")

    if sentence_id_col and sentence_id_col not in df.columns:
        raise ValueError(f"sentence_id_col not found in data: {sentence_id_col}")

    if doc_col and doc_col not in df.columns:
        raise ValueError(f"doc_col not found in data: {doc_col}")

    n = len(df) if limit is None else min(limit, len(df))
    payloads = [
        _row_payload(
            row,
            row_index=row_index,
            sentence_col=sentence_col,
            sentence_id_col=sentence_id_col,
            doc_col=doc_col,
            metadata_cols=metadata_cols,
        )
        for row_index, (_, row) in enumerate(df.iloc[:n].iterrows())
    ]
    _check_unique_sentence_ids(payloads)

    writer = FSTGraphWriter(
        out_dir=out_dir,
        allow_ambiguous_spans=allow_ambiguous_spans,
        metadata_cols=metadata_cols,
        resume=resume,
        checkpoint_every=checkpoint_every,
        save_jsonl=save_jsonl,
        save_csv=save_csv,
        save_raw_results=save_raw_results,
        stream_csv=stream,
    )

    completed = writer.completed_sentence_ids(retry_errors=retry_errors) if resume else set()
    pending: list[dict[str, Any]] = []
    skipped = 0

    for payload in payloads:
        if payload["sentence_id"] in completed:
            skipped += 1
            continue
        pending.append(payload)

    processed = 0
    batch_size = max(1, int(batch_size or 1))
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        detections = _detect_batch(fst, [item["sentence"] for item in batch])
        for payload, (ok, value) in zip(batch, detections):
            if ok:
                writer.add_result(
                    result=value,
                    sentence=payload["sentence"],
                    sentence_id=payload["sentence_id"],
                    doc_id=payload["doc_id"],
                    sentence_index=payload["row_index"],
                    metadata=payload["metadata"],
                )
            else:
                writer.add_error(
                    sentence=payload["sentence"],
                    sentence_id=payload["sentence_id"],
                    doc_id=payload["doc_id"],
                    sentence_index=payload["row_index"],
                    metadata=payload["metadata"],
                    error=value,
                )

            processed += 1
            done = processed + skipped
            if progress_every and done % progress_every == 0:
                print(f"Processed {done}/{n}", file=sys.stderr)

    report = writer.close()
    report.update(
        {
            "device": resolved_device,
            "batch_size": batch_size,
            "resume": bool(resume),
            "skipped_existing": int(skipped),
            "processed_this_run": int(processed),
        }
    )

    if build_graph:
        if graph_out_dir is None:
            graph_out_dir = Path(out_dir).with_name(Path(out_dir).name + "_graph")

        build_graph_from_clean(
            clean_dir=out_dir,
            graph_out_dir=graph_out_dir,
            framebase_core=framebase_core,
            dbp_labels=dbp_labels,
            framebase_index=framebase_index,
        )
        report["graph_out_dir"] = str(graph_out_dir)

    return report
