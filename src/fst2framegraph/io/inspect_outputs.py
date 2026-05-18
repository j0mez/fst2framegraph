from __future__ import annotations

import json
import pickle
import re
import sqlite3
import shutil
from contextlib import contextmanager
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from fst2framegraph.fst.export import (
    FSTGraphWriter,
    _json_safe,
    _make_error_record,
    _make_success_record,
    _stable_frame_instance_id,
    _stable_sentence_id,
    _utc_now,
    infer_target_text,
    materialise_run,
)


GRAPH_READY_COLUMNS = [
    "sentence_id",
    "sentence",
    "frame_index",
    "frame_name",
    "target_text",
    "target_start",
    "target_end",
    "element_name",
    "element_filler",
    "filler_start",
    "filler_end",
]
FLAT_COLUMNS = ["frame_name", "element_name", "element_filler"]
PICKLE_RE = re.compile(r".*?(\d+)_to_(\d+).*?_raw_results\.p(?:ickle|kl)$")
EXPECTED_PICKLE_SHAPES = (
    "Expected one of: a list/tuple of FST result records; a mapping with 'records' or "
    "'results'; a legacy batch mapping with 'raw_results' plus optional 'sentences', "
    "'unique_chunk_ids', and 'errors'; or a single record with 'sentence' plus 'frames' "
    "or 'result'."
)


def _scan_files(path: Path, recursive: bool = True) -> list[Path]:
    if path.is_file():
        return [path]
    iterator = path.rglob("*") if recursive else path.glob("*")
    return sorted(p for p in iterator if p.is_file())


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL record at {path}:{line_number}: {exc}") from exc
                if isinstance(value, Mapping):
                    records.append(dict(value))
        return records
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [dict(v) for v in value if isinstance(v, Mapping)]
    if isinstance(value, Mapping):
        return [dict(value)]
    return []


def _pickle_ranges(paths: Iterable[Path]) -> list[dict[str, int]]:
    ranges = []
    for path in paths:
        match = PICKLE_RE.match(path.name)
        if not match:
            continue
        ranges.append((int(match.group(1)), int(match.group(2))))
    if len(ranges) < 2:
        return []
    ranges = sorted(ranges)
    missing = []
    previous_end = ranges[0][1]
    for start, end in ranges[1:]:
        expected_start = previous_end + 1
        if start > expected_start:
            missing.append({"expected_start": expected_start, "expected_end": start - 1})
        previous_end = end
    return missing


def _inspect_csv(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path, nrows=1000)
    columns = list(df.columns)
    missing = [col for col in GRAPH_READY_COLUMNS if col not in columns]
    row_count = sum(1 for _ in path.open("r", encoding="utf-8", errors="replace")) - 1

    if not missing:
        return {
            "detected_format": "graph_ready_csv",
            "status": "graph_ready",
            "graph_ready": True,
            "convertible": True,
            "flat_only": False,
            "missing_required_columns": [],
            "warnings": [],
            "counts": {"rows": max(row_count, 0)},
            "recommended_next_command": f"fst2framegraph build --input {path} --out graph",
        }

    has_flat = all(col in columns for col in FLAT_COLUMNS)
    status = "flat_only" if has_flat else "insufficient"
    warnings = []
    if has_flat:
        warnings.append(
            "This file can support flat frame/FE counts, but is not sufficient for reliable nested graph construction."
        )
    return {
        "detected_format": "flattened_csv",
        "status": status,
        "graph_ready": False,
        "convertible": False,
        "flat_only": has_flat,
        "missing_required_columns": missing,
        "warnings": warnings,
        "counts": {"rows": max(row_count, 0)},
        "recommended_next_command": "rerun FST using encode_with_fst(..., resume=True)",
    }


def _inspect_json(path: Path) -> dict[str, Any]:
    records = _read_json_records(path)
    convertible = any("sentence" in r and ("frames" in r or "result" in r) for r in records)
    return {
        "detected_format": "fst_jsonl" if path.suffix.lower() in {".jsonl", ".ndjson"} else "fst_json",
        "status": "convertible" if convertible else "insufficient",
        "graph_ready": False,
        "convertible": convertible,
        "flat_only": False,
        "missing_required_columns": [] if convertible else ["sentence", "frames"],
        "warnings": [],
        "counts": {"records": len(records)},
        "recommended_next_command": f"fst2framegraph convert --input {path} --out fst_clean",
    }


def _inspect_run_dir(path: Path, files: list[Path]) -> dict[str, Any] | None:
    jsonl = path / "fst_clean.jsonl"
    progress = path / "progress.sqlite"
    if not jsonl.exists() and not progress.exists():
        return None
    csv_path = path / "frame_elements_long.csv"
    graph_ready = False
    missing = GRAPH_READY_COLUMNS
    if csv_path.exists():
        csv_report = _inspect_csv(csv_path)
        graph_ready = bool(csv_report["graph_ready"])
        missing = list(csv_report["missing_required_columns"])
    status = "ready" if jsonl.exists() and progress.exists() else "incomplete"
    if status == "ready" and not csv_path.exists():
        status = "rebuildable"
    return {
        "detected_format": "v0.3_run_directory",
        "status": status,
        "graph_ready": graph_ready or (jsonl.exists() and progress.exists()),
        "convertible": jsonl.exists(),
        "flat_only": False,
        "missing_required_columns": [] if graph_ready else missing,
        "warnings": [] if jsonl.exists() else ["fst_clean.jsonl is missing."],
        "counts": {},
        "recommended_next_command": f"fst2framegraph materialise --run-dir {path}",
        "pickle_files": [str(p) for p in files if p.suffix.lower() in {".pkl", ".pickle"}],
    }


@contextmanager
def _legacy_nltk_framenet_pickle_compat() -> Iterable[None]:
    """Allow trusted legacy NLTK FrameNet objects to unpickle.

    Older NLTK FrameNet helper classes implement ``__getattr__`` by indexing into
    dict-like state. During unpickling, Python probes for special attributes such
    as ``__setstate__``; those classes raise ``KeyError`` instead of
    ``AttributeError``, which aborts unpickling. Patch only inside explicit
    trusted-pickle loading so normal code paths never load or mutate pickle
    behavior by default.
    """

    try:
        import nltk.corpus.reader.framenet as framenet  # type: ignore[import-untyped]
    except Exception:
        yield
        return

    originals: list[tuple[type, Any]] = []
    for class_name in ("AttrDict", "PrettyDict", "Future"):
        cls = getattr(framenet, class_name, None)
        original = getattr(cls, "__getattr__", None)
        if cls is None or original is None:
            continue
        originals.append((cls, original))

        def safe_getattr(self: Any, name: str, _original: Any = original) -> Any:
            if str(name).startswith("__"):
                raise AttributeError(name)
            try:
                return _original(self, name)
            except (AttributeError, KeyError, RecursionError) as exc:
                raise AttributeError(name) from exc

        cls.__getattr__ = safe_getattr

    try:
        yield
    finally:
        for cls, original in originals:
            cls.__getattr__ = original


def _safe_len(value: Any) -> int | None:
    try:
        return len(value)
    except Exception:
        return None


def _type_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _safe_getattr(value: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(value, name)
    except Exception:
        return default


def _pickle_payload_summary(value: Any, *, depth: int = 0) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": _type_name(value)}
    length = _safe_len(value)
    if length is not None:
        summary["length"] = length
    if isinstance(value, Mapping):
        keys = [str(key) for key in list(value.keys())[:20]]
        summary["keys"] = keys
        for key in (
            "records",
            "results",
            "raw_results",
            "sentences",
            "unique_chunk_ids",
            "errors",
            "sentence",
            "frames",
            "result",
        ):
            if key in value:
                item = value[key]
                summary[f"{key}_type"] = _type_name(item)
                item_len = _safe_len(item)
                if item_len is not None:
                    summary[f"{key}_length"] = item_len
        return summary
    if isinstance(value, list | tuple):
        if value and depth < 1:
            summary["first_item"] = _pickle_payload_summary(value[0], depth=depth + 1)
        return summary
    attrs = list(getattr(value, "__dict__", {}).keys())[:20]
    if attrs:
        summary["attrs"] = attrs
    for attr in ("sentence", "sentence_id", "doc_id", "frames", "result"):
        attr_value = _safe_getattr(value, attr, None)
        if attr_value is not None:
            summary[f"{attr}_type"] = _type_name(attr_value)
            attr_len = _safe_len(attr_value)
            if attr_len is not None:
                summary[f"{attr}_length"] = attr_len
    return summary


def _unsupported_pickle_payload_error(payload: Any) -> ValueError:
    return ValueError(
        "Unsupported trusted pickle payload. "
        f"Detected structure: {json.dumps(_pickle_payload_summary(payload), ensure_ascii=False)}. "
        + EXPECTED_PICKLE_SHAPES
    )


def _inspect_pickles(
    paths: list[Path],
    *,
    is_directory: bool = False,
    allow_pickle: bool = False,
) -> dict[str, Any]:
    base = {
        "detected_format": "pickle_folder" if is_directory or len(paths) > 1 else "pickle_file",
        "status": "unsafe_without_pickle_permission",
        "graph_ready": False,
        "convertible": False,
        "flat_only": False,
        "unsafe_without_pickle_permission": True,
        "missing_required_columns": [],
        "warnings": [
            "Python pickles can execute code. Only convert pickles from trusted sources with --allow-pickle."
        ],
        "counts": {"pickle_files": len(paths)},
        "pickle_files": [str(p) for p in paths],
        "missing_pickle_ranges": _pickle_ranges(paths),
        "recommended_next_command": "fst2framegraph prepare --input PATH --out fst_clean --allow-pickle",
    }
    if not allow_pickle:
        return base

    payloads = []
    record_count = 0
    warnings = []
    for file in paths:
        with _legacy_nltk_framenet_pickle_compat():
            with file.open("rb") as f:
                payload = pickle.load(f)
            summary = _pickle_payload_summary(payload)
            try:
                records = _records_from_pickle_payload(payload, record_count)
            except ValueError as exc:
                warnings.append(f"{file}: {exc}")
                records = []
            record_count += len(records)
            payloads.append({"path": str(file), "structure": summary, "records": len(records)})

    convertible = not warnings
    base.update(
        {
            "status": "convertible" if convertible else "unsupported_pickle_structure",
            "convertible": convertible,
            "unsafe_without_pickle_permission": False,
            "warnings": warnings,
            "counts": {"pickle_files": len(paths), "records": record_count},
            "pickle_payloads": payloads,
            "recommended_next_command": (
                "fst2framegraph prepare --input PATH --out fst_clean --allow-pickle"
                if convertible
                else EXPECTED_PICKLE_SHAPES
            ),
        }
    )
    return base


def inspect_fst_outputs(
    path: str | Path,
    *,
    recursive: bool = True,
    allow_pickle: bool = False,
) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")

    files = _scan_files(path, recursive=recursive)
    pickle_files = [p for p in files if p.suffix.lower() in {".pkl", ".pickle"}]

    if path.is_dir():
        run_report = _inspect_run_dir(path, files)
        if run_report is not None:
            run_report.update({"files_scanned": len(files)})
            return run_report
        if pickle_files:
            report = _inspect_pickles(
                pickle_files,
                is_directory=True,
                allow_pickle=allow_pickle,
            )
            report["files_scanned"] = len(files)
            return report
        csvs = [p for p in files if p.suffix.lower() == ".csv"]
        if csvs:
            report = _inspect_csv(csvs[0])
            report["files_scanned"] = len(files)
            report["input_file"] = str(csvs[0])
            return report
        jsons = [p for p in files if p.suffix.lower() in {".json", ".jsonl", ".ndjson"}]
        if jsons:
            report = _inspect_json(jsons[0])
            report["files_scanned"] = len(files)
            report["input_file"] = str(jsons[0])
            return report
        return {
            "detected_format": "unknown_directory",
            "status": "insufficient",
            "graph_ready": False,
            "convertible": False,
            "flat_only": False,
            "missing_required_columns": GRAPH_READY_COLUMNS,
            "warnings": ["No supported FST output files were found."],
            "counts": {},
            "files_scanned": len(files),
            "recommended_next_command": "rerun FST using encode_with_fst(..., resume=True)",
        }

    suffix = path.suffix.lower()
    if suffix == ".csv":
        report = _inspect_csv(path)
    elif suffix in {".json", ".jsonl", ".ndjson"}:
        report = _inspect_json(path)
    elif suffix in {".pkl", ".pickle"}:
        report = _inspect_pickles([path], allow_pickle=allow_pickle)
    else:
        report = {
            "detected_format": "unknown_file",
            "status": "insufficient",
            "graph_ready": False,
            "convertible": False,
            "flat_only": False,
            "missing_required_columns": GRAPH_READY_COLUMNS,
            "warnings": [f"Unsupported file type: {path.suffix}"],
            "counts": {},
            "recommended_next_command": "rerun FST using encode_with_fst(..., resume=True)",
        }
    report["files_scanned"] = 1
    report["input_file"] = str(path)
    report.setdefault("pickle_files", [str(p) for p in pickle_files])
    return report


def _frame_instance_id_from_frame(sentence_id: str, frame: Mapping[str, Any]) -> str:
    existing = frame.get("frame_instance_id")
    if existing:
        return str(existing)
    return _stable_frame_instance_id(
        sentence_id,
        int(frame.get("frame_index") or 0),
        str(frame.get("frame_name") or "UNKNOWN_FRAME"),
        _coerce_int(frame.get("target_start")),
    )


def _coerce_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return int(float(value))
    except Exception:
        return None


def _canonical_record_from_frames(record: Mapping[str, Any], row_index: int) -> dict[str, Any]:
    sentence = str(record.get("sentence") or "")
    sentence_id = str(record.get("sentence_id") or _stable_sentence_id(sentence, row_index))
    doc_id = str(record.get("doc_id") or record.get("document_id") or sentence_id)
    frames = []
    for frame_index, frame_in in enumerate(record.get("frames", []) or []):
        frame = dict(frame_in)
        frame.setdefault("frame_index", frame_index)
        frame.setdefault("frame_name", frame.get("name"))
        frame.setdefault("target_start", frame.get("trigger_location"))
        if frame.get("target_text") is None:
            target_text, target_end = infer_target_text(sentence, _coerce_int(frame.get("target_start")))
            frame["target_text"] = target_text
            frame.setdefault("target_end", target_end)
        frame["frame_instance_id"] = _frame_instance_id_from_frame(sentence_id, frame)
        elements = []
        for element_index, fe_in in enumerate(frame.get("frame_elements", []) or []):
            fe = dict(fe_in)
            elements.append(
                {
                    "element_index": fe.get("element_index", element_index),
                    "element_name": fe.get("element_name") or fe.get("name"),
                    "element_filler": fe.get("element_filler") or fe.get("text"),
                    "filler_start": fe.get("filler_start"),
                    "filler_end": fe.get("filler_end"),
                    "span_status": fe.get("span_status"),
                    "filler_span_candidates": fe.get("filler_span_candidates", []),
                }
            )
        frame["frame_elements"] = elements
        frames.append(frame)
    return {
        "sentence_id": sentence_id,
        "doc_id": doc_id,
        "row_index": row_index,
        "sentence_index": row_index,
        "sentence": sentence,
        "status": record.get("status", "completed"),
        "metadata": _json_safe(record.get("metadata", {})),
        "processed_at": record.get("processed_at") or _utc_now(),
        "frames": _json_safe(frames),
        "result": _json_safe(record.get("result", {"frames": frames})),
    }


def _records_from_csv(path: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    missing = [col for col in GRAPH_READY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "CSV is not graph-ready. Missing required columns: "
            + ", ".join(missing)
            + ". Rerun FST with encode_with_fst(..., resume=True) if spans are unavailable."
        )
    if "doc_id" not in df.columns:
        df["doc_id"] = df["sentence_id"]

    records = []
    for row_index, (sentence_id, sent_df) in enumerate(df.groupby("sentence_id", sort=False)):
        first = sent_df.iloc[0]
        frames = []
        frame_group_cols = ["frame_index", "frame_name", "target_start", "target_end"]
        for _, frame_df in sent_df.groupby(frame_group_cols, sort=False, dropna=False):
            fr = frame_df.iloc[0]
            frame = {
                "frame_index": _coerce_int(fr["frame_index"]),
                "frame_name": fr["frame_name"],
                "target_text": fr["target_text"],
                "target_start": _coerce_int(fr["target_start"]),
                "target_end": _coerce_int(fr["target_end"]),
                "frame_elements": [],
            }
            frame["frame_instance_id"] = _frame_instance_id_from_frame(str(sentence_id), frame)
            for element_index, (_, fe) in enumerate(frame_df.iterrows()):
                frame["frame_elements"].append(
                    {
                        "element_index": element_index,
                        "element_name": fe["element_name"],
                        "element_filler": fe["element_filler"],
                        "filler_start": _coerce_int(fe["filler_start"]),
                        "filler_end": _coerce_int(fe["filler_end"]),
                        "span_status": fe.get("span_status", ""),
                        "filler_span_candidates": [],
                    }
                )
            frames.append(frame)
        records.append(
            _canonical_record_from_frames(
                {
                    "sentence_id": sentence_id,
                    "doc_id": first["doc_id"],
                    "sentence": first["sentence"],
                    "frames": frames,
                },
                row_index,
            )
        )
    return records


def _iter_pickle_payloads(path: Path, recursive: bool) -> Iterable[Any]:
    files = [path] if path.is_file() else [
        p for p in _scan_files(path, recursive=recursive) if p.suffix.lower() in {".pkl", ".pickle"}
    ]
    for file in files:
        with _legacy_nltk_framenet_pickle_compat():
            with file.open("rb") as f:
                payload = pickle.load(f)
            yield payload


def _mapping_looks_like_record(payload: Mapping[str, Any]) -> bool:
    return "result" in payload or "frames" in payload or (
        "sentence" in payload and ("frames" in payload or "result" in payload)
    )


def _object_looks_like_result(payload: Any) -> bool:
    return _safe_getattr(payload, "frames", None) is not None


def _records_from_legacy_raw_results_batch(
    payload: Mapping[str, Any],
    start_index: int,
) -> list[dict[str, Any]]:
    raw_results = payload.get("raw_results")
    if not isinstance(raw_results, list | tuple):
        raise _unsupported_pickle_payload_error(payload)

    sentences = list(payload.get("sentences") or [])
    sentence_ids = list(payload.get("unique_chunk_ids") or [])
    errors = list(payload.get("errors") or [])
    first_global = payload.get("first_global_unique_row")
    batch_key = payload.get("batch_key")
    records = []
    for offset, result in enumerate(raw_results):
        row_index = start_index + offset
        sentence = str(
            sentences[offset]
            if offset < len(sentences)
            else _safe_getattr(result, "sentence", "")
        )
        sentence_id = str(
            sentence_ids[offset]
            if offset < len(sentence_ids)
            else _stable_sentence_id(sentence, row_index)
        )
        doc_id = sentence_id
        metadata = {
            "legacy_pickle_batch_key": batch_key,
            "legacy_pickle_row_offset": offset,
        }
        if first_global is not None:
            try:
                metadata["legacy_pickle_global_unique_row"] = int(first_global) + offset
            except Exception:
                metadata["legacy_pickle_global_unique_row"] = first_global
        error = errors[offset] if offset < len(errors) else None
        if error is not None or result is None:
            records.append(
                _make_error_record(
                    sentence=sentence,
                    sentence_id=sentence_id,
                    error=error or "Missing raw FST result in legacy pickle batch.",
                    doc_id=doc_id,
                    row_index=row_index,
                    metadata=metadata,
                )
            )
            continue
        records.append(
            _make_success_record(
                result=result,
                sentence=sentence or str(_safe_getattr(result, "sentence", "")),
                sentence_id=sentence_id,
                doc_id=doc_id,
                row_index=row_index,
                metadata=metadata,
                allow_ambiguous_spans=False,
            )
        )
    return records


def _records_from_pickle_payload(payload: Any, start_index: int = 0) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping):
        if "records" in payload:
            return _records_from_pickle_payload(payload["records"], start_index)
        if "results" in payload:
            return _records_from_pickle_payload(payload["results"], start_index)
        if "raw_results" in payload:
            return _records_from_legacy_raw_results_batch(payload, start_index)
        if not _mapping_looks_like_record(payload):
            raise _unsupported_pickle_payload_error(payload)
        payloads = [payload]
    elif isinstance(payload, list | tuple):
        payloads = list(payload)
    else:
        if not _object_looks_like_result(payload):
            raise _unsupported_pickle_payload_error(payload)
        payloads = [payload]

    records = []
    for offset, item in enumerate(payloads):
        row_index = start_index + offset
        if isinstance(item, Mapping):
            if not _mapping_looks_like_record(item):
                raise _unsupported_pickle_payload_error(item)
            sentence = item.get("sentence") or getattr(item.get("result"), "sentence", "")
            sentence_id = item.get("sentence_id") or _stable_sentence_id(str(sentence), row_index)
            doc_id = item.get("doc_id") or item.get("document_id") or sentence_id
            result = item.get("result")
            if result is not None:
                records.append(
                    _make_success_record(
                        result=result,
                        sentence=str(sentence),
                        sentence_id=str(sentence_id),
                        doc_id=str(doc_id),
                        row_index=row_index,
                        metadata={},
                        allow_ambiguous_spans=False,
                    )
                )
            else:
                records.append(_canonical_record_from_frames(item, row_index))
        else:
            if not _object_looks_like_result(item):
                raise _unsupported_pickle_payload_error(item)
            sentence = str(_safe_getattr(item, "sentence", ""))
            sentence_id = str(
                _safe_getattr(item, "sentence_id", _stable_sentence_id(sentence, row_index))
            )
            doc_id = str(_safe_getattr(item, "doc_id", sentence_id))
            records.append(
                _make_success_record(
                    result=item,
                    sentence=sentence,
                    sentence_id=sentence_id,
                    doc_id=doc_id,
                    row_index=row_index,
                    metadata={},
                    allow_ambiguous_spans=False,
                )
            )
    return records


def _write_records(records: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    writer = FSTGraphWriter(out_dir=out_dir, resume=False, checkpoint_every=100)
    try:
        for record in records:
            writer.add_record(record)
        return writer.close()
    finally:
        try:
            writer.progress.close()
        except Exception:
            pass


def convert_fst_outputs(
    path: str | Path,
    out_dir: str | Path,
    *,
    allow_pickle: bool = False,
    recursive: bool = True,
) -> dict[str, Any]:
    path = Path(path)
    out_dir = Path(out_dir)
    report = inspect_fst_outputs(path, recursive=recursive)

    if report["detected_format"] == "v0.3_run_directory":
        if path.resolve() != out_dir.resolve():
            out_dir.mkdir(parents=True, exist_ok=True)
            for name in ["fst_clean.jsonl", "progress.sqlite"]:
                src = path / name
                if src.exists():
                    shutil.copy2(src, out_dir / name)
        return materialise_run(out_dir if path.resolve() != out_dir.resolve() else path)

    detected = report["detected_format"]
    if detected in {"pickle_file", "pickle_folder"}:
        if not allow_pickle:
            raise ValueError(
                "Python pickles can execute code. Re-run with allow_pickle=True or --allow-pickle "
                "only for trusted files."
            )
        records: list[dict[str, Any]] = []
        for payload in _iter_pickle_payloads(path, recursive):
            records.extend(_records_from_pickle_payload(payload, len(records)))
        return _write_records(records, out_dir)

    input_file = Path(report.get("input_file") or path)
    if detected == "graph_ready_csv":
        return _write_records(_records_from_csv(input_file), out_dir)
    if detected in {"fst_json", "fst_jsonl"}:
        records = [
            _canonical_record_from_frames(record, row_index)
            for row_index, record in enumerate(_read_json_records(input_file))
        ]
        return _write_records(records, out_dir)

    raise ValueError(
        f"Input is not convertible ({report['status']}). Missing: "
        + ", ".join(report.get("missing_required_columns", []))
    )


def doctor_run(
    *,
    run_dir: str | Path | None = None,
    framebase_index: str | Path | None = None,
) -> dict[str, Any]:
    checks = []
    next_commands: list[str] = []
    ok = True
    if run_dir is not None:
        report = inspect_fst_outputs(run_dir)
        run_next = _doctor_run_next_commands(
            Path(run_dir),
            report,
            framebase_index=Path(framebase_index) if framebase_index else None,
        )
        checks.append({"target": str(run_dir), **report, "next_commands": run_next})
        next_commands.extend(run_next)
        ok = ok and report["status"] in {"ready", "rebuildable", "graph_ready"}
    if framebase_index is not None:
        index_path = Path(framebase_index)
        index_report: dict[str, Any] = {
            "target": str(index_path),
            "detected_format": "framebase_index",
            "status": "missing",
            "counts": {},
            "warnings": [],
        }
        if index_path.exists():
            with sqlite3.connect(index_path) as conn:
                counts = {
                    table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                    for table in ["frames", "frame_elements", "dbp_labels"]
                }
            index_report["status"] = "ready" if counts["frames"] and counts["frame_elements"] else "incomplete"
            index_report["counts"] = counts
            if index_report["status"] != "ready":
                index_report["next_commands"] = [_framebase_index_next_command(index_path)]
        else:
            index_report["warnings"].append("FrameBase index does not exist.")
            index_report["next_commands"] = [_framebase_index_next_command(index_path)]
        checks.append(index_report)
        next_commands.extend(index_report.get("next_commands", []))
        ok = ok and index_report["status"] == "ready"
    if not checks:
        raise ValueError("doctor requires --run-dir, --framebase-index, or both.")
    return {"ok": ok, "checks": checks, "next_commands": list(dict.fromkeys(next_commands))}


def _doctor_run_next_commands(
    path: Path,
    report: Mapping[str, Any],
    *,
    framebase_index: Path | None = None,
) -> list[str]:
    status = report.get("status")
    detected = report.get("detected_format")
    index = str(framebase_index) if framebase_index else "PATH/framebase_index.sqlite"
    if status == "rebuildable":
        return [f"fst2framegraph materialise --run-dir {path}"]
    if detected == "v0.3_run_directory" and report.get("graph_ready"):
        return [
            "fst2framegraph build "
            f"--input {path} "
            "--out graph_output "
            f"--framebase-index {index}"
        ]
    if report.get("flat_only"):
        return [
            "fst2framegraph detect "
            "--input sentences.csv "
            "--text-col sentence "
            "--id-col sentence_id "
            "--doc-col doc_id "
            "--out fst_clean "
            "--resume"
        ]
    if status == "unsafe_without_pickle_permission":
        return [
            "fst2framegraph prepare "
            f"--input {path} "
            "--out fst_clean "
            "--allow-pickle"
        ]
    return [f"fst2framegraph inspect --input {path}"]


def _framebase_index_next_command(index_path: Path) -> str:
    framebase_dir = index_path.parent
    return (
        "fst2framegraph build-framebase-index "
        f"--framebase-dir {framebase_dir} "
        f"--index {index_path}"
    )
