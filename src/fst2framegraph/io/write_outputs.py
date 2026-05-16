from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def write_csv(df: pd.DataFrame, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    df.to_csv(path, index=False)
    return path


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
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


def write_json(data: dict[str, Any], out_dir: Path, name: str) -> Path:
    path = out_dir / name
    path.write_text(json.dumps(json_safe(data), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_jsonl(records: list[dict[str, Any]], out_dir: Path, name: str) -> Path:
    path = out_dir / name
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(json_safe(rec), ensure_ascii=False) + "\n")
    return path
