from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_run_summary(*, out_dir: str | Path, payload: dict[str, Any]) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# fst2framegraph v2 pipeline summary",
        "",
        f"- run_id: {payload.get('run_id')}",
        f"- input_csv: {payload.get('input_csv')}",
        f"- run_dir: {payload.get('run_dir')}",
        f"- graph_out_dir: {payload.get('graph_out_dir')}",
        f"- analysis_out_dir: {payload.get('analysis_out_dir')}",
        "",
        "## Counts",
        "",
        f"- input_rows: {payload.get('input_rows')}",
        f"- chunk_rows: {payload.get('chunk_rows')}",
        f"- frame_instances: {payload.get('extraction_report', {}).get('frame_instances')}",
        f"- frame_elements: {payload.get('extraction_report', {}).get('frame_elements')}",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary_path
