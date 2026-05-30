from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fst2framegraph.io.write_outputs import ensure_out_dir, write_json


WEB_ARTIFACT_FILES = {
    "summary.json",
    "documents.json",
    "sentences.json",
    "frames.json",
    "frame_elements.json",
    "nested_edges.json",
    "direct_edges.json",
    "dereification_diagnostics.json",
}


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    return df.to_dict(orient="records")


def write_web_artifact(
    *,
    out: Path,
    build_manifest: dict[str, Any],
    build_summary: dict[str, Any],
    documents: pd.DataFrame,
    sentences: pd.DataFrame,
    frame_instances: pd.DataFrame,
    frame_elements: pd.DataFrame,
    nested_edges: pd.DataFrame,
    direct_edges: pd.DataFrame,
    dereification_diagnostics: pd.DataFrame,
) -> Path:
    artifact_dir = out / "web_artifact"
    ensure_out_dir(artifact_dir)

    artifact_summary = {
        "documents": int(len(documents)),
        "sentences": int(len(sentences)),
        "frames": int(len(frame_instances)),
        "frame_elements": int(len(frame_elements)),
        "nested_edges": int(len(nested_edges)),
        "direct_edges": int(len(direct_edges)),
        "dereification_diagnostics": int(len(dereification_diagnostics)),
        "build_summary": build_summary,
    }
    write_json(artifact_summary, artifact_dir, "summary.json")
    write_json(_records(documents), artifact_dir, "documents.json")
    write_json(_records(sentences), artifact_dir, "sentences.json")
    write_json(_records(frame_instances), artifact_dir, "frames.json")
    write_json(_records(frame_elements), artifact_dir, "frame_elements.json")
    write_json(_records(nested_edges), artifact_dir, "nested_edges.json")
    write_json(_records(direct_edges), artifact_dir, "direct_edges.json")
    write_json(_records(dereification_diagnostics), artifact_dir, "dereification_diagnostics.json")
    write_json(
        {
            "artifact_type": "fst2framegraph.web_artifact",
            "schema_version": 1,
            "files": sorted(WEB_ARTIFACT_FILES),
            "build_manifest": build_manifest,
        },
        artifact_dir,
        "manifest.json",
    )
    return artifact_dir
