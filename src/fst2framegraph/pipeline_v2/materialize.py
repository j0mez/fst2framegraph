from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fst2framegraph.fst import materialise_run


def materialize_outputs(*, run_dir: str | Path, chunk_mapping: pd.DataFrame) -> dict[str, Any]:
    run_dir = Path(run_dir)
    report = materialise_run(run_dir)

    mapping_path = run_dir / "chunk_mapping.csv"
    chunk_mapping.to_csv(mapping_path, index=False)

    frame_instances_path = run_dir / "frame_instances.csv"
    frame_elements_path = run_dir / "frame_elements_long.csv"

    if frame_instances_path.exists() and not chunk_mapping.empty:
        frame_instances = pd.read_csv(frame_instances_path)
        expanded_frames = frame_instances.merge(
            chunk_mapping,
            on="sentence_id",
            how="left",
        )
        expanded_frames.to_csv(run_dir / "frame_instances_expanded.csv", index=False)
    else:
        pd.DataFrame().to_csv(run_dir / "frame_instances_expanded.csv", index=False)

    if frame_elements_path.exists() and not chunk_mapping.empty:
        frame_elements = pd.read_csv(frame_elements_path)
        expanded_elements = frame_elements.merge(
            chunk_mapping,
            on="sentence_id",
            how="left",
        )
        expanded_elements.to_csv(run_dir / "frame_elements_expanded.csv", index=False)
    else:
        pd.DataFrame().to_csv(run_dir / "frame_elements_expanded.csv", index=False)

    return report
