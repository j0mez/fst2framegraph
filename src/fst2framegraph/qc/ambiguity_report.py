from __future__ import annotations

import pandas as pd


def repeated_frame_warnings(frame_instances: pd.DataFrame) -> list[str]:
    if frame_instances.empty:
        return []
    grouped = frame_instances.groupby(["sentence_id", "frame_name"]).size().reset_index(name="n")
    repeats = grouped[grouped["n"] > 1]
    if repeats.empty:
        return []
    return [
        f"{len(repeats)} sentence/frame groups contain repeated frame names. "
        "Use parser-provided frame indices or target spans when possible."
    ]
