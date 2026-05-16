from __future__ import annotations

from pathlib import Path

import pandas as pd

from fst2framegraph.io.column_detection import detect_columns
from fst2framegraph.schema import ColumnMap


def read_fst_csv(path: Path, column_map: ColumnMap | None = None) -> tuple[pd.DataFrame, ColumnMap]:
    df = pd.read_csv(path)
    cmap = column_map or detect_columns(df)
    return df, cmap
