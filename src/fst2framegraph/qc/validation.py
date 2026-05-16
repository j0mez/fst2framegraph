from __future__ import annotations

from pathlib import Path


def require_file(path: Path | None, label: str) -> None:
    if path is None:
        return
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
