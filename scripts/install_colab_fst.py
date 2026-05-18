from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WHEEL_DIR = ROOT / "wheels"
SENTENCEPIECE_WHEEL = (
    WHEEL_DIR
    / "sentencepiece-0.2.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
)


def _pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", *args])


def main() -> int:
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if not SENTENCEPIECE_WHEEL.exists():
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "fetch_wheels.py")])

    _pip("install", "--force-reinstall", str(SENTENCEPIECE_WHEEL))
    _pip("install", "protobuf>=3.20.1,<4.0.0", "transformers>=4.39,<5", "torch>=2.2")
    _pip("install", "--no-deps", "frame-semantic-transformer==0.10.0")
    _pip("install", "--find-links", str(WHEEL_DIR), "-e", str(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
