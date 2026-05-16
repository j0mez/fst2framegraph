#!/usr/bin/env bash
set -euo pipefail
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
fst2framegraph build --input examples/toy_fst_output.csv --out outputs/toy
