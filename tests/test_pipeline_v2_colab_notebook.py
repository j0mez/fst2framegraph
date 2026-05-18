from __future__ import annotations

import json
from pathlib import Path


def test_colab_notebook_contains_one_call_pipeline_contract() -> None:
    notebook = json.loads(Path("run_in_colab.ipynb").read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    assert len(code_cells) >= 3

    source_text = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
    assert "run_fst2graph(" in source_text
    assert "requirements-colab.txt" in source_text
    assert "USE_TF" in source_text
    assert "TRANSFORMERS_NO_TF" in source_text
