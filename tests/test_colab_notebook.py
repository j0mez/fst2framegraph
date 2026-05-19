from __future__ import annotations

import json
from pathlib import Path


def test_colab_notebook_saves_pipeline_outputs_to_google_drive() -> None:
    notebook = json.loads(Path("run_in_colab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join(
        line
        for cell in notebook["cells"]
        for line in cell.get("source", [])
    )

    assert 'drive.mount("/content/drive")' in source
    assert "/content/drive/MyDrive/fst2framegraph_outputs" in source
    assert "output_root=output_root" in source
    assert "output_root='colab_outputs'" not in source
