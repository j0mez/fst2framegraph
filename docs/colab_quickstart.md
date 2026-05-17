# Minimal Colab quickstart

This workflow uses Google Drive as a mounted filesystem. `fst2framegraph` does not implement Google
Drive API authentication.

```python
from google.colab import drive
drive.mount("/content/drive")
```

Install with the optional FST stack:

```python
import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_FLAX"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
!pip install "protobuf>=3.20.1,<4.0.0"
!pip install -r requirements-colab.txt
!pip install -e .
```

The Colab/FST path depends on the upstream Frame Semantic Transformer package.
If your runtime uses Python 3.12 and the FST dependency stack is unavailable,
run the core graph/FrameBase workflows from existing clean outputs or switch to
a Python 3.10/3.11 runtime for FST inference.

Create or load sentences:

```python
import pandas as pd

df = pd.DataFrame({
    "sentence_id": ["s1", "s2"],
    "doc_id": ["doc1", "doc1"],
    "sentence": [
        "Technology can help consumers reduce emissions.",
        "Energy demand is rising across cities.",
    ],
})
```

Run FST with resumable outputs:

```python
from frame_semantic_transformer import FrameSemanticTransformer
from fst2framegraph import encode_with_fst

fst = FrameSemanticTransformer()

report = encode_with_fst(
    fst=fst,
    data=df,
    sentence_col="sentence",
    sentence_id_col="sentence_id",
    doc_col="doc_id",
    out_dir="/content/drive/MyDrive/fst2framegraph_runs/fst_clean",
    resume=True,
    checkpoint_every=100,
)
report
```

Rebuild materialised CSVs at any time:

```python
!fst2framegraph materialise --run-dir /content/drive/MyDrive/fst2framegraph_runs/fst_clean
```

Build or reuse the FrameBase index:

```python
!fst2framegraph build-framebase-index \
  --framebase-dir /content/drive/MyDrive/framebase \
  --index /content/drive/MyDrive/framebase/framebase_index.sqlite
```

Build the graph:

```python
!fst2framegraph build \
  --input /content/drive/MyDrive/fst2framegraph_runs/fst_clean/frame_elements_long.csv \
  --out /content/drive/MyDrive/fst2framegraph_runs/graph \
  --framebase-index /content/drive/MyDrive/framebase/framebase_index.sqlite \
  --doc-col doc_id \
  --sentence-col sentence \
  --sentence-id-col sentence_id \
  --frame-col frame_name \
  --frame-index-col frame_index \
  --target-col target_text \
  --target-start-col target_start \
  --target-end-col target_end \
  --fe-col element_name \
  --filler-col element_filler \
  --filler-start-col filler_start \
  --filler-end-col filler_end
```

Inspect the summary:

```python
import json
from pathlib import Path

summary_path = Path("/content/drive/MyDrive/fst2framegraph_runs/graph/summary.json")
json.loads(summary_path.read_text())
```
