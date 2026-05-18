# Colab quickstart (one-call)

This workflow uses Google Drive as a mounted filesystem. `fst2framegraph` does not implement Google
Drive API authentication.

```python
from google.colab import drive
drive.mount("/content/drive")
```

Install with the pinned Colab stack (no manual Python version changes needed):

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

Then run the product one-call API:

```python
from fst2framegraph import run_fst2graph

result = run_fst2graph(
    input_csv="/content/drive/MyDrive/your_folder/oxccal_20_sample.csv",
    out_root="/content/drive/MyDrive/fst2framegraph_runs",
    text_col="Transcript (text and audio)",
    id_col="Unique ID",
    doc_col="Unique ID",
    batch_size=16,
    dedupe=True,
    random_seed=42,
)
result
```

The full copy/paste notebook is committed as `run_in_colab.ipynb`.
