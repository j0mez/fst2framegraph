# v0.3 smoke workflow

This smoke workflow exercises the v0.3 reliability path with a real FST model, materialisation,
FrameBase indexing, and graph building. If the FST model is not already cached, the first Python
step needs network access to resolve/download the Hugging Face model.

## Python API smoke test

```python
import pandas as pd
from frame_semantic_transformer import FrameSemanticTransformer
from fst2framegraph import encode_with_fst

df = pd.DataFrame({
    "sentence_id": ["s1", "s2"],
    "doc_id": ["doc1", "doc1"],
    "sentence": [
        "Technology can help consumers reduce emissions.",
        "Energy demand is rising across cities.",
    ],
})

fst = FrameSemanticTransformer()

report = encode_with_fst(
    fst=fst,
    data=df,
    sentence_col="sentence",
    sentence_id_col="sentence_id",
    doc_col="doc_id",
    out_dir="outputs/fst_clean_v03_smoke",
    resume=True,
    checkpoint_every=1,
)
```

Expected:

```text
errors = 0
sentences = 2
frame_instances > 0
frame_elements > 0
outputs/fst_clean_v03_smoke/fst_clean.jsonl exists
outputs/fst_clean_v03_smoke/progress.sqlite exists
frame_elements_long.csv contains a sentence column
no .pkl or .pickle files exist in the run directory
```

## Materialise smoke test

```bash
rm outputs/fst_clean_v03_smoke/*.csv
fst2framegraph materialise --run-dir outputs/fst_clean_v03_smoke
```

Expected:

```text
CSV files are rebuilt.
extraction_report.json is rebuilt.
frame_elements_long.csv is valid.
```

## FrameBase index smoke test

```bash
fst2framegraph build-framebase-index \
  --framebase-dir data/framebase \
  --index data/framebase/framebase_index.sqlite \
  --overwrite
```

Expected:

```text
data/framebase/framebase_index.sqlite exists.
Index build does not fail if dereification rules are missing.
Index manifest metadata is present in SQLite.
```

## Graph build smoke test

```bash
fst2framegraph build \
  --input outputs/fst_clean_v03_smoke/frame_elements_long.csv \
  --out outputs/fst_clean_v03_smoke_graph \
  --framebase-index data/framebase/framebase_index.sqlite \
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

Expected:

```text
summary.json exists.
FrameBase validation counts are present.
nested_edges is reported.
warnings are clear.
```

## Verified local result

Verified on 2026-05-16 from this repository's `.venv`.

Python API smoke:

```text
sentences = 2
frame_instances = 7
frame_elements = 12
errors = 0
duplicate_sentence_ids = 0
fst_clean.jsonl exists
progress.sqlite exists
frame_elements_long.csv contains sentence
no .pkl or .pickle files were written
```

Materialise smoke:

```text
CSV files and extraction_report.json were rebuilt after deleting materialised CSV outputs.
frame_elements_long.csv was valid with 12 rows.
```

FrameBase index smoke:

```text
data/framebase/framebase_index.sqlite was created.
frames = 118395
frame_element_lookup_keys = 32477
dbp_labels = 106519
rules = 0
warning = Dereification rules unavailable; DBP dereified edges disabled.
```

Graph build smoke:

```text
summary.json exists.
framebase_validated_frames = 7
framebase_unmatched_frames = 0
framebase_validated_frame_elements = 12
framebase_unmatched_frame_elements = 0
nested_edges = 7
dereified_edges = 0
warnings clearly report unavailable dereification rules.
```
