# Quickstart

Install in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

Run the toy example:

```bash
fst2framegraph build --input examples/toy_fst_output.csv --out outputs/toy
```

Open:

```text
outputs/toy/sentence_graphs.jsonl
outputs/toy/graph_edges_reified.csv
outputs/toy/qc_report.json
```

Run the local no-network smoke path:

```bash
bash scripts/smoke_test.sh
```

## The smooth path

What do you have?

```text
Raw text / sentence CSV
  -> fst2framegraph detect
  -> fst2framegraph build

Existing FST output
  -> fst2framegraph prepare
  -> fst2framegraph build

Weird folder / unknown files
  -> fst2framegraph inspect

Clean canonical run directory
  -> fst2framegraph build --input fst_clean
```

For existing FST outputs:

```bash
fst2framegraph prepare --input old_fst_outputs --out fst_clean
fst2framegraph build \
  --input fst_clean \
  --out graph_output \
  --framebase-index data/framebase/framebase_index.sqlite
```

For raw text:

```bash
fst2framegraph detect \
  --input sentences.csv \
  --text-col sentence \
  --id-col sentence_id \
  --doc-col doc_id \
  --out fst_clean \
  --resume

fst2framegraph build \
  --input fst_clean \
  --out graph_output \
  --framebase-index data/framebase/framebase_index.sqlite
```

Core graph building, inspection, preparation, materialisation, and FrameBase indexing work on
Python 3.10-3.12. Real Frame Semantic Transformer inference is recommended on Python 3.10/3.11
because the upstream FST dependency stack is constrained on Python 3.12.

## Reliable FST run

For long FST runs, write to a run directory and resume from checkpoints:

```python
from frame_semantic_transformer import FrameSemanticTransformer
from fst2framegraph import encode_with_fst

report = encode_with_fst(
    fst=FrameSemanticTransformer(),
    data="sentences.csv",
    sentence_col="sentence",
    sentence_id_col="sentence_id",
    out_dir="outputs/fst_clean",
    resume=True,
    checkpoint_every=100,
    batch_size=16,
)
```

The authoritative files are `fst_clean.jsonl` and `progress.sqlite`. Rebuild CSV outputs after any
interrupted run with:

```bash
fst2framegraph materialise --run-dir outputs/fst_clean
```

`frame_elements_long.csv` is graph-ready when it includes instance identifiers plus target and
filler spans: `sentence_id`, `sentence`, `frame_index`, `frame_name`, `target_text`,
`target_start`, `target_end`, `element_name`, `element_filler`, `filler_start`, and `filler_end`.
Flat-only exports can support simple counts, but reliable nested graphs require `frame_index` and
target/filler spans.

Pickles are unsafe by default because Python pickles can execute code. `prepare` and `convert` only
load trusted pickle files when `--allow-pickle` is passed, and they write portable JSONL/CSV
outputs rather than new pickle files.

FrameBase data is external and not bundled in this repository. Download or register the FrameBase
files separately, then build `data/framebase/framebase_index.sqlite` with
`fst2framegraph setup-framebase --out data/framebase --build-index`.
