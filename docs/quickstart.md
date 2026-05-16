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
