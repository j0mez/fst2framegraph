# fst2framegraph

`fst2framegraph` converts FrameNet-style parser output into auditable semantic graph structures:

1. **reified frame graphs**: frame instance → frame element role → filler
2. **nested frame graphs**: frame instance → role → child frame instance, when an FE filler contains another detected frame
3. **FrameBase dereified graphs**: filler → direct binary predicate → filler, using FrameBase ReDer rules where available

The package is designed for research workflows where the evidence must remain traceable back to the original sentence, frame instance, role and FrameBase rule.

## Install locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e ".[dev]"
python -m pytest
```

For core graph/build use without the FST inference stack:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

For full FST detection use:

```bash
python -m pip install -r requirements-fst.txt
python -m pip install -e ".[fst]"
```

The upstream Frame Semantic Transformer stack currently installs most reliably on
Python 3.10/3.11. Core `fst2framegraph` graph building, inspection, conversion,
materialisation and FrameBase indexing do not require Torch or FST and work on
Python 3.10-3.12.

`fst2framegraph` is open-source software under the Apache License 2.0.

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

For existing FST-like outputs:

```bash
fst2framegraph prepare --input old_fst_outputs --out fst_clean
fst2framegraph build \
  --input fst_clean \
  --out graph_output \
  --framebase-index data/framebase/framebase_index.sqlite
```

For raw text or a sentence CSV:

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

Use `inspect` when you are unsure what a file or folder contains. Use `prepare` when you already
have FST-like outputs and want a canonical run directory for graph building.

## Set up FrameBase files

The full FrameBase files are external data. They are not committed to this repository by default.
FrameBase resources are downloaded or registered separately and remain under their own licence.

Download them into the local cache:

```bash
fst2framegraph setup-framebase --out data/framebase
```

If you already downloaded them manually, put the files into `data/framebase/` and run:

```bash
fst2framegraph setup-framebase --out data/framebase --manifest-only
```

Expected files:

```text
FrameBase_schema_core.ttl.gz
FrameBase_schema_dbps.ttl.gz
dereificationRulesSparqlFormat.txt.zip
```

FrameBase data is licensed under Creative Commons Attribution 4.0 International by the FrameBase team at Aalborg University and Rutgers University. See `third_party/FRAMEBASE_ATTRIBUTION.md`.

## Build a full graph

For repeated builds, first build the compact FrameBase index once:

```bash
fst2framegraph setup-framebase --out data/framebase --build-index
```

Then normal builds can reuse the SQLite index instead of reparsing the large TTL files:

```bash
fst2framegraph build \
  --input /path/to/FST_output_long.csv \
  --out outputs/framegraph \
  --framebase-index data/framebase/framebase_index.sqlite \
  --require-framebase
```

You can also pass the three FrameBase files explicitly:

```bash
fst2framegraph build \
  --input /path/to/FST_output_long.csv \
  --out outputs/framegraph \
  --framebase-core data/framebase/FrameBase_schema_core.ttl.gz \
  --dbp-labels data/framebase/FrameBase_schema_dbps.ttl.gz \
  --dered-rules data/framebase/dereificationRulesSparqlFormat.txt.zip \
  --require-framebase
```

If no FrameBase index or source files are supplied, the graph build still works with generated
fallback IRIs. Raw TTL files are source data; the recommended runtime path is the SQLite index.

## Workflow A: run FST over raw text

Long FST runs should write to one run directory. The canonical state is:

```text
fst_clean.jsonl
progress.sqlite
```

CSV files are materialised convenience outputs and can be rebuilt at any time:

```bash
fst2framegraph materialise --run-dir outputs/fst_clean
```

CLI example:

```bash
fst2framegraph detect \
  --input sentences.csv \
  --text-col sentence \
  --id-col sentence_id \
  --doc-col doc_id \
  --out outputs/fst_clean \
  --resume
```

Python example:

```python
from frame_semantic_transformer import FrameSemanticTransformer
from fst2framegraph import encode_with_fst

fst = FrameSemanticTransformer()

report = encode_with_fst(
    fst=fst,
    data="sentences.csv",
    sentence_col="sentence",
    sentence_id_col="sentence_id",
    doc_col="doc_id",
    out_dir="outputs/fst_clean",
    resume=True,
    checkpoint_every=100,
    batch_size=16,
)
```

Every completed sentence produces one JSONL record containing `sentence_id`, `doc_id`,
`row_index`, `sentence`, `status="completed"`, and `frames`. Failed rows use
`status="error"` and preserve the error text. Resume uses
`progress.sqlite`; materialisation uses `fst_clean.jsonl`. If a CSV is interrupted or corrupted,
rerun `fst2framegraph materialise --run-dir ...`.

Input `sentence_id` values must be unique within a run because checkpoint/resume state is keyed by
`sentence_id`.

Raw Python/FST objects are never saved by default. Portable JSONL records, including normalised
frames and a JSON-safe parser result, are saved instead, so parser detail remains inspectable
without creating pickle files.

## Workflow B: build from existing clean FST output

If you already have a clean canonical run directory, build the graph directly:

```bash
fst2framegraph build \
  --input fst_clean \
  --out graph \
  --framebase-index data/framebase/framebase_index.sqlite
```

`build --input fst_clean` uses `fst_clean/frame_elements_long.csv` when present. If the CSV is
missing but `fst_clean.jsonl` exists, the command rebuilds the CSVs before graph construction.

If you only have a clean `frame_elements_long.csv`, that still works:

```bash
fst2framegraph build \
  --input fst_clean/frame_elements_long.csv \
  --out graph \
  --framebase-index data/framebase/framebase_index.sqlite
```

Nested graphs require instance-level data. At minimum, `frame_elements_long.csv` should include:

```text
sentence_id
sentence
frame_index
frame_name
target_text
target_start
target_end
element_name
element_filler
filler_start
filler_end
```

If you have older or messy FST outputs, inspect them first:

```bash
fst2framegraph inspect --input examples/flat_only_old_fst.csv
fst2framegraph inspect --input examples/fst_like.jsonl
fst2framegraph prepare --input examples/fst_like.jsonl --out fst_clean
```

`prepare` refuses flat-only CSVs as graph input because they lack the `frame_index` and
target/filler span columns needed for reliable nested graphs. It also refuses pickle files by
default because Python pickles can execute code:

```bash
fst2framegraph prepare --input trusted_pickles --out fst_clean --allow-pickle
```

Use `doctor` before long graph builds:

```bash
fst2framegraph doctor --run-dir fst_clean --framebase-index data/framebase/framebase_index.sqlite
```

The example `examples/flat_only_old_fst.csv` demonstrates an older insufficient flat output:
it has frame/role/filler text but lacks the `frame_index` and span columns needed for reliable
nested graphs. The example `examples/fst_like.jsonl` demonstrates a small convertible structured
export.

## Colab and Google Drive

Use Colab's mounted filesystem rather than Google Drive API auth:

```python
from google.colab import drive
drive.mount("/content/drive")

from frame_semantic_transformer import FrameSemanticTransformer
from fst2framegraph import encode_with_fst, setup_framebase

setup_framebase(
    framebase_dir="/content/drive/MyDrive/framebase",
    build_index=True,
    reuse_existing=True,
)

report = encode_with_fst(
    fst=FrameSemanticTransformer(),
    data="/content/drive/MyDrive/my_project/sentences.csv",
    sentence_col="sentence",
    sentence_id_col="sentence_id",
    out_dir="/content/drive/MyDrive/my_project/fst_runs/run_001",
    resume=True,
    checkpoint_every=100,
    batch_size=32,
)
```

## Custom column mapping example

```bash
fst2framegraph build \
  --input path/to/frame_elements_long.csv \
  --out outputs/framegraph_full \
  --framebase-dir data/framebase \
  --require-framebase \
  --doc-col "doc_id" \
  --sentence-col "sentence" \
  --frame-col "frame_name" \
  --fe-col "element_name" \
  --filler-col "element_filler" \
  --sentence-id-col "sentence_id" \
  --frame-index-col "frame_index"
```

If your parser output includes character offsets, pass them too. Span-based nesting is stronger than text-overlap nesting:

```bash
  --target-start-col target_start \
  --target-end-col target_end \
  --filler-start-col filler_start \
  --filler-end-col filler_end
```

## Main outputs

```text
documents.csv
sentences.csv
frame_instances.csv
frame_elements.csv
graph_nodes.csv
graph_edges_reified.csv
graph_edges_nested.csv
graph_edges_dereified.csv
sentence_graphs.jsonl
graph.graphml
graph.ttl
qc_report.json
summary.json
manifest.json
```

The `qc_report.json` and `summary.json` files record coverage and warnings: input rows, number of frame instances, frame elements, nested edges, dereified edges, FrameBase validation counts and parser/data warnings.

## No-network smoke test

The repository includes a small smoke script that uses only local fixtures. It imports the package,
inspects example outputs, converts a tiny JSONL fixture, materialises the run, builds a tiny
FrameBase index, builds a graph and confirms no pickle files were created:

```bash
bash scripts/smoke_test.sh
```

Real FST inference smoke tests are separate because `FrameSemanticTransformer()` may need to
download Hugging Face model files on first use.

## Methodological note

The tool keeps the original FrameNet-style annotation separate from the FrameBase-compatible graph interpretation. It does not claim to discover meaning beyond the parser output. It converts frame instances and role fillers into a graph form that can be inspected, queried, aggregated and compared.

## Citation

Use `CITATION.cff` for this package. When using FrameBase-backed outputs, also cite FrameBase.
