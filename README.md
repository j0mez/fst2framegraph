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
python -m pip install --find-links=wheels/ -e .
```

For Colab/Python 3.12 FST detection, fetch the bundled wheel first, then run
the installer that installs `sentencepiece==0.2.0` before the upstream FST
package:

```bash
python scripts/fetch_wheels.py
python scripts/install_colab_fst.py
```

The expected wheel is
`wheels/sentencepiece-0.2.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`.
When the wheel is present, `python -m pip install --find-links=wheels/ -e .`
uses it and avoids a source build. Core `fst2framegraph` graph building,
inspection, conversion, materialisation and FrameBase indexing do not require
Torch or FST and work on Python 3.10-3.12.

## OxCCAL CSV pipeline

For an OxCCAL-style CSV with a `Transcript (text and audio)` column:

```bash
python run_pipeline.py oxccal_sample.csv --out pipeline_outputs
```

The runner extracts `[ad text:]` content, discards `[audio transcript:]` and
similar sections, skips empty ads with a warning, runs FST inference when the
FST stack is installed, and writes a timestamped output folder containing:

- `frame_graph.graphml`
- `agent_frame_lift.csv`
- `agent_frame_communities.json`
- `summary_report.txt`

If `frame-semantic-transformer` is unavailable, the runner uses a small offline
fallback backend for smoke tests and demos. Add `--require-real-fst` to fail
instead of falling back.

For resumable, chunked production runs, use the v2 one-call pipeline API or CLI:

```bash
fst2framegraph pipeline \
  --input oxccal_sample.csv \
  --text-col "Transcript (text and audio)" \
  --id-col "Unique ID" \
  --doc-col "Unique ID" \
  --out-root outputs
```

```python
from fst2framegraph import run_fst2graph

result = run_fst2graph(
    input_csv="oxccal_sample.csv",
    text_col="Transcript (text and audio)",
    id_col="Unique ID",
    doc_col="Unique ID",
)
```

The v2 path uses the same strict transcript cleaning and Colab install hints,
while adding chunk mapping, resumable extraction state, GraphML/pickle graph
outputs, analysis tables, and JSON/Markdown run summaries.

## Colab notebook

Open `run_in_colab.ipynb` in a fresh Colab runtime, run the cells, upload the
project zip if prompted, then upload the CSV. The notebook installs from
`wheels/`, runs `run_pipeline.py`, and displays the summary and lift table.

`fst2framegraph` is open-source software under the Apache License 2.0.

## Reusable FrameGraphBuilder API

For downstream analysis code, use the stable graph construction and read-only analysis layer directly:

```python
from fst2framegraph import AnalysisBase, FrameGraphBuilder, from_fst_output

documents = from_fst_output("path/to/fst_output")
builder = FrameGraphBuilder()
graph = builder.build_graph(documents)
builder.save_graph(graph, "my_graph.graphml")

analysis = AnalysisBase(graph)
agent_frames = analysis.frames_for_filler("we", role="Agent")
paths = analysis.trace_paths("we", max_depth=2, role_filters=["Agent", "Goal"])
assoc_df = analysis.agent_frame_lift(top_n_frames=20, top_n_agents=30)
```

The graph schema is documented in `docs/graph_schema.md`. Fillers are globally
merged by normalized text, sentence nodes are included by default, and no
domain-specific roles or categories are hard-coded.

## One-command quickstart

`run` is the easiest entry point. It means: inspect the input, plan the workflow, then execute the
safe next steps.

Prepare any supported file or folder as a canonical run directory:

```bash
fst2framegraph run --input YOUR_FILE_OR_FOLDER --out fst_clean
```

Prepare and build a graph in one go:

```bash
fst2framegraph run \
  --input YOUR_FILE_OR_FOLDER \
  --out fst_clean \
  --graph \
  --framebase-index framebase_index.sqlite
```

Preview without writing files, loading pickles, running FST, or building graphs:

```bash
fst2framegraph run --plan --input YOUR_FILE_OR_FOLDER --out fst_clean
```

Use `--interactive` for guided questions. Non-interactive mode is the default for scripts, Colab
and CI.

## The smooth path

What do you have?

```text
Raw text / sentence CSV
  -> fst2framegraph run --input sentences.csv --text-col sentence --id-col sentence_id --doc-col doc_id --out fst_clean

Existing FST output
  -> fst2framegraph run --input old_fst_outputs --out fst_clean

Weird folder / unknown files
  -> fst2framegraph inspect --input whatever

Clean canonical run directory
  -> fst2framegraph build --input fst_clean --out graph_output --framebase-index framebase_index.sqlite
```

For existing FST-like outputs:

```bash
fst2framegraph run \
  --input old_fst_outputs \
  --out fst_clean \
  --graph-out graph_output \
  --framebase-index data/framebase/framebase_index.sqlite
```

For raw text or a sentence CSV:

```bash
fst2framegraph run \
  --input sentences.csv \
  --text-col sentence \
  --id-col sentence_id \
  --doc-col doc_id \
  --out fst_clean \
  --graph-out graph_output \
  --framebase-index data/framebase/framebase_index.sqlite
```

For long-text-per-row CSVs, `run` now chunks text into sentence-like
rows automatically before FST (`--chunk-text` is enabled by default). It
writes:

- `text_chunks.csv` (chunked rows used for parsing)
- `text_chunk_mapping.csv` (source-row to chunk provenance)

Use `inspect` when you are unsure what a file or folder contains. The explicit commands
`prepare`, `detect`, `materialise`, `doctor` and `build` remain available for debugging and
reproducible advanced workflows.

## Set up FrameBase files

The full FrameBase files are external data. They are not committed to this repository by default.
FrameBase resources are downloaded or registered separately and remain under their own licence.
By default, `setup-framebase` downloads from this project's GitHub Release assets
for stability, with FrameBase upstream URLs as fallback.

Download them into the local cache:

```bash
fst2framegraph setup-framebase --out data/framebase
```

If you already downloaded them manually, put the files into `data/framebase/` and run:

```bash
fst2framegraph setup-framebase --out data/framebase --manifest-only
```

Minimal current FrameBase 2.0 files:

```text
FrameBase_schema_core.ttl.gz
FrameBase_schema_dbps.ttl.gz
dereificationRulesSpinFormat.ttl.gz
clusters.txt
clusterPairs.txt
lexicalClusters.txt
manual/FrameBase_schema_manual_extensions.ttl
manual/inferenceRulesForSchema.txt
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

You can also pass the source files explicitly:

```bash
fst2framegraph build \
  --input /path/to/FST_output_long.csv \
  --out outputs/framegraph \
  --framebase-core data/framebase/FrameBase_schema_core.ttl.gz \
  --dbp-labels data/framebase/FrameBase_schema_dbps.ttl.gz \
  --dered-rules data/framebase/dereificationRulesSpinFormat.ttl.gz \
  --require-framebase
```

If no FrameBase index or source files are supplied, the graph build still works with generated
fallback IRIs. Raw TTL files are source data; the recommended runtime path is the SQLite index.

## ReDer matching

Nested/reified FE edges are always derivable from FST output. Official direct DBP edges are only
emitted when the current FrameBase 2.0 SPIN dereification rules match safely.

- DBP schema files provide predicate vocabulary and labels.
- SPIN dereification rules provide the actual mapping from frame plus FE pair to DBP predicate.
- Ambiguous matches are reported in `dereification_diagnostics.csv`, not guessed silently.

Tiny local fixtures in tests and smoke scripts prove the mechanics; they do not promise that the
same toy frames are covered by current real FrameBase. For example, the common toy
`Capability / Entity=Technology / Event=reduce emissions / target=can` remains ambiguous/unmatched
against the current real FrameBase index because there is no unique `Capability.can.verb`
Entity/Event rule.

A small real-index positive probe is:

```csv
doc_id,sentence_id,sentence,frame_name,frame_index,target_text,target_start,target_end,element_name,element_filler,filler_start,filler_end
doc-real,s-real,Companies use renewable power to reduce emissions.,Using,0,use,10,13,Agent,Companies,0,9
doc-real,s-real,Companies use renewable power to reduce emissions.,Using,0,use,10,13,Purpose,reduce emissions,33,49
```

With a current full FrameBase 2.0 index, this resolves through the unique
`Using / target=use / Agent -> Purpose` rule and emits one
`official_framebase_reder_edge` with DBP predicate `Using.usesForPurpose`.

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
fst2framegraph run \
  --input sentences.csv \
  --text-col sentence \
  --id-col sentence_id \
  --doc-col doc_id \
  --out outputs/fst_clean \
  --dedupe \
  --resume
```

`detect` is the explicit lower-level equivalent for raw text:

```bash
fst2framegraph detect \
  --input sentences.csv \
  --text-col sentence \
  --id-col sentence_id \
  --doc-col doc_id \
  --out outputs/fst_clean \
  --dedupe \
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

Before FST inference, exact text dedupe is enabled by default. It runs the parser once per unique
sentence text and then expands the result back to every original `sentence_id`, `doc_id` and
`row_index`. Use `--no-dedupe` or `dedupe=False` to preserve one-row-one-FST-call behavior. The
optional `normalised` mode only trims leading/trailing whitespace and collapses repeated internal
whitespace; fuzzy, embedding, semantic, lowercase, punctuation-stripping or Levenshtein dedupe is
deliberately not included.

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

import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_FLAX"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
!pip install "protobuf>=3.20.1,<4.0.0"
!pip install -r requirements-colab.txt
!pip install -e .

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
