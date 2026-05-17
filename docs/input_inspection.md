# Input inspection and rescue

Use `inspect` when you already have FST outputs but are not sure whether they can be used by
`fst2framegraph`.

```bash
fst2framegraph inspect --input PATH
```

The command reports one of the following statuses:

- `graph_ready`: the input has the columns needed for graph building.
- `convertible`: JSON/JSONL-like records can be converted into canonical v0.3 output.
- `flat_only`: frame/FE/filler columns exist, but instance/spans are missing.
- `insufficient`: required structure is missing.
- `unsafe_without_pickle_permission`: pickle files were found but were not loaded.

## Graph-ready data

Reliable nested graphs require instance-level rows with spans:

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

If those columns are present, you can build directly:

```bash
fst2framegraph build --input frame_elements_long.csv --out graph
```

Or use the one-command path:

```bash
fst2framegraph run --input frame_elements_long.csv --out fst_clean --graph --framebase-index data/framebase/framebase_index.sqlite
```

## Flat-only data

Older flattened CSVs often have `frame_name`, `element_name`, and `element_filler`, but lack frame
indices and spans. These files can support simple frame/FE counts, but they are not sufficient for
reliable nested frame graphs.

Recommended recovery:

```bash
fst2framegraph inspect --input examples/flat_only_old_fst.csv
```

Then either rerun FST:

```python
encode_with_fst(..., resume=True)
```

or convert trusted richer exports:

```bash
fst2framegraph inspect --input examples/fst_like.jsonl
fst2framegraph convert --input examples/fst_like.jsonl --out fst_clean
```

## Pickle safety

Python pickles can execute code. `fst2framegraph` never loads pickle files by default.

Filename-only inspection can still identify likely FST batch files and missing numeric ranges:

```bash
fst2framegraph inspect --input raw_result_pickles
```

Only load pickles from trusted sources:

```bash
fst2framegraph convert --input raw_result_pickles --out fst_clean --allow-pickle
```

Conversion never writes new pickle files. It writes the canonical run directory:

```text
fst_clean.jsonl
progress.sqlite
sentences.csv
frame_instances.csv
frame_elements.csv
frame_elements_long.csv
errors.csv
extraction_report.json
manifest.json
```
