# FrameBase setup

FrameBase TTL files are source data. They are large, so normal graph builds should use a compact
SQLite index instead of reparsing TTL every time.

## One-time setup

Download or register the files:

```bash
fst2framegraph setup-framebase --out data/framebase
```

If the files are already present:

```bash
fst2framegraph setup-framebase --out data/framebase --manifest-only
```

Then build or reuse the index:

```bash
fst2framegraph setup-framebase --out data/framebase --manifest-only --build-index
```

or:

```bash
fst2framegraph build-framebase-index --framebase-dir data/framebase
```

Expected current FrameBase 2.0 source files can be compressed or uncompressed:

```text
FrameBase_schema_core.ttl[.gz]
FrameBase_schema_dbps.ttl[.gz]
dereificationRulesSpinFormat.ttl[.gz]
clusters.txt
clusterPairs.txt
lexicalClusters.txt
manual/FrameBase_schema_manual_extensions.ttl
manual/inferenceRulesForSchema.txt
```

The index is written as:

```text
data/framebase/framebase_index.sqlite
```

## Runtime builds

Use the index directly:

```bash
fst2framegraph build \
  --input fst_clean/frame_elements_long.csv \
  --out outputs/framegraph \
  --framebase-index data/framebase/framebase_index.sqlite
```

If `--framebase-dir` contains `framebase_index.sqlite`, the build command will auto-discover it.
Raw TTL parsing is only the fallback when no usable index is supplied.

Dereification rules are optional. Missing or unparsable rules never break schema indexing; the
index builder reports that official DBP dereified edges are disabled.

`FrameBase_schema_dbps.ttl.gz` alone is not enough for official direct edges. The DBP schema gives
predicate vocabulary and labels; `dereificationRulesSpinFormat.ttl.gz` gives the actual current
FrameBase 2.0 dereification mappings.
