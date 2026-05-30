# Static web artifact

`fst2framegraph build` and `run_pipeline.py` write a static JSON artifact under
`web_artifact/` in each FrameBase graph output directory. The artifact is for
dashboards, notebooks and static hosting workflows that inspect an already
computed run. It does not run Frame Semantic Transformer or FrameBase inference.

## Files

```text
web_artifact/
  manifest.json
  summary.json
  documents.json
  sentences.json
  frames.json
  frame_elements.json
  nested_edges.json
  direct_edges.json
  dereification_diagnostics.json
```

`manifest.json` records `artifact_type="fst2framegraph.web_artifact"`,
`schema_version=1`, the file list and the graph build manifest. The data files
are JSON arrays except `summary.json`, which contains dashboard counts plus the
original build summary.

## Intended workflow

1. Run FST locally or in Colab into a canonical run directory.
2. Build the graph with a real FrameBase index.
3. Share or host the `web_artifact/` folder for inspection.

The artifact preserves provenance fields from the normal outputs: document IDs,
sentence IDs, frame instance IDs, frame element fillers, direct-edge rule IDs
and dereification diagnostics.
