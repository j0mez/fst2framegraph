# Local FrameBase cache

This directory is for the external FrameBase files used by `fst2framegraph`.

The full FrameBase files are **not committed** to the repository. They are downloaded or registered locally:

```bash
fst2framegraph setup-framebase --out data/framebase
```

If you have already downloaded the files manually, place them here and write the manifest:

```bash
fst2framegraph setup-framebase --out data/framebase --manifest-only
```

Expected files:

- `FrameBase_schema_core.ttl.gz`
- `FrameBase_schema_dbps.ttl.gz`
- `dereificationRulesSparqlFormat.txt.zip`

FrameBase data is licensed under Creative Commons Attribution 4.0 International by the FrameBase team at Aalborg University and Rutgers University.
