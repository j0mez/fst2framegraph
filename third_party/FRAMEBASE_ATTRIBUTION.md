# FrameBase attribution

This project can optionally use FrameBase data files for FrameNet-compatible frame and frame-element IRIs and for FrameBase Reification-Dereification direct binary predicate rules.

FrameBase data is licensed under the Creative Commons Attribution 4.0 International License by the FrameBase team at Aalborg University and Rutgers University.

- Website: https://www.framebase.org/
- Data page: https://www.framebase.org/data
- Licence: https://creativecommons.org/licenses/by/4.0/

When using the FrameBase schema or dereification-rule outputs produced by this package, cite FrameBase in the associated paper, report or software release.

The repository does not vendor the full FrameBase downloads by default. Users can retrieve them with:

```bash
fst2framegraph setup-framebase --out data/framebase
```

or provide local paths explicitly to `fst2framegraph build`.
