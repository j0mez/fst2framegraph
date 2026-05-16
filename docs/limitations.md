# Limitations

- The package does not perform frame-semantic parsing.
- If the parser output lacks frame indices or spans, repeated frame instances in one sentence may be merged.
- v0.1 nesting uses target-text containment when spans are not available.
- FrameBase ReDer edges are structural inferences, not guaranteed rhetorical interpretations.
- Direct binary edges should be used with provenance, examples and close reading.
