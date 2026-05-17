# SPIN dereification

`fst2framegraph` now treats current FrameBase 2.0 SPIN rules as the normal source of official
direct DBP edges.

## What each layer means

- Reified graph: frame instance nodes linked to FE fillers.
- Nested graph: frame instance linked to child frame instance when a filler contains another frame.
- Direct DBP graph: filler A linked to filler B through an official FrameBase DBP predicate.

## Important distinction

- `FrameBase_schema_dbps.ttl.gz` provides DBP vocabulary and labels.
- `dereificationRulesSpinFormat.ttl.gz` provides the actual dereification rules.
- DBP labels alone are not dereification rules.

## Conservative matching

FST output usually gives broad FrameNet frame names, while FrameBase dereification rules are often
microframe or lexical-unit specific. `fst2framegraph` therefore emits official DBP edges only when
one rule matches safely:

- exact microframe IRI when the input is already exact
- unique `frame_name + target_text + FE pair`
- unique `frame_name + FE pair`

If multiple rules match, no official edge is emitted. The ambiguity is written to
`dereification_diagnostics.csv`.
