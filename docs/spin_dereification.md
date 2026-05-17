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

## Real-index positive probe

Tiny fixtures are useful for smoke tests, but they only prove parser/index/build mechanics. Real
FrameBase matching is intentionally conservative: plausible broad frames can remain ambiguous or
unmatched when FrameBase only has lexical-unit-specific microframes.

A current full FrameBase 2.0 index contains a simple positive for the `Using` frame:

```csv
doc_id,sentence_id,sentence,frame_name,frame_index,target_text,target_start,target_end,element_name,element_filler,filler_start,filler_end
doc-real,s-real,Companies use renewable power to reduce emissions.,Using,0,use,10,13,Agent,Companies,0,9
doc-real,s-real,Companies use renewable power to reduce emissions.,Using,0,use,10,13,Purpose,reduce emissions,33,49
```

Expected real-index result:

- `frame_name`: `Using`
- `target_text`: `use`
- subject FE: `Agent=Companies`
- object FE: `Purpose=reduce emissions`
- matched DBP predicate: `http://framebase.org/dbp/Using.usesForPurpose`
- match tier: `frame_target_fe_unique`

The Capability toy example is deliberately not documented as a real-positive example. Against the
current real FrameBase index, `Capability / target=can / Entity -> Event` has broad candidate rules
but no unique real `Capability.can.verb` Entity/Event dereification rule, so no official DBP edge is
emitted.
