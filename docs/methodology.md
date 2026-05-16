# Methodology

`fst2framegraph` separates raw extraction from derived structure.

## Reified graph

Each frame instance is represented as an event-like node. Each frame element creates a role-labelled edge from the frame instance to a filler node.

This preserves the essential FrameNet structure: fillers are not treated as independent words but as roles inside a semantic scene.

## Nested graph

If one frame element filler contains another detected frame, the filler can be represented as a nested event. This is crucial for sentences such as:

> Technology can help consumers reduce emissions.

The `Goal` of an `Assistance` frame may itself contain a `Cause_change` or `Reducing` frame.

## Dereified graph

FrameBase ReDer rules can infer direct binary predicates between role fillers. These are useful for motif mining and network analysis, but they are always kept as a derived layer with provenance.
