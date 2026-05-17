# Mini Pilot Report

Date: 2026-05-17

## Scope

This mini-pilot tested whether `fst2framegraph` can produce useful graph outputs for short climate, corporate, and energy sentences.

Pipeline attempted:

1. Raw sentences in `data/pilot/mini_pilot_sentences.csv`
2. Real FST inference attempt with `fst2framegraph detect`
3. Fallback to synthetic graph-ready fixture in `data/pilot/mini_pilot_fst_output.csv` because the local `frame_semantic_transformer` package was installed but its Hugging Face model weights were not cached and the environment had no network access
4. `fst2framegraph prepare`
5. `fst2framegraph build` against a temporary tiny FrameBase index with a small rule set

The tiny FrameBase fixture was created in a temporary directory only and was not committed. These
results prove graph mechanics and ambiguity handling; they are not claims about current real
FrameBase 2.0 coverage for the same toy sentences.

## Counts

| metric | value |
| --- | ---: |
| number of sentences | 24 |
| number of frame instances | 27 |
| number of frame elements | 56 |
| nested_edges | 3 |
| projected_fe_edges | 0 |
| official_framebase_reder_edges | 16 |
| ambiguous matches | 1 |
| unmatched opportunities | 0 |
| dereification opportunities | 17 |

Notes:

- `official_framebase_reder_edges` came from the tiny-fixture build summary and matched the `graph_edges_dereified.csv` row count for that fixture.
- `unmatched opportunities` was `0`, but that does not mean coverage was complete. In this pilot, frames with no dereification rules were skipped before they reached that counter.

## What Worked

The output is useful for the intended research direction when the input already has frame structure and the FrameBase rule pack covers the frame/role pair.

Examples of good direct edges from the temporary tiny fixture:

- `p002`: `ExxonMobil -> invests_in -> carbon capture`
- `p011`: `Banks -> finances -> renewable energy projects`
- `p015`: `Airlines -> buys -> sustainable aviation fuel`
- `p020`: `Data centers -> consumes -> more electricity`
- `p021`: `Mining companies -> restores -> damaged land`

Useful nested edges were also found:

- `p001`: the `Capability` event filler `reduce emissions` contains the child frame target `reduce`
- `p003`: the `Assistance` goal filler `industry lower emissions` contains the child frame target `lower`
- `p023`: the `Designing` goal filler `use less energy` contains the child frame target `use`

## What Broke Or Stayed Weak

Examples of bad, missing, or ambiguous edges:

- `p003`: `Assistance(Helper, Goal)` was ambiguous because two temporary rules matched the same pair, so no official direct edge was emitted.
- `p004`: `Energy demand is rising.` produced a plausible frame instance but no direct edge because no dereification rule existed for `Change_position_on_a_scale`.
- `p010`: `Governments set stricter methane rules.` stayed reified only; there was no direct predicate for the `Setting` frame.
- `p024`: `Energy companies balance profits and transition spending.` remained hard to flatten into one safe binary predicate without losing nuance.

Broadly, the pilot suggests:

- Reified and nested outputs are already informative.
- Direct binary edges are useful when rules are precise; real FrameBase may still leave plausible
  sentences ambiguous or unmatched.
- Coverage gaps are driven more by frame/rule availability than by graph export itself.
- Ambiguity handling is conservative, which is good for auditability but lowers direct-edge yield.

## Assessment

For this use case, `fst2framegraph` is already useful as an auditable graph-construction layer, especially for:

- company-action relations
- investment and financing relations
- product or technology adoption relations
- nested event structures such as `help -> lower emissions`

It is less useful today for broad state-change, policy-setting, balancing, or expectation-style sentences unless a stronger rule inventory is available.

## Recommended Next Improvements

1. Cache the real FST model locally so the raw-sentence path works in offline or restricted environments.
2. Add a committed tiny reproducible FrameBase fixture for pilot and smoke workflows, including one intentional ambiguity case.
3. Expand dereification coverage for common climate/corporate frames such as reporting, regulation, transition spending, target-setting, and state-change.
4. Separate `no rule available` from `rule lookup attempted but unmatched` in the summary metrics; the current `unmatched opportunities` number hides an important failure mode.
5. Add a small benchmark set with expected direct edges so pilot regressions can be checked automatically.
