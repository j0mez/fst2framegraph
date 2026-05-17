# FrameGraphBuilder Schema

`FrameGraphBuilder` builds a generic, reified semantic event graph from parsed
frame-semantic documents. This graph is the stable construction layer for later
analysis; it does not hard-code domains, agent roles, FrameBase assumptions, or
research-specific categories.

## Minimal Usage

```python
from fst2framegraph import FrameGraphBuilder

builder = FrameGraphBuilder()
graph = builder.build_graph(documents)
```

## Input Documents

`build_graph()` accepts a list of dictionaries:

```python
{
    "doc_id": "unique_string",
    "text": "raw text, optional",
    "metadata": {"any_key": "any_value"},
    "frames": [
        {
            "frame_type": "Investing",
            "trigger": "investing",
            "sent_idx": 0,
            "frame_elements": [
                {"role": "Agent", "text": "We"},
                {"role": "Goal", "text": "clean energy"},
            ],
        },
    ],
}
```

`sent_idx` is optional. If missing, frames are placed under sentence index `0`.
Adapters are provided for common existing fst2framegraph outputs:

```python
from fst2framegraph import from_fst_output, from_frame_elements_long_csv, from_legacy_pickle
```

## Node Types

| Type | ID format | Required attributes |
| --- | --- | --- |
| `Document` | `doc:{doc_id}` | `node_type`, `doc_id`, plus flat user metadata |
| `Sentence` | `sent:{doc_id}:{sent_idx}` | `node_type`, `doc_id`, `sent_idx` |
| `FrameInstance` | `frame:{doc_id}:{i}` | `node_type`, `frame_type`, `trigger`, `doc_id`, `sent_idx` |
| `Filler` | `filler:{hash}` | `node_type`, `text` |

`i` is the running frame index within the whole document.

Sentence nodes are included by default. Set
`FrameGraphBuilder(include_sentence_nodes=False)` to connect documents directly
to frame instances.

## Edge Types

| Edge label | From | To | Attributes |
| --- | --- | --- | --- |
| `HAS_FRAME` | `Document` | `Sentence` | `edge_type="HAS_FRAME"` |
| `HAS_FRAME` | `Sentence` | `FrameInstance` | `edge_type="HAS_FRAME"` |
| frame-element role | `FrameInstance` | `Filler` | `edge_type=role`, `role=role` |

When sentence nodes are disabled, `HAS_FRAME` edges go directly from
`Document` to `FrameInstance`.

Role edges use a `networkx.MultiDiGraph` edge key equal to the role name, such
as `Agent`, `Cause`, `Theme`, or `Goal`.

## Filler Merging

Filler nodes are globally merged across the entire graph. If two documents use
the same normalized filler text, both frame instances point to the same filler
node. This is the mechanism that makes cross-document path tracing possible.

Default filler normalization:

1. Convert to string.
2. Strip leading and trailing whitespace.
3. Lowercase.
4. Collapse multiple whitespace characters to one space.
5. Remove one trailing punctuation mark from `. , ; : ! ?`.

Filler IDs use the first 12 hex characters of SHA-256 over the normalized text:

```text
filler:{sha256(normalized_text)[:12]}
```

Override normalization with:

```python
builder = FrameGraphBuilder(normalize_filler=my_normalizer)
```

## Saving And Loading

`FrameGraphBuilder.save_graph(graph, path)` and
`FrameGraphBuilder.load_graph(path)` support:

- `.graphml` for interoperable graph exchange.
- `.pkl`, `.pickle`, and `.gpickle` for Python/networkx round-trips.

Pickle graph files are for trusted local use only.
