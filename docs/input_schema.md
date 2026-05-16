# Input schema

The minimal long-table input is one row per frame-element filler.

Required columns:

| Column | Meaning |
|---|---|
| `doc_id` | Document, advert or source identifier |
| `sentence` | Sentence text |
| `frame_name` | FrameNet frame name |
| `element_name` | Frame element / semantic role name |
| `element_filler` | Text filling the role |

Recommended columns:

| Column | Why it helps |
|---|---|
| `sentence_id` | Stable sentence identity |
| `frame_index` | Distinguishes repeated frame instances in the same sentence |
| `target_text` | Enables nested frame detection |
| `target_start`, `target_end` | Span-based nested frame detection |
| `filler_start`, `filler_end` | Span-based nested frame detection |
| `confidence` | Parser confidence or score |

The tool can detect common OxCCAL/FST column names automatically, but explicit columns are safer.
