# Third-party notices

`fst2framegraph` is distributed under the Apache License 2.0. It does not bundle the large
FrameBase data dumps.

## Frame Semantic Transformer

The optional FST detection workflow uses `frame-semantic-transformer`, an external Python package
for FrameNet-style frame-semantic parsing. Install the optional FST dependencies only when you want
`fst2framegraph` to run FST inference itself.

Recommended citation:

David Chanin. 2023. *Open-source Frame Semantic Parsing*. arXiv:2303.12788.

## FrameNet

FrameNet is an external lexical database and annotation framework. `fst2framegraph` consumes
FrameNet-style parser output but does not redistribute FrameNet data.

Recommended citation:

Collin F. Baker, Charles J. Fillmore, and John B. Lowe. 1998. *The Berkeley FrameNet Project*.
DOI: 10.3115/980845.980860.

## FrameBase

FrameBase resources are downloaded or registered separately by the user and remain under their own
license. The FrameBase data page describes the data as licensed under Creative Commons Attribution
4.0 International by the FrameBase team at Aalborg University and Rutgers University.

`fst2framegraph` may build a compact local SQLite index from user-provided FrameBase source files,
but the repository should not commit or redistribute the large FrameBase TTL dumps.

Recommended citation:

Jacobo Rouces, Gerard de Melo, and Katja Hose. 2017. *FrameBase: Enabling integration of
heterogeneous knowledge*. DOI: 10.3233/SW-170279.
