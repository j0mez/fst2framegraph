from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fst2framegraph import run_fst2graph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-call fst2framegraph v2 pipeline runner.",
    )
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--out-root", default="outputs", help="Root output directory.")
    parser.add_argument("--text-col", default=None, help="Input text column.")
    parser.add_argument("--id-col", default=None, help="Input ID column.")
    parser.add_argument("--doc-col", default=None, help="Input document ID column.")
    parser.add_argument("--framebase-index", default=None, help="Optional FrameBase index path.")
    parser.add_argument("--framebase-dir", default=None, help="Optional FrameBase directory.")
    parser.add_argument("--run-name", default=None, help="Optional stable run directory name.")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume behavior.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-dedupe", action="store_true", help="Disable dedupe before FST.")
    parser.add_argument("--dedupe-normalise", default="exact", choices=["exact", "normalised"])
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-chunk-text", action="store_true", help="Disable text chunking.")
    parser.add_argument("--chunk-min-words", type=int, default=2)
    parser.add_argument("--chunk-max-words", type=int, default=70)
    parser.add_argument("--top-n-frames", type=int, default=20)
    parser.add_argument("--top-n-agents", type=int, default=30)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--n-communities", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    payload = run_fst2graph(
        input_csv=Path(args.input),
        out_root=Path(args.out_root),
        text_col=args.text_col,
        id_col=args.id_col,
        doc_col=args.doc_col,
        framebase_index=Path(args.framebase_index) if args.framebase_index else None,
        framebase_dir=Path(args.framebase_dir) if args.framebase_dir else None,
        run_name=args.run_name,
        resume=not args.no_resume,
        batch_size=args.batch_size,
        dedupe=not args.no_dedupe,
        dedupe_normalise=args.dedupe_normalise,
        checkpoint_every=args.checkpoint_every,
        device=args.device,
        chunk_text=not args.no_chunk_text,
        chunk_min_words=args.chunk_min_words,
        chunk_max_words=args.chunk_max_words,
        top_n_frames=args.top_n_frames,
        top_n_agents=args.top_n_agents,
        min_count=args.min_count,
        n_communities=args.n_communities,
        random_seed=args.random_seed,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
