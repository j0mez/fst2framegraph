#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fst2framegraph.framebase.download import download_framebase_files, write_framebase_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the FrameBase files used by fst2framegraph.")
    parser.add_argument("--out", type=Path, default=Path("data/framebase"), help="Output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Re-download even if files already exist.")
    parser.add_argument("--manifest-only", action="store_true", help="Only write checksums/manifest for existing files.")
    args = parser.parse_args()

    if args.manifest_only:
        manifest = write_framebase_manifest(args.out)
    else:
        manifest = download_framebase_files(args.out, overwrite=args.overwrite)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
