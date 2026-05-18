from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WHEEL_DIR = ROOT / "wheels"
MANIFEST = WHEEL_DIR / "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    WHEEL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for item in manifest["wheels"]:
        path = WHEEL_DIR / item["filename"]
        if not path.exists():
            print(f"Downloading {item['filename']}")
            urllib.request.urlretrieve(item["url"], path)
        if path.stat().st_size != int(item["size"]):
            raise RuntimeError(f"Unexpected size for {path}: {path.stat().st_size}")
        digest = _sha256(path)
        if digest != item["sha256"]:
            raise RuntimeError(f"Unexpected sha256 for {path}: {digest}")
        print(f"Verified {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
