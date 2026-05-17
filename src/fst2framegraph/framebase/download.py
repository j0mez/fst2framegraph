from __future__ import annotations

import datetime as _dt
import gzip
import hashlib
import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

FRAMEBASE_VERSION = "2.0"
FRAMEBASE_LICENSE = "Creative Commons Attribution 4.0 International"
FRAMEBASE_WEBSITE = "https://www.framebase.org/"
FRAMEBASE_DATA_PAGE = "https://www.framebase.org/data"


@dataclass(frozen=True)
class FrameBaseDownload:
    key: str
    file_name: str
    url: str
    required: bool = True
    description: str = ""


FRAMEBASE_DOWNLOADS: tuple[FrameBaseDownload, ...] = (
    FrameBaseDownload(
        key="core_schema",
        file_name="FrameBase_schema_core.ttl.gz",
        url="https://www.framebase.org/files/data/dump/schema/FrameBase_schema_core.ttl.gz",
        description="FrameBase 2.0 core schema: reified frame/FE vocabulary.",
    ),
    FrameBaseDownload(
        key="dbp_labels",
        file_name="FrameBase_schema_dbps.ttl.gz",
        url="https://www.framebase.org/files/data/dump/schema/FrameBase_schema_dbps.ttl.gz",
        description="FrameBase 2.0 labels for direct binary predicates.",
    ),
    FrameBaseDownload(
        key="dereification_rules_spin",
        file_name="dereificationRulesSpinFormat.ttl.gz",
        url="https://www.framebase.org/files/data/dump/schema/dereificationRulesSpinFormat.ttl.gz",
        description="FrameBase 2.0 dereification rules as SPIN/Turtle.",
    ),
)


class DownloadError(RuntimeError):
    pass


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_url(url: str, path: Path, overwrite: bool = False, timeout: int = 60) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "fst2framegraph/0.2"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out)
        tmp.replace(path)
    except Exception as exc:  # pragma: no cover - network dependent
        if tmp.exists():
            tmp.unlink()
        raise DownloadError(f"Could not download {url}: {exc}") from exc


def _basic_validate_file(path: Path) -> dict[str, object]:
    info: dict[str, object] = {"exists": path.exists(), "valid_container": False, "size_bytes": 0}
    if not path.exists():
        return info
    info["size_bytes"] = path.stat().st_size
    if path.suffix == ".gz":
        try:
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                f.read(1024)
            info["valid_container"] = True
        except Exception as exc:
            info["error"] = str(exc)
    elif path.suffix == ".zip":
        try:
            with zipfile.ZipFile(path) as zf:
                bad = zf.testzip()
                info["valid_container"] = bad is None
                info["zip_members"] = len(zf.namelist())
                if bad is not None:
                    info["error"] = f"Corrupt zip member: {bad}"
        except Exception as exc:
            info["error"] = str(exc)
    else:
        info["valid_container"] = True
    return info


def write_framebase_manifest(out_dir: Path, downloads: Iterable[FrameBaseDownload] = FRAMEBASE_DOWNLOADS) -> dict:
    found = find_framebase_files(out_dir)
    files = []
    for item in downloads:
        path = found.get(item.key) or (out_dir / item.file_name)
        validation = _basic_validate_file(path)
        files.append(
            {
                "key": item.key,
                "name": path.name,
                "path": str(path),
                "url": item.url,
                "description": item.description,
                "required": item.required,
                "sha256": sha256_file(path) if path.exists() else None,
                **validation,
            }
        )
    manifest = {
        "source": "FrameBase",
        "version": FRAMEBASE_VERSION,
        "license": FRAMEBASE_LICENSE,
        "attribution": "FrameBase team at Aalborg University and Rutgers University",
        "website": FRAMEBASE_WEBSITE,
        "data_page": FRAMEBASE_DATA_PAGE,
        "downloaded_or_validated_at_utc": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        "files": files,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "framebase_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def download_framebase_files(out_dir: Path, overwrite: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in FRAMEBASE_DOWNLOADS:
        _download_url(item.url, out_dir / item.file_name, overwrite=overwrite)
    return write_framebase_manifest(out_dir)


def default_framebase_dir() -> Path:
    return Path("data") / "framebase"


def find_framebase_files(framebase_dir: Path | None = None) -> dict[str, Path | None]:
    base = framebase_dir or default_framebase_dir()
    candidates = {
        "core_schema": ["FrameBase_schema_core.ttl.gz", "FrameBase_schema_core.ttl"],
        "dbp_labels": ["FrameBase_schema_dbps.ttl.gz", "FrameBase_schema_dbps.ttl"],
        "dereification_rules_spin": [
            "dereificationRulesSpinFormat.ttl.gz",
            "dereificationRulesSpinFormat.ttl",
        ],
        "dereification_rules_sparql": [
            "dereificationRulesSparqlFormat.txt.zip",
            "dereificationRulesSparqlFormat.zip",
            "dereificationRulesSparqlFormat.txt",
        ],
        "clusters": ["clusters.txt"],
        "cluster_pairs": ["clusterPairs.txt"],
        "lexical_clusters": ["lexicalClusters.txt"],
        "manual_schema_extensions": ["manual/FrameBase_schema_manual_extensions.ttl"],
        "manual_inference_rules": ["manual/inferenceRulesForSchema.txt"],
    }
    found: dict[str, Path | None] = {}
    for key, names in candidates.items():
        found[key] = None
        for name in names:
            p = base / name
            if p.exists():
                found[key] = p
                break
    return found
