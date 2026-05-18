from __future__ import annotations

import importlib
import importlib.metadata
import os
import re
import sys
from dataclasses import dataclass
from typing import Any


COLAB_ENV_GUARDS: dict[str, str] = {
    "USE_TF": "0",
    "TRANSFORMERS_NO_TF": "1",
    "USE_FLAX": "0",
    "TOKENIZERS_PARALLELISM": "false",
}

COLAB_INSTALL_HINT = "\n".join(
    [
        "pip install --find-links=wheels/ -e .",
        "python scripts/install_colab_fst.py",
        "or explicitly: pip install --force-reinstall wheels/sentencepiece-0.2.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl  # sentencepiece==0.2.0",
        "then: pip install --no-deps frame-semantic-transformer==0.10.0",
    ]
)


class PreflightError(RuntimeError):
    """Raised when runtime preflight requirements are not met."""


@dataclass(frozen=True)
class PreflightReport:
    ok: bool
    python_version: str
    in_colab: bool
    fst_available: bool
    protobuf_version: str | None
    env_guards_applied: dict[str, str]
    warnings: list[str]


def apply_runtime_env_guards() -> dict[str, str]:
    for key, value in COLAB_ENV_GUARDS.items():
        os.environ[key] = value
    return dict(COLAB_ENV_GUARDS)


def detect_colab() -> bool:
    if "google.colab" in sys.modules:
        return True
    return "COLAB_GPU" in os.environ


def _parse_version_tuple(version: str) -> tuple[int, ...]:
    numbers = [int(part) for part in re.findall(r"\d+", version)]
    return tuple(numbers or [0])


def _version_in_range(version: str, minimum: str, maximum_exclusive: str) -> bool:
    parsed = _parse_version_tuple(version)
    return _parse_version_tuple(minimum) <= parsed < _parse_version_tuple(maximum_exclusive)


def protobuf_version() -> str | None:
    try:
        return importlib.metadata.version("protobuf")
    except importlib.metadata.PackageNotFoundError:
        return None


def validate_protobuf_version(
    *,
    minimum: str = "3.20.1",
    maximum_exclusive: str = "4.0.0",
) -> str:
    version = protobuf_version()
    if version is None:
        raise PreflightError(
            "protobuf is not installed.\n"
            "Install the known-good FST stack with:\n"
            f"{COLAB_INSTALL_HINT}"
        )
    if not _version_in_range(version, minimum, maximum_exclusive):
        raise PreflightError(
            f"Incompatible protobuf version {version!r}. "
            f"Expected >= {minimum} and < {maximum_exclusive} for FrameSemanticTransformer 0.10.0.\n"
            "Install the known-good FST stack with:\n"
            f"{COLAB_INSTALL_HINT}"
        )
    return version


def ensure_fst_dependency(*, fst: Any | None = None) -> None:
    if fst is not None:
        return
    try:
        importlib.import_module("frame_semantic_transformer")
    except Exception as exc:
        raise PreflightError(
            "frame-semantic-transformer is required for automatic FST inference.\n"
            "Install the known-good stack with:\n"
            f"{COLAB_INSTALL_HINT}"
        ) from exc


def run_preflight(
    *,
    fst: Any | None = None,
    apply_env_guards: bool = True,
    require_fst: bool = True,
) -> PreflightReport:
    warnings: list[str] = []
    guards = apply_runtime_env_guards() if apply_env_guards else {}
    colab = detect_colab()

    proto_version: str | None = None
    fst_available = False
    if require_fst:
        ensure_fst_dependency(fst=fst)
        if fst is None:
            proto_version = validate_protobuf_version()
        else:
            proto_version = protobuf_version()
        fst_available = True
    else:
        try:
            ensure_fst_dependency(fst=fst)
            fst_available = True
            proto_version = protobuf_version()
        except PreflightError as exc:
            warnings.append(str(exc))

    if sys.version_info >= (3, 12):
        warnings.append(
            "Python 3.12 detected. If FST import/runtime issues appear, reinstall the pinned "
            "Colab stack with scripts/install_colab_fst.py before rerunning."
        )

    return PreflightReport(
        ok=True,
        python_version=".".join(str(x) for x in sys.version_info[:3]),
        in_colab=colab,
        fst_available=fst_available,
        protobuf_version=proto_version,
        env_guards_applied=guards,
        warnings=warnings,
    )
