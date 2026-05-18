from __future__ import annotations

import os

import pytest

from fst2framegraph.pipeline_v2.preflight import (
    COLAB_INSTALL_HINT,
    PreflightError,
    apply_runtime_env_guards,
    run_preflight,
    validate_protobuf_version,
)


def test_apply_runtime_env_guards_sets_required_flags() -> None:
    guards = apply_runtime_env_guards()
    for key, value in guards.items():
        assert os.environ.get(key) == value


def test_validate_protobuf_version_rejects_incompatible_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fst2framegraph.pipeline_v2.preflight.protobuf_version", lambda: "4.25.3")
    with pytest.raises(PreflightError) as exc:
        validate_protobuf_version()
    message = str(exc.value)
    assert "Incompatible protobuf version" in message
    assert "frame-semantic-transformer==0.10.0" in message


def test_run_preflight_fails_with_friendly_missing_fst_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(name: str):
        raise ImportError(name)

    monkeypatch.setattr("importlib.import_module", _boom)
    with pytest.raises(PreflightError) as exc:
        run_preflight(fst=None, require_fst=True, apply_env_guards=False)
    message = str(exc.value)
    assert "frame-semantic-transformer is required" in message
    assert "requirements-colab.txt" in message


def test_run_preflight_allows_external_fst_object_without_strict_dependency_check() -> None:
    report = run_preflight(fst=object(), require_fst=True, apply_env_guards=False)
    assert report.ok is True
    assert report.fst_available is True
