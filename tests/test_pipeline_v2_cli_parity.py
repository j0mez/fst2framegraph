from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import run_pipeline
from fst2framegraph.cli import app


def _fake_run_fst2graph(**kwargs):
    out_root = Path(kwargs["out_root"])
    run_root = out_root / "run_stub"
    run_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": "stub",
        "input_csv": str(kwargs["input_csv"]),
        "input_rows": 2,
        "chunk_rows": 2,
        "run_root": str(run_root),
        "run_dir": str(run_root / "fst_clean"),
        "graph_out_dir": str(run_root / "graph"),
        "analysis_out_dir": str(run_root / "analysis"),
        "preflight": {"ok": True},
        "extraction_report": {"frame_instances": 1, "frame_elements": 2},
        "graph_report": {"nodes": 3, "edges": 2},
        "analysis_report": {"lift_rows": 1},
    }
    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["summary_path"] = str(summary_path)
    return payload


def test_pipeline_cli_and_script_wrap_same_one_call_contract(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("id,text\n1,hello world\n2,goodbye world\n", encoding="utf-8")

    monkeypatch.setattr(run_pipeline, "run_fst2graph", _fake_run_fst2graph)
    rc = run_pipeline.main(
        [
            "--input",
            str(csv_path),
            "--out-root",
            str(tmp_path / "script_out"),
        ]
    )
    assert rc == 0
    script_payload = json.loads(capsys.readouterr().out)

    import fst2framegraph.cli as cli_module

    monkeypatch.setattr(cli_module, "run_fst2graph", _fake_run_fst2graph)
    result = CliRunner().invoke(
        app,
        [
            "pipeline",
            "--input",
            str(csv_path),
            "--out-root",
            str(tmp_path / "cli_out"),
        ],
    )
    assert result.exit_code == 0, result.output
    cli_payload = json.loads(result.output)

    for payload in [script_payload, cli_payload]:
        assert payload["run_id"] == "stub"
        assert payload["input_rows"] == 2
        assert payload["chunk_rows"] == 2
        assert payload["preflight"]["ok"] is True
        assert Path(payload["summary_path"]).exists()
