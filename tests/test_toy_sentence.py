from pathlib import Path

from typer.testing import CliRunner

from fst2framegraph.cli import app


def test_toy_build(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "build",
            "--input",
            "examples/toy_fst_output.csv",
            "--out",
            str(tmp_path / "out"),
            "--no-rdf",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "frame_instances.csv").exists()
    assert (tmp_path / "out" / "graph_edges_reified.csv").exists()
    assert (tmp_path / "out" / "sentence_graphs.jsonl").exists()
