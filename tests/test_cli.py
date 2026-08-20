"""Smoke tests for the CLI entry point."""

from ccr.cli import main


def test_cli_runs_sample_scenario(capsys):
    rc = main(["examples/sample_scenario.py"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "CVA" in out
    assert "Max PFE" in out


def test_cli_json_output(capsys):
    import json

    rc = main(["examples/sample_scenario.py", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert payload["cva"] > 0
    assert payload["n_paths"] > 0
