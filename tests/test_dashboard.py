"""Smoke test: the Streamlit dashboard script runs without raising."""

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "src" / "ccr" / "dashboard" / "app.py"


def test_dashboard_runs_clean():
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "CVA" in labels
    assert "EPE" in labels
