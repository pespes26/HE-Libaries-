"""Headless smoke tests for the Streamlit app (streamlit.testing AppTest).

These execute the whole app script in-process (no browser) and exercise the live-demo
and use-case buttons, so a broken Streamlit call or HE pipeline fails CI.
"""
from __future__ import annotations

from streamlit.testing.v1 import AppTest

APP = "app/streamlit_app.py"


def _run():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at


def test_app_loads_without_exception():
    at = _run()
    assert not at.exception, at.exception


def test_live_demo_button_runs():
    at = _run()
    labels = [b.label for b in at.button]
    assert "Run encrypted computation" in labels
    for b in at.button:
        if b.label == "Run encrypted computation":
            b.click()
    at.run()
    assert not at.exception, at.exception


def test_use_case_button_runs():
    at = _run()
    matches = [b for b in at.button if b.label.startswith("Encrypt and compute the average")]
    assert matches, "use-case button not found"
    matches[0].click()
    at.run()
    assert not at.exception, at.exception
    # The HE pipeline must actually have produced output, not silently no-op.
    texts = [m.value for m in at.success] + [m.value for m in at.markdown]
    assert any("average" in t.lower() for t in texts), \
        "use-case run produced no average output"
