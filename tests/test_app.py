"""Headless smoke test of the Streamlit interface."""
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def test_app_enrol_recognise_and_evaluate_flow():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    next(b for b in at.button if "demo vocabulary" in b.label).click().run()
    assert not at.exception and any("Enrolled 36" in s.value for s in at.success)
    at.radio[0].set_value("Demo test clip").run()
    next(b for b in at.button if b.label == "Recognise").click().run()
    assert not at.exception
    assert any("Recognised word" in s.value for s in at.success)
    next(b for b in at.button if b.label == "Run evaluation").click().run()
    assert not at.exception and not at.error
