"""Smoke test of the whole application through ``streamlit.testing.v1.AppTest``.

A GET to ``/`` only returns the static shell, so this is the only check that
actually executes the script body and every section renderer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest

from ui.labels import SECTIONS
from ui.state import K_ACTIVE_SECTION

APP_PATH = str(Path(__file__).resolve().parents[1] / 'app.py')
TIMEOUT = 120


def _run() -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    app.run()
    return app


def test_app_boots_without_exception() -> None:
    app = _run()
    assert not app.exception


@pytest.mark.parametrize('section', list(SECTIONS))
def test_every_section_renders(section: str) -> None:
    app = _run()
    app.session_state[K_ACTIVE_SECTION] = section
    app.run()
    assert not app.exception


def test_sample_data_flows_through_the_pipeline() -> None:
    app = _run()
    app.button(key='s1_sample').click().run()
    app.button(key='s1_load').click().run()
    assert not app.exception
    assert app.session_state['aligned_data'] is not None

    app.session_state[K_ACTIVE_SECTION] = SECTIONS[2]
    app.run()
    app.button(key='s3_run').click().run()
    assert not app.exception
    assert app.session_state['optimization'] is not None

    # Section 5 computes on render, with no button of its own.
    app.session_state[K_ACTIVE_SECTION] = SECTIONS[4]
    app.run()
    assert not app.exception
    assert app.session_state['purchase_balance'] is not None

    # Section 6 with the annual rule keeps the walk-forward run short.
    app.session_state[K_ACTIVE_SECTION] = SECTIONS[5]
    app.run()
    app.session_state['rebalance_rule'] = 'A'
    app.run()
    app.button(key='s6_run').click().run()
    assert not app.exception
    assert app.session_state['backtest_result'] is not None

    # Section 7: the priority section, down to the verdict.
    app.session_state[K_ACTIVE_SECTION] = SECTIONS[6]
    app.run()
    app.session_state['n_paths'] = 200
    app.session_state['s7_n_installments'] = 6
    app.run()
    app.button(key='s7_run').click().run()
    assert not app.exception
    assert app.session_state['capacity_metrics'] is not None
    assert app.session_state['capacity_verdict'] is not None
