"""Portfolio Optimizer — Streamlit entry point.

Routing uses a vertical ``st.radio`` and not ``st.tabs`` for the reason stated
in design.md section 13.3: ``st.tabs`` executes the body of every tab on every
rerun, so the backtest and the Monte Carlo sections would repaint on each
interaction. With ``st.radio`` only the active section runs.

Everything happens inside ``main()``, called under the ``__main__`` guard.
Streamlit runs the script with ``__name__ == '__main__'``, so the guard does
not get in the way and ``import app`` stays free of side effects.
"""

from __future__ import annotations

import logging
import traceback

import streamlit as st

from ui.labels import GENERIC_ERROR, SECTIONS
from ui.sections import (s1_data, s2_assets, s3_optimize, s4_frontier, s5_purchase,
                         s6_backtest, s7_capacity)
from ui.sidebar import render_sidebar
from ui.state import K_ACTIVE_SECTION, init_state
from ui.theme import inject_theme, render_header, section_header, status_box

LOGGER_NAME = 'portfolio_optimizer'

SECTION_RENDERERS: dict = {
    SECTIONS[0]: s1_data.render,
    SECTIONS[1]: s2_assets.render,
    SECTIONS[2]: s3_optimize.render,
    SECTIONS[3]: s4_frontier.render,
    SECTIONS[4]: s5_purchase.render,
    SECTIONS[5]: s6_backtest.render,
    SECTIONS[6]: s7_capacity.render,
}


def configure_logging() -> logging.Logger:
    """``INFO`` to the console, once per process. No ``print`` anywhere."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s %(name)s %(message)s'))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def render_section(name: str, params: dict, logger: logging.Logger) -> None:
    """Render one section, never letting an exception reach the user (RNF-5)."""
    renderer = SECTION_RENDERERS.get(name)
    if renderer is None:
        status_box(GENERIC_ERROR, 'err')
        return
    try:
        renderer(params)
    except Exception:                              # noqa: BLE001 - RNF-5
        logger.error('section render failed: %s', name, exc_info=True)
        status_box(GENERIC_ERROR, 'err')
        with st.expander('Detalle técnico'):
            st.code(traceback.format_exc())


def main() -> None:
    st.set_page_config(page_title='Portfolio Optimizer', layout='wide',
                       page_icon='◉')
    logger = configure_logging()
    inject_theme()
    render_header('Optimización, backtesting y capacidad de pago de cuotas')
    init_state()

    params = render_sidebar()

    with st.sidebar:
        section_header('Secciones')
        section = st.radio('Secciones', SECTIONS, key=K_ACTIVE_SECTION,
                           label_visibility='collapsed')

    logger.info('rendering section: %s', section)
    render_section(section, params, logger)


if __name__ == '__main__':
    main()
