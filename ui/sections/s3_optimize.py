"""Section 3 — Optimización de portafolios (design.md section 14, block 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.backtest import TANGENCY_UNDEFINED
from core.formatting import fmt_num, fmt_pct, sign_class, style_table
from ui.charts import PLOTLY_CONFIG, weights_bar
from ui.common import (base_fingerprint_params, ensure_stats, export_button,
                       no_data_box, stale_badge)
from ui.compute import cached_optimization
from ui.labels import COLUMN_LABELS, STRATEGY_LABELS, WARNING_LABELS, label
from ui.state import K_OPTIMIZATION, store_fingerprint
from ui.theme import PALETTE, badge, metric_card, panel, small_note, status_box

__all__ = ['render']

STRATEGY_ORDER = ('min_variance', 'max_sharpe', 'equal_weight')

CONVENTION_NOTE = """
Las rentabilidades son logarítmicas diarias. La anualización multiplica la media
por los días bursátiles del año y la covarianza por el mismo factor:

    mu_anual  = mu_diaria * TDY
    Sigma_anual = Sigma_diaria * TDY

La tasa libre de riesgo se introduce como **tasa anual efectiva** y se convierte a
capitalización continua para el Sharpe:

    rf_cc_anual = ln(1 + rf_anual_efectiva)
    Sharpe      = (w'mu_anual - rf_cc_anual) / sqrt(w' Sigma_anual w)
"""


def _weights_frame(summary, names) -> pd.DataFrame:
    weights = np.asarray(summary.weights, dtype=float).ravel()
    frame = pd.DataFrame({'asset': [str(name) for name in names],
                          'weight': weights})
    return frame.iloc[np.argsort(-np.abs(frame['weight'].to_numpy()))]


def _render_cards(summary, stats, params) -> None:
    fmt = params['fmt']
    cards = st.columns(4)
    with cards[0]:
        metric_card('Rentabilidad anual esperada',
                    fmt_pct(summary.expected_return_annual, 2, fmt),
                    'Rentabilidad log anualizada',
                    sign=sign_class(summary.expected_return_annual))
    with cards[1]:
        metric_card('Volatilidad anual', fmt_pct(summary.volatility_annual, 2, fmt),
                    'Desviación típica anualizada', variant='vol')
    with cards[2]:
        metric_card('Razón de Sharpe', fmt_num(summary.sharpe, 4, fmt),
                    '(Rp - Rf) / sigma_p', variant='sharpe',
                    sign=sign_class(summary.sharpe))
    with cards[3]:
        metric_card('Activos con posición positiva', str(int(summary.n_positive)),
                    f'Días bursátiles alineados: {int(stats.n_obs)}')


def render(params: dict) -> None:
    fingerprint_params = base_fingerprint_params(params)
    aligned, stats = ensure_stats(params, fingerprint_params)
    if aligned is None:
        with panel('Optimización de portafolios'):
            no_data_box()
        return

    with panel('Ejecución'):
        stale_badge('optimization', fingerprint_params)
        if not params['constraints_feasible']:
            status_box('Corrige el límite máximo por activo en la barra lateral '
                       'antes de optimizar.', 'err')
        run = st.button('Ejecutar optimización', type='primary', key='s3_run',
                        disabled=not params['constraints_feasible'])
        if run:
            try:
                optimization = cached_optimization(
                    stats.mu_annual, stats.cov_annual,
                    params['convention'].rf_cc_annual, params['allow_short'],
                    params['max_weight'], params['short_lb'], params['seed'])
            except Exception as error:             # noqa: BLE001
                status_box(f'No se pudo optimizar: {error}', 'err')
            else:
                st.session_state[K_OPTIMIZATION] = optimization
                store_fingerprint('optimization', fingerprint_params)

    optimization = st.session_state.get(K_OPTIMIZATION)
    if optimization is None:
        with panel('Resultados'):
            status_box('Pulsa «Ejecutar optimización» para calcular los tres '
                       'portafolios.', 'info')
        return

    with panel('Estrategia'):
        strategy = st.radio(
            'Estrategia', STRATEGY_ORDER, horizontal=True,
            format_func=lambda key: label(STRATEGY_LABELS, key), key='s3_strategy')
    summary = optimization[strategy]

    with panel('Resultados de la estrategia'):
        _render_cards(summary, stats, params)
        if summary.certified:
            st.markdown(badge('Óptimo verificado', PALETTE['accent']),
                        unsafe_allow_html=True)
        else:
            st.markdown(badge('No certificado — ver diagnóstico', PALETTE['warn']),
                        unsafe_allow_html=True)
            status_box(label(WARNING_LABELS, 'optimization_not_certified'), 'info')
        diagnostics = dict(summary.diagnostics or {})
        message = str(diagnostics.get('message', '') or '')
        restarts = diagnostics.get('n_restarts', 0)
        if message or not summary.certified:
            small_note(f'Mensaje del solver: {message or "sin mensaje"} · '
                       f'Reintentos: {restarts} · '
                       f'Inicio usado: {diagnostics.get("start_used", "n/d")}')
        if message == TANGENCY_UNDEFINED:
            status_box(label(WARNING_LABELS, TANGENCY_UNDEFINED), 'info')

    with panel('Distribución de pesos'):
        left, right = st.columns([1.1, 0.9])
        with left:
            frame = _weights_frame(summary, stats.names)
            frame = frame.rename(columns=lambda key: label(COLUMN_LABELS, key))
            st.dataframe(
                style_table(frame, pct_cols=[label(COLUMN_LABELS, 'weight')],
                            signed_cols=[label(COLUMN_LABELS, 'weight')],
                            fmt=params['fmt']),
                use_container_width=True, hide_index=True)
            small_note('Ordenados por peso absoluto descendente')
        with right:
            st.plotly_chart(weights_bar(summary, stats.names),
                            use_container_width=True, config=PLOTLY_CONFIG)

    with panel('Comparación de estrategias'):
        rows = []
        for key in STRATEGY_ORDER:
            item = optimization[key]
            rows.append({
                'strategy': label(STRATEGY_LABELS, key),
                'expected_return_annual': float(item.expected_return_annual),
                'volatility_annual': float(item.volatility_annual),
                'sharpe': float(item.sharpe),
                'n_positive': int(item.n_positive),
                'certified': bool(item.certified),
            })
        table = pd.DataFrame(rows).rename(columns=lambda key: label(COLUMN_LABELS, key))
        st.dataframe(
            style_table(table,
                        pct_cols=[label(COLUMN_LABELS, key) for key in
                                  ('expected_return_annual', 'volatility_annual')],
                        num_cols=[label(COLUMN_LABELS, 'sharpe')],
                        signed_cols=[label(COLUMN_LABELS, 'expected_return_annual')],
                        fmt=params['fmt']),
            use_container_width=True, hide_index=True)

    with st.expander('Convención de frecuencia'):
        st.markdown(CONVENTION_NOTE)

    with panel('Informe'):
        export_button(params, 's3_export')
