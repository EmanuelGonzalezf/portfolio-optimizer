"""Section 2 — Análisis de activos (design.md section 14, block 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.formatting import fmt_num, fmt_pct, style_table
from ui.charts import (PLOTLY_CONFIG, correlation_heatmap, covariance_heatmap,
                       normalized_prices, returns_distribution)
from ui.common import base_fingerprint_params, ensure_stats, no_data_box
from ui.labels import COLUMN_LABELS, WARNING_LABELS, label
from ui.theme import metric_card, panel, small_note, status_box

__all__ = ['render']

TABLE_COLUMNS = ('asset', 'n_obs', 'annual_return', 'annual_volatility', 'sharpe',
                 'min_return', 'max_return')


def _stats_frame(stats) -> pd.DataFrame:
    """The prototype table, with the raw figures already produced by core/stats."""
    return pd.DataFrame({
        'asset': [str(name) for name in stats.names],
        'n_obs': [int(stats.n_obs)] * len(stats.names),
        'annual_return': np.asarray(stats.mu_annual, dtype=float),
        'annual_volatility': np.asarray(stats.sigma_annual, dtype=float),
        'sharpe': np.asarray(stats.sharpe_individual, dtype=float),
        'min_return': np.asarray(stats.min_return, dtype=float),
        'max_return': np.asarray(stats.max_return, dtype=float),
    }, columns=list(TABLE_COLUMNS))


def _mean_off_diagonal(corr: np.ndarray) -> float:
    matrix = np.asarray(corr, dtype=float)
    size = matrix.shape[0]
    if size < 2:
        return float('nan')
    mask = ~np.eye(size, dtype=bool)
    return float(matrix[mask].mean())


def render(params: dict) -> None:
    fingerprint_params = base_fingerprint_params(params)
    aligned, stats = ensure_stats(params, fingerprint_params)
    if aligned is None:
        with panel('Análisis de activos'):
            no_data_box()
        return

    fmt = params['fmt']
    with panel('Resumen del universo'):
        cards = st.columns(4)
        with cards[0]:
            metric_card('Activos', str(len(aligned.names)))
        with cards[1]:
            metric_card('Observaciones alineadas',
                        fmt_num(int(stats.n_obs), 0, fmt),
                        'Retornos logarítmicos diarios')
        with cards[2]:
            metric_card('Rango', f'{aligned.dates[0]:%d/%m/%Y} – '
                                 f'{aligned.dates[-1]:%d/%m/%Y}')
        with cards[3]:
            mean_corr = _mean_off_diagonal(stats.corr)
            metric_card(label(COLUMN_LABELS, 'mean_correlation'),
                        fmt_num(mean_corr, 4, fmt),
                        'Media fuera de la diagonal')
        if stats.psd_repaired:
            status_box(label(WARNING_LABELS, 'psd_repaired'), 'info')

    with panel('Estadísticas individuales'):
        frame = _stats_frame(stats)
        frame = frame.rename(columns=lambda key: label(COLUMN_LABELS, key))
        st.dataframe(
            style_table(frame,
                        pct_cols=[label(COLUMN_LABELS, key) for key in
                                  ('annual_return', 'annual_volatility',
                                   'min_return', 'max_return')],
                        num_cols=[label(COLUMN_LABELS, 'sharpe')],
                        signed_cols=[label(COLUMN_LABELS, key) for key in
                                     ('annual_return', 'sharpe')],
                        fmt=fmt),
            use_container_width=True, hide_index=True)
        small_note('Rentabilidad y volatilidad anualizadas con '
                   f'{params["trading_days"]} días bursátiles; el Sharpe individual '
                   f'usa la tasa libre de riesgo de la barra lateral '
                   f'({fmt_pct(params["rf"], 2, fmt)} anual efectiva).')

    with panel('Series y distribución'):
        left, right = st.columns(2)
        with left:
            st.plotly_chart(normalized_prices(aligned), use_container_width=True,
                            config=PLOTLY_CONFIG)
        with right:
            st.plotly_chart(returns_distribution(aligned), use_container_width=True,
                            config=PLOTLY_CONFIG)

    with panel('Matrices'):
        corr_tab, cov_tab = st.tabs(['Correlación', 'Covarianza'])
        with corr_tab:
            st.plotly_chart(correlation_heatmap(stats.corr, stats.names),
                            use_container_width=True, config=PLOTLY_CONFIG)
        with cov_tab:
            st.plotly_chart(covariance_heatmap(stats.cov_annual, stats.names),
                            use_container_width=True, config=PLOTLY_CONFIG)
