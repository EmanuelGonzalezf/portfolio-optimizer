"""Section 6 — Backtesting (design.md section 14, block 6)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.backtest import BacktestConfig, REBALANCE_RULES, STRATEGIES
from ui.charts import (CYCLE, PLOTLY_CONFIG, STRATEGY_COLORS, base_layout,
                       cumulative_return_chart, drawdown_chart, equity_curves,
                       weights_area)
from ui.common import (base_fingerprint_params, ensure_stats, local_badge,
                       no_data_box, stale_badge)
from ui.compute import cached_backtest
from ui.labels import (COLUMN_LABELS, REBALANCE_LABELS, STRATEGY_LABELS,
                       WARNING_LABELS, label)
from ui.state import (K_BACKTEST_CONFIG, K_BACKTEST_RESULT, effective_value,
                      store_fingerprint)
from ui.theme import panel, small_note, status_box

__all__ = ['render']

METRIC_ROWS = ('total_return', 'cagr', 'mu_cc_real', 'vol_real', 'sharpe_real',
               'max_drawdown', 'final_capital', 'total_costs', 'n_rebalances',
               'n_observations')
PCT_METRICS = ('total_return', 'cagr', 'mu_cc_real', 'vol_real', 'max_drawdown')
MONEY_METRICS = ('final_capital', 'total_costs')

GLOBAL_NOTE = ('El backtesting reestima los momentos en cada rebalanceo con el '
               'periodo, la tasa libre de riesgo y las restricciones de la barra '
               'lateral.')


def _asset_comparison(result, names) -> go.Figure:
    """The individual assets of the equity frame, already computed by core."""
    figure = go.Figure()
    equity = result.equity
    for position, name in enumerate(names):
        if name not in equity.columns:
            continue
        figure.add_trace(go.Scatter(
            x=equity.index, y=equity[name].to_numpy(dtype=float), mode='lines',
            name=str(name), line={'width': 1.6,
                                  'color': CYCLE[position % len(CYCLE)]}))
    for strategy in STRATEGIES:
        if strategy not in equity.columns:
            continue
        figure.add_trace(go.Scatter(
            x=equity.index, y=equity[strategy].to_numpy(dtype=float), mode='lines',
            name=label(STRATEGY_LABELS, strategy),
            line={'width': 2.6, 'color': STRATEGY_COLORS.get(strategy)}))
    figure.update_layout(**base_layout(yaxis={'title': 'Capital'}))
    return figure


def render(params: dict) -> None:
    fingerprint_params = base_fingerprint_params(params)
    aligned, stats = ensure_stats(params, fingerprint_params)
    if aligned is None:
        with panel('Backtesting'):
            no_data_box()
        return

    total_returns = int(len(aligned.return_dates))
    max_window = max(min(2_520, total_returns - 1), 30)

    with panel('Configuración del backtesting'):
        left, right = st.columns(2)
        with left:
            window = int(st.number_input(
                'Ventana de estimación (observaciones)', min_value=30,
                max_value=max_window, value=min(252, max_window), step=10,
                key='window'))
            rules = list(REBALANCE_RULES)
            rebalance_rule = st.selectbox(
                'Regla de rebalanceo', rules,
                index=rules.index('M') if 'M' in rules else 0,
                format_func=lambda key: label(REBALANCE_LABELS, key),
                key='rebalance_rule')
        with right:
            capital, capital_local = effective_value(params, 'capital', 'capital_s6')
            capital = float(st.number_input(
                'Capital inicial', min_value=0.01, value=float(capital),
                step=1_000.0, format='%.2f', key='capital_s6'))
            local_badge(capital_local)
            cost_rate = float(st.number_input(
                'Costo por turnover (fracción)', min_value=0.0, max_value=0.1,
                value=0.001, step=0.0005, format='%.4f', key='cost_rate'))
        small_note(GLOBAL_NOTE)

        fingerprint_params.update({
            'window': window, 'rebalance_rule': rebalance_rule,
            'capital_s6': capital, 'cost_rate': cost_rate,
        })
        stale_badge('backtest_result', fingerprint_params)
        if st.button('Ejecutar backtesting', type='primary', key='s6_run',
                     disabled=not params['constraints_feasible']):
            config = BacktestConfig(window=window, rebalance_rule=str(rebalance_rule),
                                    initial_capital=capital, cost_rate=cost_rate,
                                    strategies=STRATEGIES)
            try:
                result = cached_backtest(
                    aligned.log_returns, aligned.return_dates.asi8,
                    tuple(aligned.names), config.to_tuple(),
                    params['convention'].to_tuple(),
                    params['constraints'].to_tuple())
            except Exception as error:             # noqa: BLE001
                status_box(f'No se pudo ejecutar el backtesting: {error}', 'err')
            else:
                st.session_state[K_BACKTEST_CONFIG] = config
                st.session_state[K_BACKTEST_RESULT] = result
                store_fingerprint('backtest_result', fingerprint_params)

    result = st.session_state.get(K_BACKTEST_RESULT)
    if result is None:
        with panel('Resultados'):
            status_box('Pulsa «Ejecutar backtesting» para recorrer la serie '
                       'reestimando en cada rebalanceo.', 'info')
        return

    with panel('Métricas por estrategia'):
        metrics = result.metrics.copy()
        available = [row for row in METRIC_ROWS if row in metrics.columns]
        table = metrics[available].transpose()
        table.index = [label(COLUMN_LABELS, key) for key in table.index]
        table = table.rename(columns=lambda key: label(STRATEGY_LABELS, key))
        st.dataframe(table.style.format('{:,.4f}'), use_container_width=True)
        small_note('Rentabilidades y drawdown en fracción; importes en la moneda '
                   'configurada.')

    with panel('Evolución del capital'):
        left, right = st.columns(2)
        with left:
            st.plotly_chart(equity_curves(result), use_container_width=True,
                            config=PLOTLY_CONFIG)
        with right:
            st.plotly_chart(cumulative_return_chart(result),
                            use_container_width=True, config=PLOTLY_CONFIG)

    with panel('Riesgo y comparación'):
        left, right = st.columns(2)
        with left:
            st.plotly_chart(drawdown_chart(result), use_container_width=True,
                            config=PLOTLY_CONFIG)
        with right:
            st.plotly_chart(_asset_comparison(result, aligned.names),
                            use_container_width=True, config=PLOTLY_CONFIG)
            small_note('Estrategias frente a la compra y mantenimiento de cada '
                       'activo individual.')

    with panel('Trayectoria de los pesos'):
        strategies = [key for key in STRATEGIES if key in result.weights_history]
        strategy = st.radio(
            'Estrategia', strategies, horizontal=True,
            format_func=lambda key: label(STRATEGY_LABELS, key),
            key='s6_weights_strategy')
        st.plotly_chart(weights_area(result.weights_history[strategy]),
                        use_container_width=True, config=PLOTLY_CONFIG)
        small_note(f'{len(result.rebalance_dates)} rebalanceos entre '
                   f'{result.rebalance_dates[0]:%d/%m/%Y} y '
                   f'{result.rebalance_dates[-1]:%d/%m/%Y}')

    with panel('Eventos del recorrido'):
        if not result.events:
            status_box('Todos los rebalanceos contaron con un óptimo verificado.',
                       'ok')
        else:
            counts: dict = {}
            for event in result.events:
                code = str(event).split(':')[0]
                counts[code] = counts.get(code, 0) + 1
            for code, times in counts.items():
                status_box(f'{label(WARNING_LABELS, code)} ({times} veces)', 'info')
            with st.expander('Detalle de los eventos'):
                rows = []
                for event in result.events:
                    parts = str(event).split(':')
                    rows.append({'Evento': label(WARNING_LABELS, parts[0]),
                                 'Estrategia': label(STRATEGY_LABELS,
                                                     parts[1] if len(parts) > 1
                                                     else ''),
                                 'Fecha': parts[2] if len(parts) > 2 else ''})
                st.dataframe(pd.DataFrame(rows), use_container_width=True,
                             hide_index=True)
