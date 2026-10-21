"""Section 5 — Balance de compra (design.md section 14, block 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.formatting import fmt_currency, fmt_pct, style_table
from core.purchase import PurchaseConfig, build_purchase_balance
from ui.common import (base_fingerprint_params, ensure_stats, local_badge,
                       no_data_box, no_optimization_box, stale_badge)
from ui.labels import COLUMN_LABELS, NOTE_LABELS, STRATEGY_LABELS, label
from ui.state import (K_CURRENT_POSITIONS, K_OPTIMIZATION, K_PURCHASE_BALANCE,
                      K_PURCHASE_CONFIG, effective_value, store_fingerprint)
from ui.theme import metric_card, panel, small_note, status_box

__all__ = ['render']

STRATEGY_ORDER = ('min_variance', 'max_sharpe', 'equal_weight')

POSITIONS_NOTE = ('Sin posiciones actuales el balance asigna únicamente el capital '
                  'nuevo: todas las operaciones son compras.')
ADVICE_NOTE = ('La conveniencia de cada activo se evalúa dentro del portafolio, no de '
               'forma aislada: los pesos salen de la optimización conjunta y una '
               'posición pequeña puede estar cumpliendo una función de '
               'diversificación.')


def _current_positions(names: tuple) -> np.ndarray | None:
    """Optional editor of the units already held. ``None`` when all are zero."""
    stored = st.session_state.get(K_CURRENT_POSITIONS)
    base = pd.DataFrame({'asset': [str(name) for name in names],
                         'current_units': np.zeros(len(names), dtype=float)})
    if stored is not None and np.asarray(stored).size == len(names):
        base['current_units'] = np.asarray(stored, dtype=float)
    frame = base.rename(columns=lambda key: label(COLUMN_LABELS, key))
    edited = st.data_editor(frame, use_container_width=True, hide_index=True,
                            disabled=[label(COLUMN_LABELS, 'asset')],
                            key='s5_positions_editor')
    units = np.asarray(edited[label(COLUMN_LABELS, 'current_units')].to_numpy(),
                       dtype=float)
    units = np.nan_to_num(units, nan=0.0)
    if np.all(units <= 0.0):
        st.session_state[K_CURRENT_POSITIONS] = None
        small_note(POSITIONS_NOTE)
        return None
    st.session_state[K_CURRENT_POSITIONS] = units
    return units


def _render_rows(balance, fmt) -> None:
    rows = balance.rows.copy()
    rows['action'] = [label(COLUMN_LABELS, value) for value in rows['action']]
    rows['note_code'] = [label(NOTE_LABELS, value) if value else ''
                         for value in rows['note_code']]
    rows = rows.rename(columns=lambda key: label(COLUMN_LABELS, key))
    money = [label(COLUMN_LABELS, key) for key in
             ('target_value', 'price', 'gross_amount', 'costs', 'final_value')]
    percent = [label(COLUMN_LABELS, key) for key in
               ('target_weight', 'current_weight', 'final_weight', 'deviation')]
    units = [label(COLUMN_LABELS, key) for key in
             ('current_units', 'target_units', 'delta_units')]
    st.dataframe(
        style_table(rows, money_cols=money, pct_cols=percent, num_cols=units,
                    signed_cols=[label(COLUMN_LABELS, 'delta_units')], fmt=fmt),
        use_container_width=True, hide_index=True)


def render(params: dict) -> None:
    fingerprint_params = base_fingerprint_params(params)
    aligned, stats = ensure_stats(params, fingerprint_params)
    if aligned is None:
        with panel('Balance de compra'):
            no_data_box()
        return
    optimization = st.session_state.get(K_OPTIMIZATION)
    if optimization is None:
        with panel('Balance de compra'):
            no_optimization_box()
        return

    fmt = params['fmt']
    with panel('Configuración de la compra'):
        left, right = st.columns(2)
        with left:
            capital, capital_local = effective_value(params, 'capital', 'capital_s5')
            capital = float(st.number_input(
                'Capital disponible (dinero nuevo)', min_value=0.01,
                value=float(capital), step=1_000.0, format='%.2f', key='capital_s5'))
            local_badge(capital_local)
            strategy = st.radio(
                'Estrategia', STRATEGY_ORDER, horizontal=True,
                format_func=lambda key: label(STRATEGY_LABELS, key),
                key='s5_strategy')
            integer_units = st.toggle('Comprar unidades enteras', value=True,
                                      key='integer_units')
        with right:
            pct_fee, fee_local = effective_value(params, 'pct_fee', 'pct_fee_s5')
            pct_fee = float(st.number_input(
                'Comisión porcentual (%)', min_value=0.0, max_value=10.0,
                value=float(pct_fee) * 100.0, step=0.1, format='%.3f',
                key='s5_pct_fee_percent')) / 100.0
            st.session_state['pct_fee_s5'] = pct_fee
            local_badge(fee_local or pct_fee != params['pct_fee'])
            fixed_fee, fixed_local = effective_value(params, 'fixed_fee',
                                                     'fixed_fee_s5')
            fixed_fee = float(st.number_input(
                'Comisión fija por operación', min_value=0.0,
                value=float(fixed_fee), step=1.0, format='%.2f', key='fixed_fee_s5'))
            local_badge(fixed_local)
            cash_reserve = float(st.number_input(
                'Reserva de efectivo', min_value=0.0, value=0.0, step=100.0,
                format='%.2f', key='cash_reserve'))
            weight_tolerance = float(st.number_input(
                'Tolerancia de peso', min_value=0.0, max_value=0.5, value=0.01,
                step=0.005, format='%.3f', key='weight_tolerance'))

    with panel('Posiciones actuales (opcional)'):
        current_positions = _current_positions(stats.names)

    fingerprint_params.update({
        'capital_s5': capital, 'pct_fee_s5': pct_fee, 'fixed_fee_s5': fixed_fee,
        'cash_reserve': cash_reserve, 'integer_units': integer_units,
        'weight_tolerance': weight_tolerance, 'current_positions': current_positions,
    })

    summary = optimization[strategy]
    prices = np.asarray(aligned.prices, dtype=float)[-1, :]
    config = PurchaseConfig(capital=capital, cash_reserve=cash_reserve,
                            pct_fee=pct_fee, fixed_fee=fixed_fee,
                            integer_units=bool(integer_units),
                            weight_tolerance=weight_tolerance)
    try:
        balance = build_purchase_balance(summary.weights, prices, tuple(stats.names),
                                         config, current_positions)
    except Exception as error:                     # noqa: BLE001
        with panel('Operaciones'):
            status_box(f'No se pudo construir el balance: {error}', 'err')
        return

    st.session_state[K_PURCHASE_CONFIG] = config
    st.session_state[K_PURCHASE_BALANCE] = balance
    store_fingerprint('purchase_balance', fingerprint_params)

    with panel('Operaciones'):
        stale_badge('purchase_balance', fingerprint_params)
        small_note(f'Precios de referencia del {aligned.dates[-1]:%d/%m/%Y}')
        _render_rows(balance, fmt)
        for code in balance.note_codes or ():
            status_box(label(NOTE_LABELS, code), 'info')
        if not balance.feasible:
            status_box('El presupuesto no permite ejecutar el balance completo con '
                       'las comisiones indicadas.', 'err')

    with panel('Totales'):
        cards = st.columns(5)
        with cards[0]:
            metric_card('Capital disponible', fmt_currency(balance.capital, fmt))
        with cards[1]:
            metric_card('Importe invertido', fmt_currency(balance.invested, fmt),
                        f'Nocional asignable: {fmt_currency(balance.notional, fmt)}')
        with cards[2]:
            metric_card('Costos totales', fmt_currency(balance.total_costs, fmt),
                        'Comisiones porcentuales y fijas', variant='vol')
        with cards[3]:
            metric_card('Efectivo remanente',
                        fmt_currency(balance.leftover_cash, fmt))
        with cards[4]:
            metric_card('Desviación máxima', fmt_pct(balance.max_deviation, 2, fmt),
                        'Frente al peso objetivo', variant='sharpe')
        if current_positions is not None:
            extra = st.columns(2)
            with extra[0]:
                metric_card('Valor de posiciones actuales',
                            fmt_currency(balance.current_value, fmt))
            with extra[1]:
                metric_card('Producto neto de ventas',
                            fmt_currency(balance.sell_proceeds, fmt),
                            'Neto de sus propios costos')
        small_note(ADVICE_NOTE)
