"""Global parameter sidebar (RF-P1, RF-P2) with the ranges of design.md section 17.

Every control here feeds a real calculation: the returned dict is what the
sections read, what the fingerprints of ``ui/state.py`` hash and what the
parameters sheet of the export reports. No control is decorative.

Ownership rule (design.md section 13.1): this is the global default. Sections 4
to 7 may override some of these values locally and then the local value wins
and the section paints ``badge('valor local')``. The single exception is the
analysis period, which is configured only here.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.formatting import CurrencyFormat
from core.types import Constraints, FrequencyConvention
from ui.state import K_ALIGNED
from ui.theme import section_header, small_note, status_box

__all__ = ['render_sidebar', 'DEFAULTS', 'SEED_MAX']

SEED_MAX = 2 ** 31 - 1

DEFAULTS = {
    'rf_percent': 4.50,
    'trading_days': 252,
    'allow_short': False,
    'short_lb': -1.0,
    'n_weight_sims': 10_000,
    'n_frontier_points': 60,
    'capital': 100_000.0,
    'max_weight': 1.0,
    'pct_fee_percent': 0.1,
    'fixed_fee': 0.0,
    'seed': 42,
}

RF_NOTE = ('Tasa anual efectiva; se convierte a capitalización continua para el '
           'Sharpe')

THOUSANDS_OPTIONS = {'.': 'Punto', ',': 'Coma', ' ': 'Espacio'}
DECIMAL_OPTIONS = {',': 'Coma', '.': 'Punto'}


def _asset_count() -> int:
    """Number of aligned assets, or five while no data is loaded."""
    aligned = st.session_state.get(K_ALIGNED)
    if aligned is None:
        return 5
    return max(int(len(aligned.names)), 1)


def _period_controls(label: str) -> tuple:
    """Analysis period. Read-only everywhere else (section 13.1 exception)."""
    aligned = st.session_state.get(K_ALIGNED)
    if aligned is None:
        small_note('El periodo se podrá acotar cuando haya datos cargados')
        return None, None
    first = pd.Timestamp(aligned.dates[0]).date()
    last = pd.Timestamp(aligned.dates[-1]).date()
    chosen = st.date_input(label, value=(first, last), min_value=first,
                           max_value=last, key='sidebar_period')
    if isinstance(chosen, (list, tuple)) and len(chosen) == 2:
        return str(chosen[0]), str(chosen[1])
    return str(first), str(last)


def render_sidebar() -> dict:
    """Render the sidebar and return the dict of global effective parameters."""
    with st.sidebar:
        section_header('Parámetros generales')

        rf_percent = st.number_input(
            'Tasa libre de riesgo anual (%)', min_value=-10.0, max_value=50.0,
            value=float(DEFAULTS['rf_percent']), step=0.05, format='%.2f',
            key='sidebar_rf')
        small_note(RF_NOTE)

        trading_days = int(st.number_input(
            'Días bursátiles por año', min_value=1, max_value=366,
            value=int(DEFAULTS['trading_days']), step=1, key='sidebar_tdy'))

        allow_short = st.toggle('Permitir ventas en corto',
                                value=bool(DEFAULTS['allow_short']),
                                key='sidebar_shorts')
        short_lb = float(DEFAULTS['short_lb'])
        if allow_short:
            short_lb = st.number_input(
                'Cota inferior de peso con cortos', min_value=-5.0, max_value=-0.01,
                value=float(DEFAULTS['short_lb']), step=0.01, format='%.2f',
                key='sidebar_short_lb')
            small_note('Con ventas en corto el peso de un activo puede ser negativo: '
                       'aumenta el riesgo y exige garantías no modelizadas aquí')

        section_header('Restricciones de ponderación')
        n_assets = _asset_count()
        min_weight = 1.0 / float(n_assets)
        max_weight = st.number_input(
            'Límite máximo por activo', min_value=0.01, max_value=1.0,
            value=float(DEFAULTS['max_weight']), step=0.05, format='%.2f',
            key='sidebar_max_weight')
        small_note(f'Mínimo admisible con {n_assets} activos: 1/n = {min_weight:.4f}')
        constraints_feasible = max_weight * n_assets >= 1.0 - 1e-12
        if not constraints_feasible:
            status_box(f'El límite máximo por activo es infactible con {n_assets} '
                       f'activos: el mínimo admisible es 1/n = {min_weight:.4f}. '
                       f'Los cálculos quedan bloqueados hasta corregirlo.', 'err')

        section_header('Simulación')
        n_weight_sims = int(st.number_input(
            'Simulaciones Monte Carlo de pesos', min_value=1_000, max_value=200_000,
            value=int(DEFAULTS['n_weight_sims']), step=1_000,
            key='sidebar_weight_sims'))
        n_frontier_points = int(st.number_input(
            'Puntos de la frontera eficiente', min_value=10, max_value=200,
            value=int(DEFAULTS['n_frontier_points']), step=5,
            key='sidebar_frontier_points'))
        seed = int(st.number_input('Semilla', min_value=0, max_value=SEED_MAX,
                                   value=int(DEFAULTS['seed']), step=1,
                                   key='sidebar_seed'))

        section_header('Capital y costos')
        capital = float(st.number_input(
            'Capital inicial de inversión', min_value=0.01,
            value=float(DEFAULTS['capital']), step=1_000.0, format='%.2f',
            key='sidebar_capital'))
        pct_fee_percent = st.number_input(
            'Comisión porcentual (%)', min_value=0.0, max_value=10.0,
            value=float(DEFAULTS['pct_fee_percent']), step=0.1, format='%.2f',
            key='sidebar_pct_fee')
        fixed_fee = float(st.number_input(
            'Comisión fija por operación', min_value=0.0,
            value=float(DEFAULTS['fixed_fee']), step=1.0, format='%.2f',
            key='sidebar_fixed_fee'))

        section_header('Periodo de análisis')
        period_start, period_end = _period_controls('Periodo de análisis')

        section_header('Moneda y formato')
        symbol = st.text_input('Símbolo monetario', value='$', max_chars=4,
                               key='sidebar_symbol')
        decimals = int(st.number_input('Decimales', min_value=0, max_value=6, value=2,
                                       step=1, key='sidebar_decimals'))
        thousands = st.selectbox(
            'Separador de miles', list(THOUSANDS_OPTIONS),
            format_func=lambda key: THOUSANDS_OPTIONS[key], key='sidebar_thousands')
        decimal = st.selectbox(
            'Separador decimal', list(DECIMAL_OPTIONS),
            format_func=lambda key: DECIMAL_OPTIONS[key], key='sidebar_decimal')
        symbol_first = st.toggle('Símbolo antes del importe', value=True,
                                 key='sidebar_symbol_first')

    rf = float(rf_percent) / 100.0
    pct_fee = float(pct_fee_percent) / 100.0
    fmt = CurrencyFormat(symbol=symbol or '$', decimals=decimals, thousands=thousands,
                         decimal=decimal, symbol_first=bool(symbol_first))
    convention = FrequencyConvention(trading_days=trading_days,
                                     rf_annual_effective=rf)
    constraints = Constraints(allow_short=bool(allow_short),
                              max_weight=float(max_weight),
                              short_lower_bound=float(short_lb))

    return {
        'rf': rf,
        'rf_percent': float(rf_percent),
        'trading_days': trading_days,
        'allow_short': bool(allow_short),
        'short_lb': float(short_lb),
        'max_weight': float(max_weight),
        'min_weight': min_weight,
        'n_assets': n_assets,
        'constraints_feasible': bool(constraints_feasible),
        'n_weight_sims': n_weight_sims,
        'n_frontier_points': n_frontier_points,
        'seed': seed,
        'capital': capital,
        'pct_fee': pct_fee,
        'fixed_fee': fixed_fee,
        'period': (period_start, period_end),
        'period_start': period_start,
        'period_end': period_end,
        'fmt': fmt,
        'convention': convention,
        'constraints': constraints,
    }
