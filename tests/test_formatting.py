"""Tests for core/formatting.py plus the two state tests of design.md section 21.

``ui/state.py`` imports Streamlit, but ``compute_fingerprint`` is a pure
function, so it is exercised here without launching the app.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from core.backtest import BacktestConfig
from core.formatting import (NOT_AVAILABLE, CurrencyFormat, fmt_currency, fmt_num, fmt_pct,
                             fmt_units, sign_class, style_table)
from core.types import Constraints, FrequencyConvention
from ui.state import FINGERPRINT_DEPS, compute_fingerprint


def test_currency_uses_spanish_separators(fmt):
    assert fmt_currency(1234567.891, fmt) == '$1.234.567,89'
    assert fmt_currency(0.0, fmt) == '$0,00'
    assert fmt_currency(-2500.5, fmt) == '-$2.500,50'


def test_currency_respects_a_custom_format():
    custom = CurrencyFormat(symbol='EUR', decimals=0, thousands=' ', decimal=',',
                            symbol_first=False)
    assert fmt_currency(1234567.891, custom) == '1 234 568 EUR'


def test_percentages_are_readable(fmt):
    assert fmt_pct(0.1234) == '12,34 %'
    assert fmt_pct(-0.005, 2) == '-0,50 %'
    assert fmt_pct(1.5, 1) == '150,0 %'


def test_plain_numbers_and_units(fmt):
    assert fmt_num(3.14159265, 4) == '3,1416'
    assert fmt_num(12345.6789, 2) == '12.345,68'
    assert fmt_units(1234.0, integer=True) == '1.234'
    assert fmt_units(1234.5678, integer=False) == '1.234,5678'


def test_non_finite_values_render_as_a_dash(fmt):
    for value in (None, float('nan'), float('inf'), float('-inf'), np.nan, pd.NA):
        assert fmt_currency(value, fmt) == NOT_AVAILABLE
        assert fmt_pct(value) == NOT_AVAILABLE
        assert fmt_num(value) == NOT_AVAILABLE
        assert fmt_units(value, integer=True) == NOT_AVAILABLE


def test_sign_class_covers_the_three_cases():
    assert sign_class(0.01) == 'pos'
    assert sign_class(-0.01) == 'neg'
    assert sign_class(0.0) == ''
    assert sign_class(float('nan')) == ''
    assert sign_class(None) == ''


def test_degenerate_realized_metrics_render_as_dashes(fmt):
    metrics = {'cagr': math.nan, 'vol_real': math.nan, 'sharpe_real': math.nan}
    assert [fmt_pct(metrics['cagr']), fmt_pct(metrics['vol_real']),
            fmt_num(metrics['sharpe_real'], 2)] == [NOT_AVAILABLE] * 3


def test_style_table_formats_present_columns_only(fmt):
    frame = pd.DataFrame({'asset': ['A', 'B'],
                          'weight': [0.25, 0.75],
                          'amount': [1000.0, 2500.5],
                          'change': [0.01, -0.02]})
    styler = style_table(frame, pct_cols=('weight', 'missing_column'),
                         money_cols=('amount',), num_cols=(), signed_cols=('change',),
                         fmt=fmt)
    rendered = styler.to_html()
    assert '25,00 %' in rendered
    assert '$2.500,50' in rendered
    assert '#4ade80' in rendered
    assert '#f87171' in rendered


def test_currency_format_roundtrip(fmt):
    assert CurrencyFormat.from_tuple(fmt.to_tuple()) == fmt


# --------------------------------------------------------------------------- #
# ui/state.py: one fingerprint per result (RF-P4, design.md section 13.1b)
# --------------------------------------------------------------------------- #
PARAM_DEPS = sorted({dependency
                     for dependencies in FINGERPRINT_DEPS.values()
                     for dependency in dependencies
                     if dependency not in FINGERPRINT_DEPS})

BASE_PARAMS = {name: f'base_{name}' for name in PARAM_DEPS}
BASE_PARENTS = {name: f'fp_{name}' for name in FINGERPRINT_DEPS}


def test_fingerprint_depends_only_on_declared_deps():
    for result_key, dependencies in FINGERPRINT_DEPS.items():
        reference = compute_fingerprint(result_key, BASE_PARAMS, BASE_PARENTS)

        # (a) every declared parameter moves the fingerprint.
        for dependency in dependencies:
            if dependency in FINGERPRINT_DEPS:
                continue
            changed = dict(BASE_PARAMS)
            changed[dependency] = 'moved'
            assert compute_fingerprint(result_key, changed, BASE_PARENTS) != reference, \
                f'{result_key} ignored its declared dependency {dependency}'

        # (b) a parameter that is not declared leaves it untouched.
        for dependency in PARAM_DEPS:
            if dependency in dependencies:
                continue
            changed = dict(BASE_PARAMS)
            changed[dependency] = 'moved'
            assert compute_fingerprint(result_key, changed, BASE_PARENTS) == reference, \
                f'{result_key} reacted to the undeclared {dependency}'

        # (c) a change in a parent result propagates to the child.
        for dependency in dependencies:
            if dependency not in FINGERPRINT_DEPS:
                continue
            parents = dict(BASE_PARENTS)
            parents[dependency] = 'fp_moved'
            assert compute_fingerprint(result_key, BASE_PARAMS, parents) != reference


def test_frontier_points_do_not_move_the_backtest():
    reference = compute_fingerprint('backtest_result', BASE_PARAMS, BASE_PARENTS)
    changed = dict(BASE_PARAMS, n_frontier_points=200)
    assert compute_fingerprint('backtest_result', changed, BASE_PARENTS) == reference
    assert (compute_fingerprint('frontier', changed, BASE_PARENTS)
            != compute_fingerprint('frontier', BASE_PARAMS, BASE_PARENTS))


def test_a_local_value_changes_only_its_own_section():
    purchase = compute_fingerprint('purchase_balance', BASE_PARAMS, BASE_PARENTS)
    capacity = compute_fingerprint('capacity_metrics', BASE_PARAMS, BASE_PARENTS)
    local = dict(BASE_PARAMS, capital_s5=250_000.0)
    assert compute_fingerprint('purchase_balance', local, BASE_PARENTS) != purchase
    assert compute_fingerprint('capacity_metrics', local, BASE_PARENTS) == capacity


def test_fingerprint_is_stable_and_rejects_unknown_results():
    assert (compute_fingerprint('optimization', BASE_PARAMS, BASE_PARENTS)
            == compute_fingerprint('optimization', dict(BASE_PARAMS), dict(BASE_PARENTS)))
    with pytest.raises(KeyError):
        compute_fingerprint('not_a_result', BASE_PARAMS, BASE_PARENTS)


# --------------------------------------------------------------------------- #
# Serialisation contract of everything that crosses st.cache_data (section 13.2)
# --------------------------------------------------------------------------- #
def _plain(value):
    """Recursively turn a ``to_tuple`` payload into comparable primitives."""
    if isinstance(value, np.ndarray):
        return ['ndarray', list(value.shape), np.asarray(value, dtype=float).tolist()]
    if isinstance(value, (pd.DatetimeIndex, pd.Index)):
        return ['index', [str(item) for item in value]]
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def test_dataclass_tuple_roundtrip(initial_position, schedule_simple, thresholds):
    cashflow = pytest.importorskip('core.cashflow')
    scenarios = pytest.importorskip('core.scenarios')
    data_loader = pytest.importorskip('core.data_loader')

    spec = scenarios.plan_path_simulation(
        np.eye(3) * 1e-4, 20, 100, method='gaussian_cholesky', block_size=None,
        seed=42)

    instances = [
        data_loader.MissingPolicy(method='ffill', limit_days=3),
        data_loader.ParsePreferences(sheet='Precios', header_row=2, decimal='comma',
                                     dayfirst='day_first'),
        Constraints(allow_short=True, max_weight=0.4, short_lower_bound=-0.3),
        FrequencyConvention(trading_days=250, rf_annual_effective=0.031),
        BacktestConfig(window=120, rebalance_rule='Q', initial_capital=50_000.0,
                       cost_rate=0.002),
        schedule_simple,
        cashflow.LiquidityPolicy(pct_fee=0.002, fixed_fee=5.0, allow_sales=True,
                                 cash_reserve_initial=100.0, cash_interest=True,
                                 sale_rule='pro_rata'),
        initial_position,
        thresholds,
        spec,
    ]

    for instance in instances:
        payload = instance.to_tuple()
        restored = type(instance).from_tuple(payload)
        assert type(restored) is type(instance)
        assert _plain(restored.to_tuple()) == _plain(payload), type(instance).__name__
