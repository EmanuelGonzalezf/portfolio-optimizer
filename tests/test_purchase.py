"""Tests of core/purchase.py (design.md section 21, row ``test_purchase.py``).

Covers CA-15 (the cash invariant, in its general form and in the collapsed form
of the requirement), CA-16 (fractional units without fees reproduce the target
weights) and CA-17 (the buy / sell / hold classification follows the weight
tolerance and never the sign of a historical return).
"""

from __future__ import annotations

import numpy as np
import pytest

from core.purchase import (CASH_INVARIANT_TOL, PurchaseConfig, allocate_integer_units,
                           build_purchase_balance, compute_notional_budget)
from core.types import InfeasibleConstraintsError

PRICES = np.array([25.0, 48.0, 120.0, 310.0, 7.0])
NAMES = ('ACTIVO_A', 'ACTIVO_B', 'ACTIVO_C', 'ACTIVO_D', 'ACTIVO_E')
EQUAL = np.full(5, 0.2)


def assert_cash_invariant(balance, config):
    """``buy_gross + buy_costs <= capital + sell_proceeds - cash_reserve + 1e-9``."""
    available = (config.capital + balance.sell_proceeds - config.cash_reserve
                 + CASH_INVARIANT_TOL)
    buy = balance.rows['gross_amount'].to_numpy(dtype=float)
    costs = balance.rows['costs'].to_numpy(dtype=float)
    buy_mask = buy > 0.0
    buy_gross = float(buy[buy_mask].sum())
    buy_costs = float(costs[buy_mask].sum())
    assert buy_gross == pytest.approx(balance.buy_gross, abs=1e-9)
    assert buy_gross + buy_costs <= available
    assert balance.leftover_cash >= 0.0
    assert balance.leftover_cash == pytest.approx(
        max(config.capital + balance.sell_proceeds - buy_gross - buy_costs, 0.0),
        abs=1e-9)
    assert balance.invested == pytest.approx(
        float((balance.rows['target_units'] * balance.rows['price']).sum()), abs=1e-9)


# --------------------------------------------------------------------------- #
# Cash invariant in three configurations (CA-15)
# --------------------------------------------------------------------------- #
def test_cash_invariant_without_current_positions():
    config = PurchaseConfig(capital=100_000.0, cash_reserve=2_000.0, pct_fee=0.001,
                            fixed_fee=5.0, integer_units=True)
    balance = build_purchase_balance(EQUAL, PRICES, NAMES, config)
    assert balance.feasible is True
    assert_cash_invariant(balance, config)
    # With no current positions the invariant collapses into the literal CA-15 form.
    assert balance.sell_proceeds == 0.0
    assert balance.current_value == 0.0
    assert balance.invested + balance.total_costs <= config.capital - config.cash_reserve


def test_cash_invariant_rebalance_funded_only_by_sales():
    current = np.array([100.0, 50.0, 10.0, 5.0, 200.0])
    config = PurchaseConfig(capital=0.0, pct_fee=0.002, fixed_fee=5.0,
                            integer_units=True)
    balance = build_purchase_balance(EQUAL, PRICES, NAMES, config,
                                     current_positions=current)
    assert balance.feasible is True
    assert balance.sell_proceeds > 0.0
    assert balance.buy_gross > 0.0
    assert_cash_invariant(balance, config)


def test_cash_invariant_mixed_rebalance():
    current = np.array([40.0, 20.0, 8.0, 2.0, 150.0])
    config = PurchaseConfig(capital=20_000.0, cash_reserve=500.0, pct_fee=0.0015,
                            fixed_fee=3.0, integer_units=True)
    balance = build_purchase_balance(np.array([0.35, 0.15, 0.25, 0.05, 0.20]),
                                     PRICES, NAMES, config, current_positions=current)
    assert balance.feasible is True
    assert_cash_invariant(balance, config)


def test_cash_invariant_on_a_full_turnover_rebalance():
    """Sell one asset entirely to buy another: both legs pay the percentage fee."""
    current = np.array([400.0, 0.0, 0.0, 0.0, 0.0])
    config = PurchaseConfig(capital=0.0, pct_fee=0.003, fixed_fee=8.0,
                            integer_units=True)
    balance = build_purchase_balance(np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
                                     PRICES, NAMES, config, current_positions=current)
    assert balance.feasible is True
    assert balance.rows.loc[0, 'target_units'] == 0.0
    assert balance.rows.loc[4, 'target_units'] > 0.0
    assert_cash_invariant(balance, config)


# --------------------------------------------------------------------------- #
# Fractional mode (CA-16)
# --------------------------------------------------------------------------- #
def test_fractional_without_fees_reproduces_target_weights():
    weights = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    config = PurchaseConfig(capital=50_000.0, integer_units=False)
    balance = build_purchase_balance(weights, PRICES, NAMES, config)
    assert np.allclose(balance.rows['final_weight'].to_numpy(dtype=float), weights,
                       atol=1e-12)
    assert balance.max_deviation <= 1e-12
    assert balance.invested == pytest.approx(config.capital, abs=1e-9)
    assert balance.total_costs == 0.0


def test_notional_budget_is_the_single_formula():
    assert compute_notional_budget(1000.0, 0.0, 0.0, 5) == pytest.approx(1000.0)
    assert compute_notional_budget(1000.0, 0.01, 10.0, 3) == pytest.approx(
        (1000.0 - 30.0) / 1.01)
    assert compute_notional_budget(10.0, 0.01, 10.0, 5) == 0.0


# --------------------------------------------------------------------------- #
# Integer allocation details
# --------------------------------------------------------------------------- #
def test_price_above_budget_is_flagged_and_never_rounded_up():
    config = PurchaseConfig(capital=100.0, pct_fee=0.0, fixed_fee=0.0,
                            integer_units=True)
    balance = build_purchase_balance(EQUAL, PRICES, NAMES, config)
    rows = balance.rows
    expensive = rows[rows['price'] > 100.0]
    assert (expensive['target_units'] == 0.0).all()
    assert (expensive['note_code'] == 'price_above_assignable_budget').all()
    assert 'price_above_assignable_budget' in balance.note_codes
    assert_cash_invariant(balance, config)


def test_integer_allocation_respects_the_budget_and_the_floor():
    target_values = np.array([2000.0, 2000.0, 2000.0, 2000.0, 2000.0])
    units, outlay, leftover = allocate_integer_units(
        target_values, PRICES, 10_000.0, 0.001, 5.0, np.zeros(5, dtype=bool))
    assert (units >= np.floor(target_values / PRICES)).all()
    assert outlay <= 10_000.0 + 1e-12
    assert leftover == pytest.approx(10_000.0 - outlay)
    charged = float(np.sum(units > 0.0))
    assert outlay == pytest.approx(float(np.sum(units * PRICES)) * 1.001 + 5.0 * charged)


def test_capital_consumed_by_the_fixed_fee_yields_no_operations():
    config = PurchaseConfig(capital=10.0, fixed_fee=50.0, integer_units=True)
    balance = build_purchase_balance(EQUAL, PRICES, NAMES, config)
    assert balance.feasible is False
    assert balance.notional == 0.0
    assert (balance.rows['target_units'] == 0.0).all()
    assert balance.leftover_cash == pytest.approx(10.0)
    assert 'notional_non_positive' in balance.note_codes


def test_non_positive_budget_is_reported():
    config = PurchaseConfig(capital=1_000.0, cash_reserve=1_500.0, integer_units=True)
    balance = build_purchase_balance(EQUAL, PRICES, NAMES, config)
    assert balance.feasible is False
    assert balance.note_codes == ('budget_non_positive',)
    assert (balance.rows['delta_units'] == 0.0).all()


# --------------------------------------------------------------------------- #
# Classification (CA-17)
# --------------------------------------------------------------------------- #
def test_classification_follows_the_weight_tolerance():
    value = 100_000.0
    current_weights = np.array([0.205, 0.195, 0.200, 0.260, 0.140])
    current = current_weights * value / PRICES
    config = PurchaseConfig(capital=0.0, pct_fee=0.0, fixed_fee=0.0,
                            integer_units=False, weight_tolerance=0.01)
    balance = build_purchase_balance(EQUAL, PRICES, NAMES, config,
                                     current_positions=current)
    actions = list(balance.rows['action'])
    assert actions == ['hold', 'hold', 'hold', 'sell', 'buy']
    assert np.allclose(balance.rows['current_weight'].to_numpy(dtype=float),
                       current_weights, atol=1e-12)


def test_tighter_tolerance_turns_holds_into_orders():
    value = 100_000.0
    current_weights = np.array([0.205, 0.195, 0.200, 0.260, 0.140])
    current = current_weights * value / PRICES
    config = PurchaseConfig(capital=0.0, integer_units=False, weight_tolerance=0.001)
    balance = build_purchase_balance(EQUAL, PRICES, NAMES, config,
                                     current_positions=current)
    assert list(balance.rows['action']) == ['sell', 'buy', 'hold', 'sell', 'buy']


def test_classification_ignores_historical_performance():
    """The action depends only on the weight gap: no return enters the balance."""
    value = 100_000.0
    current = np.array([0.30, 0.20, 0.20, 0.20, 0.10]) * value / PRICES
    config = PurchaseConfig(capital=0.0, integer_units=False, weight_tolerance=0.01)
    balance = build_purchase_balance(EQUAL, PRICES, NAMES, config,
                                     current_positions=current)
    gaps = (balance.rows['current_weight'].to_numpy(dtype=float)
            - balance.rows['target_weight'].to_numpy(dtype=float))
    for action, gap in zip(balance.rows['action'], gaps):
        if abs(gap) <= config.weight_tolerance:
            assert action == 'hold'
        else:
            assert action == ('sell' if gap > 0 else 'buy')


def test_weight_tolerance_out_of_range_raises():
    with pytest.raises(ValueError, match='weight_tolerance_out_of_range'):
        PurchaseConfig(capital=1000.0, weight_tolerance=0.9)
    with pytest.raises(ValueError, match='weight_tolerance_out_of_range'):
        PurchaseConfig(capital=1000.0, weight_tolerance=-0.01)


def test_percentage_fee_at_or_above_one_raises():
    with pytest.raises(InfeasibleConstraintsError):
        PurchaseConfig(capital=1000.0, pct_fee=1.0)


# --------------------------------------------------------------------------- #
# Single implementation of the notional budget (FD-5)
# --------------------------------------------------------------------------- #
def test_notional_budget_shared_by_purchase_and_cashflow():
    """``build_purchase_balance`` and ``build_initial_position`` with the same
    parameters produce identical units in fractional mode, to 1e-12.

    Pinning this down is what proves core/cashflow.py IMPORTS
    ``compute_notional_budget`` instead of repeating the arithmetic: two copies
    of a formula can be asserted equal today and diverge tomorrow, a single
    function cannot.
    """
    from core.cashflow import LiquidityPolicy, build_initial_position

    capital, reserve, pct_fee, fixed_fee = 250_000.0, 7_500.0, 0.0015, 6.0
    config = PurchaseConfig(capital=capital, cash_reserve=reserve, pct_fee=pct_fee,
                            fixed_fee=fixed_fee, integer_units=False)
    balance = build_purchase_balance(EQUAL, PRICES, NAMES, config)
    policy = LiquidityPolicy(pct_fee=pct_fee, fixed_fee=fixed_fee,
                             cash_reserve_initial=reserve)
    position = build_initial_position(capital, EQUAL, PRICES, policy,
                                      integer_units=False)

    assert np.asarray(position.units) == pytest.approx(
        balance.rows['target_units'].to_numpy(dtype=float), abs=1e-12)
    assert position.invested == pytest.approx(balance.invested, abs=1e-9)
    assert compute_notional_budget(capital - reserve, pct_fee, fixed_fee, 5) == \
        pytest.approx(balance.notional, abs=1e-12)
