"""Tests of core/cashflow.py (design.md section 21, row ``test_cashflow.py``).

Covers CA-18 (forced sale solved by the coverage equation), CA-19 (default
bookkeeping against a hand-made case), CA-20 (no returns on fictitious capital),
CA-21 (the accounting identity, step by step and aggregate, with and without
rebalancing) and CA-22 (additional capital by bisection on total capital).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.backtest import generate_rebalance_indices
from core.cashflow import (SALE_NOT_EXECUTED, TWO_INSTALLMENTS_SAME_DAY,
                           InitialPosition, InstallmentSchedule, LiquidityPolicy,
                           build_calendar, build_initial_position, build_schedule,
                           check_wealth_identity, map_payment_indices,
                           required_additional_capital, schedule_from_frame,
                           simulate_cashflow_paths, simulate_cashflow_single,
                           solve_gross_sale)
from core.purchase import PurchaseConfig, build_purchase_balance
from core.types import InfeasibleConstraintsError, PortfolioError, ScheduleError

PRICES = np.array([25.0, 48.0, 120.0, 310.0, 7.0])
NAMES = ('ACTIVO_A', 'ACTIVO_B', 'ACTIVO_C', 'ACTIVO_D', 'ACTIVO_E')
EQUAL = np.full(5, 0.2)
VALUATION = pd.Timestamp('2024-01-31')          # a Wednesday
RF_SIMPLE_DAILY = (1.045 ** (1.0 / 252.0)) - 1.0


class Setup:
    """Everything one engine run needs, assembled from a few knobs."""

    def __init__(self, capital=100_000.0, n_installments=6, amount=1000.0,
                 growth_rate=0.0, external_income=0.0, frequency='monthly',
                 first_date='2024-02-15', pct_fee=0.0, fixed_fee=0.0,
                 allow_sales=True, cash_reserve_initial=0.0, cash_interest=False,
                 integer_units=False, holding_period_days=0, weights=EQUAL,
                 prices=PRICES):
        self.policy = LiquidityPolicy(pct_fee=pct_fee, fixed_fee=fixed_fee,
                                      allow_sales=allow_sales,
                                      cash_reserve_initial=cash_reserve_initial,
                                      cash_interest=cash_interest)
        self.schedule = build_schedule(pd.Timestamp(first_date), frequency,
                                       n_installments, amount, growth_rate,
                                       external_income)
        self.payment_indices, self.horizon, self.effective = map_payment_indices(
            VALUATION, self.schedule, holding_period_days)
        self.calendar = build_calendar(VALUATION, self.schedule, holding_period_days)
        self.weights = np.asarray(weights, dtype=float)
        self.prices = np.asarray(prices, dtype=float)
        self.initial = build_initial_position(capital, self.weights, self.prices,
                                              self.policy,
                                              integer_units=integer_units)
        self.capital = float(capital)

    def zero_path(self) -> np.ndarray:
        return np.zeros((self.horizon, self.prices.size), dtype=float)

    def flat_path(self, daily) -> np.ndarray:
        rate = np.asarray(daily, dtype=float)
        if rate.ndim == 0:
            rate = np.full(self.prices.size, float(rate))
        return np.repeat(rate[None, :], self.horizon, axis=0)

    def run(self, path, rebalance_indices=None, target_weights=None):
        return simulate_cashflow_single(path, self.initial, self.schedule,
                                        self.payment_indices, self.policy,
                                        RF_SIMPLE_DAILY, rebalance_indices,
                                        target_weights, self.effective)


def assert_aggregate_identity(result, initial, index=0):
    """``wealth_final == W0 + returns + external - paid - (sale + rebalance)``."""
    expected = (initial.wealth_initial + result.total_returns[index]
                + result.total_external[index] - result.total_paid[index]
                - result.total_sale_costs[index] - result.total_rebalance_costs[index])
    assert result.final_wealth[index] == pytest.approx(expected, abs=1e-9)
    assert result.total_costs[index] == pytest.approx(
        result.total_sale_costs[index] + result.total_rebalance_costs[index], abs=1e-12)


# --------------------------------------------------------------------------- #
# Forced sale
# --------------------------------------------------------------------------- #
def test_solve_gross_sale_is_exact():
    need = np.array([1000.0, 250.0, 0.0])
    n_sellable = np.array([5.0, 3.0, 1.0])
    gross = solve_gross_sale(need, 0.002, 5.0, n_sellable)
    recovered = gross * (1.0 - 0.002) - 5.0 * n_sellable
    assert recovered == pytest.approx(need, abs=1e-12)


def test_payment_is_covered_without_residual_deficit_after_a_forced_sale():
    setup = Setup(capital=100_000.0, n_installments=6, amount=1000.0,
                  pct_fee=0.002, fixed_fee=5.0)
    # Fractional purchase with no cash reserve leaves exactly zero cash, so every
    # installment can only be paid by selling.
    assert setup.initial.cash == pytest.approx(0.0, abs=1e-9)
    ledger, result = setup.run(setup.zero_path())
    assert bool(result.covered_all[0]) is True
    assert ledger['deficit'].to_numpy() == pytest.approx(np.zeros(6), abs=1e-12)
    assert (ledger['gross_sold'].to_numpy() > ledger['paid'].to_numpy()).all()
    assert result.total_sale_costs[0] > 0.0
    assert_aggregate_identity(result, setup.initial)


def test_unprofitable_sale_is_not_executed():
    """``mv_total`` small against ``fixed_fee*n`` must not liquidate for nothing."""
    initial = InitialPosition(units=np.array([0.4, 0.0, 0.0, 0.0, 0.0]), cash=0.0,
                              purchase_costs=0.0, invested=10.0, prices=PRICES,
                              integer_units=False, source='synthetic')
    policy = LiquidityPolicy(pct_fee=0.0, fixed_fee=20.0)
    schedule = build_schedule(pd.Timestamp('2024-02-15'), 'monthly', 1, 5.0, 0.0)
    payment_indices, horizon, effective = map_payment_indices(VALUATION, schedule)
    ledger, result = simulate_cashflow_single(np.zeros((horizon, 5)), initial, schedule,
                                              payment_indices, policy, RF_SIMPLE_DAILY,
                                              None, None, effective)
    assert ledger['note_code'].iloc[0] == SALE_NOT_EXECUTED
    assert ledger['gross_sold'].iloc[0] == 0.0
    assert ledger['sale_costs'].iloc[0] == 0.0
    assert ledger['mv_after'].iloc[0] == pytest.approx(10.0, abs=1e-12)
    assert ledger['paid'].iloc[0] == 0.0
    assert ledger['deficit'].iloc[0] == pytest.approx(5.0, abs=1e-12)
    assert SALE_NOT_EXECUTED in result.note_codes


def test_cash_and_positions_are_never_negative():
    setup = Setup(capital=4_000.0, n_installments=8, amount=1000.0, pct_fee=0.002,
                  fixed_fee=2.0)
    ledger, result = setup.run(setup.flat_path(0.0003))
    assert (ledger['cash_after'].to_numpy() >= -1e-12).all()
    assert (ledger['cash_before'].to_numpy() >= -1e-12).all()
    assert (ledger['mv_after'].to_numpy() >= -1e-12).all()
    assert (ledger['wealth_after'].to_numpy() >= -1e-12).all()
    assert (result.cash_at_payments >= -1e-12).all()
    assert result.min_wealth[0] >= -1e-12


# --------------------------------------------------------------------------- #
# Accounting identity
# --------------------------------------------------------------------------- #
def test_identity_step_by_step_and_aggregate():
    setup = Setup(capital=60_000.0, n_installments=9, amount=1500.0, growth_rate=0.02,
                  external_income=120.0, pct_fee=0.0015, fixed_fee=3.0,
                  cash_reserve_initial=800.0, cash_interest=True)
    daily = np.array([0.0002, -0.0001, 0.0004, 0.00005, 0.0006])
    ledger, result = setup.run(setup.flat_path(daily))
    # simulate_cashflow_single already ran the step-by-step check; re-run it to
    # make the assertion explicit in this test.
    check_wealth_identity(ledger, setup.initial.wealth_initial)
    opening = ledger['wealth_open'].to_numpy()
    closing = ledger['wealth_after'].to_numpy()
    assert opening[0] == pytest.approx(setup.initial.wealth_initial, abs=1e-9)
    assert opening[1:] == pytest.approx(closing[:-1], abs=1e-9)
    assert ledger['wealth_before'].to_numpy() == pytest.approx(
        opening + ledger['returns_in_period'].to_numpy()
        - ledger['rebalance_costs_pre'].to_numpy(), abs=1e-9)
    assert_aggregate_identity(result, setup.initial)
    assert setup.initial.cash >= setup.policy.cash_reserve_initial - 1e-9


def test_identity_holds_with_rebalancing_on_payment_date():
    setup = Setup(capital=80_000.0, n_installments=6, amount=1200.0, pct_fee=0.001,
                  fixed_fee=1.0)
    target = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
    ledger, result = setup.run(setup.flat_path([0.0005, 0.0001, -0.0002, 0.0003, 0.0007]),
                               rebalance_indices=setup.payment_indices,
                               target_weights=target)
    assert (ledger['rebalance_costs_post'].to_numpy() > 0.0).all()
    assert ledger['rebalance_costs_pre'].to_numpy() == pytest.approx(np.zeros(6),
                                                                     abs=1e-12)
    assert ledger['rebalance_costs'].to_numpy() == pytest.approx(
        ledger['rebalance_costs_pre'].to_numpy()
        + ledger['rebalance_costs_post'].to_numpy(), abs=1e-12)
    check_wealth_identity(ledger, setup.initial.wealth_initial)
    assert_aggregate_identity(result, setup.initial)


def test_identity_holds_with_rebalancing_strictly_inside_period():
    """Monthly rule with quarterly installments: the case that used to break.

    A rebalance strictly inside ``(t_{k-1}, t_k)`` is already reflected in the
    market value measured at ``t_k``, so its cost belongs to
    ``rebalance_costs_pre[k]`` and must NOT be subtracted again in
    ``wealth_after[k]``.
    """
    # The holding period pushes the horizon past the last installment, so the
    # last calendar position is not a payment date and the final month-end
    # rebalance lands in the tail instead of on the last installment.
    setup = Setup(capital=90_000.0, n_installments=4, amount=1500.0,
                  frequency='quarterly', first_date='2024-02-15', pct_fee=0.001,
                  fixed_fee=2.0, holding_period_days=230)
    rebalance_indices = generate_rebalance_indices(setup.calendar, 'M', 0)
    assert rebalance_indices.size >= 6
    # No month end coincides with a 15th, so every rebalance is strictly inside.
    assert not set(rebalance_indices.tolist()) & set(setup.payment_indices.tolist())
    inside = [int(np.sum((rebalance_indices > low) & (rebalance_indices < high)))
              for low, high in zip(np.r_[0, setup.payment_indices[:-1]],
                                   setup.payment_indices)]
    assert max(inside) >= 2

    target = np.array([0.35, 0.25, 0.2, 0.1, 0.1])
    ledger, result = setup.run(setup.flat_path([0.0006, 0.0002, -0.0003, 0.0004, 0.0008]),
                               rebalance_indices=rebalance_indices,
                               target_weights=target)
    pre = ledger['rebalance_costs_pre'].to_numpy()
    post = ledger['rebalance_costs_post'].to_numpy()
    assert (pre[1:] > 0.0).all()
    assert post == pytest.approx(np.zeros(4), abs=1e-12)
    assert ledger['rebalance_costs'].to_numpy() == pytest.approx(pre + post, abs=1e-12)
    check_wealth_identity(ledger, setup.initial.wealth_initial)
    assert_aggregate_identity(result, setup.initial)
    # tail_rebalance_costs: the rebalance after the last installment has no `k`
    # to attach to and only reaches the aggregate identity.
    assert result.total_rebalance_costs[0] > float(pre.sum() + post.sum())


# --------------------------------------------------------------------------- #
# Index origin convention (section 10.1)
# --------------------------------------------------------------------------- #
def test_payment_index_convention():
    """A single non-zero return on one day fixes the index origin.

    Calendar from 2024-01-31: [Jan31, Feb1, Feb2, Feb5, Feb6], so the two
    installments land on indices 3 and 4. With ``log_returns[3] = ln 2`` (the row
    that moves ``calendar[3] -> calendar[4]``), the installment at index 4 sees
    the revalued price and the one at index 3 does not. A one-day shift in either
    direction breaks one of the two assertions.
    """
    frame = pd.DataFrame({'date': ['2024-02-05', '2024-02-06'], 'amount': [0.0, 0.0]})
    schedule = schedule_from_frame(frame)
    payment_indices, horizon, effective = map_payment_indices(VALUATION, schedule)
    assert payment_indices.tolist() == [3, 4]
    assert horizon == 4

    policy = LiquidityPolicy(pct_fee=0.0, fixed_fee=0.0)
    initial = build_initial_position(100_000.0, EQUAL, PRICES, policy)
    path = np.zeros((horizon, 5))
    path[payment_indices[1] - 1, :] = np.log(2.0)

    ledger, _ = simulate_cashflow_single(path, initial, schedule, payment_indices,
                                         policy, RF_SIMPLE_DAILY, None, None, effective)
    mv_before = ledger['mv_before'].to_numpy()
    assert mv_before[0] == pytest.approx(initial.invested, rel=1e-12)
    assert mv_before[1] == pytest.approx(2.0 * mv_before[0], rel=1e-12)


def test_rebalance_indices_share_the_payment_convention():
    setup = Setup(n_installments=3, frequency='quarterly')
    indices = generate_rebalance_indices(setup.calendar, 'M', 0)
    assert indices.min() >= 1
    assert indices.max() <= setup.horizon
    assert -1 not in indices.tolist()


# --------------------------------------------------------------------------- #
# Default bookkeeping (CA-19)
# --------------------------------------------------------------------------- #
def test_default_bookkeeping_matches_handmade_case():
    """Four installments of 1.000 against a wealth of exactly 2.500, zero returns.

    By hand: 1.000, 1.000, then 500 with a deficit of 500, then 0 with a deficit
    of 1.000 -> two defaults, the first at index 2, max deficit 1.000 and
    cumulative deficit 1.500.
    """
    setup = Setup(capital=2_500.0, n_installments=4, amount=1000.0)
    ledger, result = setup.run(setup.zero_path())
    assert int(result.n_defaults[0]) == 2
    assert int(result.first_default_idx[0]) == 2
    assert result.max_deficit[0] == pytest.approx(1000.0, abs=1e-9)
    assert result.cumulative_deficit[0] == pytest.approx(1500.0, abs=1e-9)
    assert ledger['paid'].to_numpy() == pytest.approx([1000.0, 1000.0, 500.0, 0.0],
                                                      abs=1e-9)
    assert ledger['deficit'].to_numpy() == pytest.approx([0.0, 0.0, 500.0, 1000.0],
                                                         abs=1e-9)
    # Without an intra-period rebalance the first step collapses into
    # wealth_before[k] == wealth_after[k-1] + returns_in_period[k].
    opening = ledger['wealth_open'].to_numpy()
    assert ledger['wealth_before'].to_numpy() == pytest.approx(
        opening + ledger['returns_in_period'].to_numpy(), abs=1e-9)
    assert bool(result.covered_all[0]) is False


def test_no_returns_are_generated_on_exhausted_wealth():
    setup = Setup(capital=3_000.0, n_installments=6, amount=1000.0)
    ledger, result = setup.run(setup.flat_path(0.001))
    first = int(result.first_default_idx[0])
    assert first >= 0
    tail = ledger.iloc[first + 1:]
    assert (tail['wealth_after'].to_numpy() == 0.0).all()
    assert (tail['paid'].to_numpy() == 0.0).all()
    assert tail['deficit'].to_numpy() == pytest.approx(tail['due'].to_numpy(), abs=1e-12)
    assert result.final_wealth[0] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Initial position (section 10.2b)
# --------------------------------------------------------------------------- #
def test_initial_position_adds_up():
    policy = LiquidityPolicy(pct_fee=0.0012, fixed_fee=4.0, cash_reserve_initial=2_500.0)
    position = build_initial_position(150_000.0, EQUAL, PRICES, policy)
    assert position.invested + position.cash + position.purchase_costs == \
        pytest.approx(150_000.0, abs=1e-9)
    assert position.cash >= policy.cash_reserve_initial - 1e-9
    assert position.wealth_initial == pytest.approx(150_000.0 - position.purchase_costs,
                                                    abs=1e-9)
    assert position.source == 'synthetic'


def test_initial_position_from_balance_reproduces_the_balance():
    config = PurchaseConfig(capital=100_000.0, cash_reserve=1_000.0, pct_fee=0.001,
                            fixed_fee=5.0, integer_units=True)
    balance = build_purchase_balance(EQUAL, PRICES, NAMES, config)
    policy = LiquidityPolicy(pct_fee=0.001, fixed_fee=5.0, cash_reserve_initial=1_000.0)
    position = build_initial_position(100_000.0, EQUAL, PRICES, policy,
                                      integer_units=True, from_balance=balance)
    assert position.units == pytest.approx(
        balance.rows['target_units'].to_numpy(dtype=float), abs=1e-12)
    assert position.cash == pytest.approx(balance.leftover_cash, abs=1e-12)
    assert position.purchase_costs == pytest.approx(balance.total_costs, abs=1e-12)
    assert position.source == 'purchase_balance'


def test_initial_position_raises_when_capital_is_consumed():
    policy = LiquidityPolicy(pct_fee=0.001, fixed_fee=500.0, cash_reserve_initial=100.0)
    with pytest.raises(InfeasibleConstraintsError) as excinfo:
        build_initial_position(1_000.0, EQUAL, PRICES, policy)
    assert excinfo.value.code == 'initial_position_capital_consumed'


def test_integer_units_position_respects_the_budget():
    policy = LiquidityPolicy(pct_fee=0.001, fixed_fee=5.0)
    position = build_initial_position(50_000.0, EQUAL, PRICES, policy,
                                      integer_units=True)
    units = np.asarray(position.units)
    assert units == pytest.approx(np.floor(units), abs=1e-12)
    assert position.invested + position.purchase_costs <= 50_000.0 + 1e-9
    assert position.cash >= -1e-9


# --------------------------------------------------------------------------- #
# Calendar
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('frequency, months', [('monthly', 1), ('quarterly', 3),
                                               ('semiannual', 6), ('annual', 12)])
def test_schedule_frequencies_with_growth(frequency, months):
    schedule = build_schedule(pd.Timestamp('2024-02-15'), frequency, 5, 1000.0, 0.05)
    expected = 1000.0 * 1.05 ** np.arange(5)
    assert np.asarray(schedule.amounts) == pytest.approx(expected, rel=1e-12)
    spacing = [(schedule.dates[k] - schedule.dates[k - 1]).days for k in range(1, 5)]
    assert all(months * 27 <= gap <= months * 32 for gap in spacing)


def test_unknown_frequency_raises():
    with pytest.raises(ScheduleError) as excinfo:
        build_schedule(pd.Timestamp('2024-02-15'), 'weekly', 3, 100.0, 0.0)
    assert excinfo.value.code == 'unknown_frequency'


def test_weekend_installment_is_evaluated_on_the_next_business_day():
    frame = pd.DataFrame({'date': ['2024-02-03'], 'amount': [500.0]})
    schedule = schedule_from_frame(frame)
    _, _, effective = map_payment_indices(VALUATION, schedule)
    assert effective[0] == pd.Timestamp('2024-02-05')


def test_two_installments_on_the_same_business_day_are_aggregated():
    frame = pd.DataFrame({'date': ['2024-02-03', '2024-02-04'],
                          'amount': [500.0, 700.0],
                          'external_income': [10.0, 20.0]})
    schedule = schedule_from_frame(frame)
    payment_indices, horizon, effective = map_payment_indices(VALUATION, schedule)
    assert payment_indices.tolist() == [3, 3]
    policy = LiquidityPolicy(pct_fee=0.0, fixed_fee=0.0)
    initial = build_initial_position(50_000.0, EQUAL, PRICES, policy)
    ledger, result = simulate_cashflow_single(np.zeros((horizon, 5)), initial, schedule,
                                              payment_indices, policy, RF_SIMPLE_DAILY,
                                              None, None, effective)
    assert len(ledger) == 1
    assert ledger['due'].iloc[0] == pytest.approx(1200.0, abs=1e-12)
    assert ledger['external_income'].iloc[0] == pytest.approx(30.0, abs=1e-12)
    assert TWO_INSTALLMENTS_SAME_DAY in result.note_codes


def test_schedule_from_frame_reports_the_offending_row():
    frame = pd.DataFrame({'date': ['2024-02-05', '2024-03-05'], 'amount': [100.0, -5.0]})
    with pytest.raises(ScheduleError) as excinfo:
        schedule_from_frame(frame)
    assert excinfo.value.code == 'schedule_amount_invalid'
    assert excinfo.value.ctx['row'] == 1

    frame = pd.DataFrame({'date': ['2024-03-05', '2024-02-05'], 'amount': [100.0, 100.0]})
    with pytest.raises(ScheduleError) as excinfo:
        schedule_from_frame(frame)
    assert excinfo.value.code == 'schedule_dates_not_increasing'
    assert excinfo.value.ctx['row'] == 1


def test_schedule_tuple_roundtrip(schedule_simple):
    restored = InstallmentSchedule.from_tuple(schedule_simple.to_tuple())
    assert restored.dates.equals(schedule_simple.dates)
    assert np.asarray(restored.amounts) == pytest.approx(
        np.asarray(schedule_simple.amounts), abs=1e-12)


def test_holding_period_extends_the_horizon_and_the_final_wealth():
    base = Setup(capital=70_000.0, n_installments=4, amount=1000.0)
    extended = Setup(capital=70_000.0, n_installments=4, amount=1000.0,
                     holding_period_days=90)
    assert extended.horizon > base.horizon
    assert extended.calendar[-1] > base.calendar[-1]
    _, base_result = base.run(base.flat_path(0.0004))
    _, extended_result = extended.run(extended.flat_path(0.0004))
    assert extended_result.final_wealth[0] > base_result.final_wealth[0]
    assert base_result.total_paid[0] == pytest.approx(extended_result.total_paid[0],
                                                      abs=1e-9)


# --------------------------------------------------------------------------- #
# Rebalancing does not change the payments
# --------------------------------------------------------------------------- #
def test_rebalancing_leaves_payments_untouched_and_shows_up_in_costs():
    setup = Setup(capital=120_000.0, n_installments=6, amount=1000.0, pct_fee=0.001,
                  fixed_fee=1.0)
    daily = [0.0005, 0.0001, -0.0002, 0.0003, 0.0007]
    target = np.array([0.3, 0.3, 0.2, 0.1, 0.1])
    plain_ledger, plain = setup.run(setup.flat_path(daily))
    rebalance_indices = generate_rebalance_indices(setup.calendar, 'M', 0)
    reb_ledger, rebalanced = setup.run(setup.flat_path(daily),
                                       rebalance_indices=rebalance_indices,
                                       target_weights=target)
    assert reb_ledger['paid'].to_numpy() == pytest.approx(
        plain_ledger['paid'].to_numpy(), abs=1e-9)
    assert reb_ledger['deficit'].to_numpy() == pytest.approx(
        plain_ledger['deficit'].to_numpy(), abs=1e-9)
    assert rebalanced.total_rebalance_costs[0] > 0.0
    assert rebalanced.total_costs[0] > plain.total_costs[0]
    assert_aggregate_identity(rebalanced, setup.initial)


def test_rebalance_without_target_weights_is_a_programming_error():
    setup = Setup(n_installments=3)
    with pytest.raises(PortfolioError) as excinfo:
        setup.run(setup.zero_path(), rebalance_indices=np.array([2, 5]))
    assert excinfo.value.code == 'cashflow_rebalance_without_target_weights'


# --------------------------------------------------------------------------- #
# Additional capital (CA-22)
# --------------------------------------------------------------------------- #
def test_required_additional_capital_is_zero_when_already_covered():
    setup = Setup(capital=100_000.0, n_installments=6, amount=1000.0)
    extra = required_additional_capital(setup.capital, setup.weights, setup.prices,
                                        setup.schedule, setup.payment_indices,
                                        setup.zero_path(), setup.policy,
                                        RF_SIMPLE_DAILY)
    assert extra == 0.0


def test_required_additional_capital_covers_when_reinjected():
    setup = Setup(capital=3_000.0, n_installments=8, amount=1000.0, pct_fee=0.001,
                  fixed_fee=1.0)
    path = setup.zero_path()
    extra = required_additional_capital(setup.capital, setup.weights, setup.prices,
                                        setup.schedule, setup.payment_indices, path,
                                        setup.policy, RF_SIMPLE_DAILY)
    assert np.isfinite(extra) and extra > 0.0
    boosted = build_initial_position(setup.capital + extra, setup.weights,
                                     setup.prices, setup.policy)
    _, result = simulate_cashflow_single(path, boosted, setup.schedule,
                                         setup.payment_indices, setup.policy,
                                         RF_SIMPLE_DAILY, None, None, setup.effective)
    assert result.cumulative_deficit[0] <= 1e-6
    assert bool(result.covered_all[0]) is True


def test_required_additional_capital_shares_the_rebalancing_configuration():
    setup = Setup(capital=5_000.0, n_installments=6, amount=1000.0, pct_fee=0.002,
                  fixed_fee=2.0)
    rebalance_indices = generate_rebalance_indices(setup.calendar, 'M', 0)
    target = np.array([0.25, 0.25, 0.2, 0.15, 0.15])
    path = setup.flat_path(0.0)
    extra = required_additional_capital(
        setup.capital, setup.weights, setup.prices, setup.schedule,
        setup.payment_indices, path, setup.policy, RF_SIMPLE_DAILY,
        rebalance_indices=rebalance_indices, target_weights=target)
    assert np.isfinite(extra) and extra > 0.0
    boosted = build_initial_position(setup.capital + extra, setup.weights,
                                     setup.prices, setup.policy)
    _, result = simulate_cashflow_single(path, boosted, setup.schedule,
                                         setup.payment_indices, setup.policy,
                                         RF_SIMPLE_DAILY, rebalance_indices, target,
                                         setup.effective)
    assert result.cumulative_deficit[0] <= 1e-6


# --------------------------------------------------------------------------- #
# Vectorised engine
# --------------------------------------------------------------------------- #
def test_vectorised_engine_matches_the_single_path_wrapper():
    setup = Setup(capital=40_000.0, n_installments=6, amount=1200.0, pct_fee=0.001,
                  fixed_fee=2.0)
    path = setup.flat_path([0.0004, 0.0001, -0.0001, 0.0002, 0.0005])
    ledger, single = setup.run(path)
    batched = simulate_cashflow_paths([path[None, :, :]], setup.initial, setup.schedule,
                                      setup.payment_indices, setup.policy,
                                      RF_SIMPLE_DAILY, debug_identity=True)
    assert batched.final_wealth[0] == pytest.approx(single.final_wealth[0], abs=1e-12)
    assert batched.wealth_at_payments[0] == pytest.approx(
        ledger['wealth_after'].to_numpy(), abs=1e-12)
    assert batched.coverage_by_installment == pytest.approx(
        ledger['covered'].to_numpy().astype(float), abs=1e-12)
