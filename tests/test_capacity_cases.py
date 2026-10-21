"""The seven mandatory payment-capacity cases (design.md section 21) plus the
Monte Carlo consistency check.

Every case builds its own calendar and its own returns; there is no randomness
except in the two cases that explicitly need a cloud of paths. Criteria covered:
CA-26 .. CA-32 and CA-23.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.cashflow import (LiquidityPolicy, build_initial_position, build_schedule,
                           map_payment_indices, required_additional_capital,
                           simulate_cashflow_paths, simulate_cashflow_single)
from core.rng import child_rng
from core.scenarios import (aggregate_capacity_metrics, build_scenarios,
                            classify_payment_capacity, deterministic_path,
                            iter_path_batches, plan_path_simulation)
from core.types import FrequencyConvention
# tests/ has no __init__.py; pytest puts the directory on sys.path, so the
# helper of the sibling module is importable by module name.
from test_scenarios import make_stats

PRICES = np.array([25.0, 48.0, 120.0, 310.0, 7.0])
EQUAL = np.full(5, 0.2)
VALUATION = pd.Timestamp('2024-01-31')
CONVENTION = FrequencyConvention(trading_days=252, rf_annual_effective=0.045)
RF_SIMPLE_DAILY = CONVENTION.rf_simple_daily


def build_case(capital, n_installments, amount, *, growth_rate=0.0, pct_fee=0.0,
               fixed_fee=0.0, cash_reserve_initial=0.0, frequency='monthly'):
    """Return ``(policy, schedule, payment_indices, horizon, effective, initial)``."""
    policy = LiquidityPolicy(pct_fee=pct_fee, fixed_fee=fixed_fee,
                             cash_reserve_initial=cash_reserve_initial)
    schedule = build_schedule(pd.Timestamp('2024-02-15'), frequency, n_installments,
                              amount, growth_rate)
    payment_indices, horizon, effective = map_payment_indices(VALUATION, schedule)
    initial = build_initial_position(capital, EQUAL, PRICES, policy)
    return policy, schedule, payment_indices, horizon, effective, initial


def run_flat(capital, n_installments, amount, daily_log_return=0.0, **kwargs):
    """Run one deterministic path with a constant daily log return."""
    policy, schedule, payment_indices, horizon, effective, initial = build_case(
        capital, n_installments, amount, **kwargs)
    path = np.full((horizon, PRICES.size), float(daily_log_return))
    ledger, result = simulate_cashflow_single(path, initial, schedule,
                                              payment_indices, policy,
                                              RF_SIMPLE_DAILY, None, None, effective)
    return ledger, result, initial, schedule, payment_indices, policy, path


# --------------------------------------------------------------------------- #
# Case 1 (CA-26)
# --------------------------------------------------------------------------- #
def test_case1_zero_returns_sufficient():
    """Zero returns and capital above the calendar plus its costs.

    ``final_wealth == capital - sum(paid) - purchase_costs - sale_costs`` to 1e-9,
    the explicit form of design.md section 10.2b.
    """
    capital = 100_000.0
    ledger, result, initial, _, _, _, _ = run_flat(capital, 12, 1000.0,
                                                   pct_fee=0.002, fixed_fee=5.0)
    assert bool(result.covered_all[0]) is True
    assert int(result.n_defaults[0]) == 0
    assert int(result.first_default_idx[0]) == -1
    assert result.cumulative_deficit[0] == pytest.approx(0.0, abs=1e-12)
    expected = (capital - result.total_paid[0] - initial.purchase_costs
                - result.total_sale_costs[0])
    assert result.final_wealth[0] == pytest.approx(expected, abs=1e-9)
    assert result.total_paid[0] == pytest.approx(12_000.0, abs=1e-9)
    assert ledger['covered'].all()


# --------------------------------------------------------------------------- #
# Case 2 (CA-27)
# --------------------------------------------------------------------------- #
def test_case2_insufficient_from_start():
    """Capital below the first installment: default at index 0, and the estimated
    additional capital really removes the deficit when reinjected."""
    capital = 500.0
    ledger, result, initial, schedule, payment_indices, policy, path = run_flat(
        capital, 12, 1000.0)
    assert int(result.first_default_idx[0]) == 0
    assert ledger['deficit'].iloc[0] > 0.0
    assert bool(result.covered_all[0]) is False

    extra = required_additional_capital(capital, EQUAL, PRICES, schedule,
                                        payment_indices, path, policy,
                                        RF_SIMPLE_DAILY)
    assert np.isfinite(extra) and extra > 0.0

    boosted = build_initial_position(capital + extra, EQUAL, PRICES, policy)
    _, after = simulate_cashflow_single(path, boosted, schedule, payment_indices,
                                        policy, RF_SIMPLE_DAILY)
    assert after.cumulative_deficit[0] <= 1e-6
    assert bool(after.covered_all[0]) is True


# --------------------------------------------------------------------------- #
# Case 3 (CA-28)
# --------------------------------------------------------------------------- #
def test_case3_central_ok_conservative_fails():
    """mu and sigma such that the central scenario covers and ``mu - sigma`` does not."""
    stats = make_stats(np.full(5, 0.40), np.full(5, 0.50))
    policy = LiquidityPolicy(pct_fee=0.0, fixed_fee=0.0)
    schedule = build_schedule(pd.Timestamp('2024-02-15'), 'monthly', 12, 1200.0, 0.0)
    payment_indices, horizon, effective = map_payment_indices(VALUATION, schedule)
    initial = build_initial_position(12_500.0, EQUAL, PRICES, policy)
    scenarios = build_scenarios(stats, EQUAL, 1.0)
    spec = plan_path_simulation(stats.cov_daily, horizon, 300,
                                method='gaussian_cholesky', block_size=21, seed=42)
    batches = iter_path_batches(stats.mu_daily, stats.cov_daily, horizon, 300,
                                rng=child_rng(42, 'paths'))
    metrics = aggregate_capacity_metrics(stats, EQUAL, initial, schedule,
                                         payment_indices, policy, scenarios, batches,
                                         spec, CONVENTION,
                                         effective_payment_dates=effective)
    assert metrics.central_covers_all is True
    assert metrics.conservative_covers_all is False
    assert metrics.conservative_deficit_ratio > 0.0
    verdict = classify_payment_capacity(metrics)
    assert verdict.category != 'sufficient'


# --------------------------------------------------------------------------- #
# Case 4 (CA-29)
# --------------------------------------------------------------------------- #
def test_case4_covered_only_by_selling():
    """Zero initial cash and everything invested: sales with costs cover every
    installment and the gross sold exceeds the amount paid."""
    ledger, result, initial, _, _, _, _ = run_flat(100_000.0, 12, 1000.0,
                                                   pct_fee=0.002, fixed_fee=1.0)
    assert initial.cash == pytest.approx(0.0, abs=1e-9)
    assert bool(result.covered_all[0]) is True
    assert result.total_sale_costs[0] > 0.0
    assert result.total_gross_sold[0] > result.total_paid[0]
    assert (ledger['gross_sold'].to_numpy() > ledger['paid'].to_numpy()).all()
    assert (ledger['sale_costs'].to_numpy() > 0.0).all()


# --------------------------------------------------------------------------- #
# Case 5 (CA-30)
# --------------------------------------------------------------------------- #
def test_case5_growing_installments():
    """5 % growth per installment, and coverage never increases with the index.

    Once a path exhausts its wealth it stays exhausted, so
    ``coverage_by_installment`` is non-increasing and any default concentrates in
    the final installments.
    """
    stats = make_stats(np.full(5, 0.05), np.full(5, 0.30))
    policy = LiquidityPolicy(pct_fee=0.0, fixed_fee=0.0)
    schedule = build_schedule(pd.Timestamp('2024-02-15'), 'monthly', 12, 1000.0, 0.05)
    assert np.asarray(schedule.amounts) == pytest.approx(
        1000.0 * 1.05 ** np.arange(12), rel=1e-12)

    payment_indices, horizon, effective = map_payment_indices(VALUATION, schedule)
    initial = build_initial_position(16_000.0, EQUAL, PRICES, policy)
    scenarios = build_scenarios(stats, EQUAL, 1.0)
    spec = plan_path_simulation(stats.cov_daily, horizon, 400,
                                method='gaussian_cholesky', block_size=21, seed=42)
    batches = iter_path_batches(stats.mu_daily, stats.cov_daily, horizon, 400,
                                rng=child_rng(42, 'paths'))
    metrics = aggregate_capacity_metrics(stats, EQUAL, initial, schedule,
                                         payment_indices, policy, scenarios, batches,
                                         spec, CONVENTION,
                                         effective_payment_dates=effective)
    coverage = np.asarray(metrics.coverage_by_installment)
    assert coverage.shape == (12,)
    assert np.all(np.diff(coverage) <= 1e-12)
    assert coverage[-1] < coverage[0]
    assert coverage[-1] < 1.0


# --------------------------------------------------------------------------- #
# Case 6 (CA-31)
# --------------------------------------------------------------------------- #
def test_case6_costs_reduce_capital():
    """With EXACTLY zero returns the comparison is exact and needs no compounding
    factor: ``final_wealth(0) - final_wealth(pct_fee) == total_costs(pct_fee)``,
    where ``total_costs`` adds ``purchase_costs`` to the sale costs.
    """
    free_ledger, free, free_initial, _, _, _, _ = run_flat(100_000.0, 12, 1000.0,
                                                           pct_fee=0.0, fixed_fee=0.0)
    paid_ledger, charged, charged_initial, _, _, _, _ = run_flat(
        100_000.0, 12, 1000.0, pct_fee=0.003, fixed_fee=0.0)
    assert bool(free.covered_all[0]) is True
    assert bool(charged.covered_all[0]) is True
    assert free.total_paid[0] == pytest.approx(charged.total_paid[0], abs=1e-9)
    assert free_initial.purchase_costs == pytest.approx(0.0, abs=1e-12)
    assert free.total_sale_costs[0] == pytest.approx(0.0, abs=1e-12)

    total_costs = charged_initial.purchase_costs + charged.total_sale_costs[0]
    assert total_costs > 0.0
    assert free.final_wealth[0] - charged.final_wealth[0] == pytest.approx(
        total_costs, abs=1e-9)


# --------------------------------------------------------------------------- #
# Case 7 (CA-32)
# --------------------------------------------------------------------------- #
def test_case7_wealth_exhausted():
    """Wealth exhausted halfway: the remaining installments carry ``paid = 0`` and
    ``deficit = due``, wealth stays exactly 0 (no returns on fictitious capital,
    even though the path has a positive drift), nothing goes negative, and
    ``cumulative_deficit`` equals the unpaid sum.
    """
    ledger, result, _, _, _, _, _ = run_flat(5_000.0, 10, 1000.0,
                                             daily_log_return=0.0005)
    first = int(result.first_default_idx[0])
    assert 0 < first < 9

    tail = ledger.iloc[first + 1:]
    assert (tail['paid'].to_numpy() == 0.0).all()
    assert tail['deficit'].to_numpy() == pytest.approx(tail['due'].to_numpy(),
                                                       abs=1e-12)
    assert (ledger['wealth_after'].to_numpy()[first:] == 0.0).all()
    assert (ledger['mv_after'].to_numpy()[first:] == 0.0).all()
    assert (ledger['cash_after'].to_numpy() >= 0.0).all()
    assert (ledger['mv_before'].to_numpy() >= 0.0).all()
    assert result.final_wealth[0] == 0.0

    unpaid = float((ledger['due'].to_numpy() - ledger['paid'].to_numpy()).sum())
    assert result.cumulative_deficit[0] == pytest.approx(unpaid, abs=1e-9)
    assert int(result.n_defaults[0]) == 10 - first


# --------------------------------------------------------------------------- #
# Monte Carlo consistency (CA-23)
# --------------------------------------------------------------------------- #
def test_case_mc_consistency():
    """500 paths: ``p_cover_all`` equals the mean of the recorded coverage, and the
    deterministic path injected into a batch reproduces the deterministic ledger.
    """
    stats = make_stats(np.full(5, 0.10), np.full(5, 0.25))
    policy = LiquidityPolicy(pct_fee=0.001, fixed_fee=1.0)
    schedule = build_schedule(pd.Timestamp('2024-02-15'), 'monthly', 12, 1300.0, 0.0)
    payment_indices, horizon, effective = map_payment_indices(VALUATION, schedule)
    initial = build_initial_position(18_000.0, EQUAL, PRICES, policy)
    scenarios = build_scenarios(stats, EQUAL, 1.0)
    spec = plan_path_simulation(stats.cov_daily, horizon, 500,
                                method='gaussian_cholesky', block_size=21, seed=42)
    batches = iter_path_batches(stats.mu_daily, stats.cov_daily, horizon, 500,
                                rng=child_rng(42, 'paths'))
    metrics = aggregate_capacity_metrics(stats, EQUAL, initial, schedule,
                                         payment_indices, policy, scenarios, batches,
                                         spec, CONVENTION,
                                         effective_payment_dates=effective)
    assert metrics.mc_covered_all.shape == (500,)
    assert metrics.p_cover_all == pytest.approx(
        float(metrics.mc_covered_all.mean()), abs=1e-12)

    # Coverage recomputed path by path from the recorded payments.
    central = metrics.deterministic['central']
    assert central.n_paths == 1

    path = deterministic_path(scenarios['central'].mu_annual_by_asset,
                              CONVENTION.trading_days, horizon)
    ledger, single = simulate_cashflow_single(path[0], initial, schedule,
                                              payment_indices, policy,
                                              RF_SIMPLE_DAILY, None, None, effective)
    assert single.final_wealth[0] == pytest.approx(central.final_wealth[0], abs=1e-12)

    # The same path injected as one row of a multi-path batch gives the same ledger.
    noise = next(iter_path_batches(stats.mu_daily, stats.cov_daily, horizon, 3,
                                   rng=child_rng(1, 'paths')))
    injected = np.concatenate([path, noise], axis=0)
    mixed = simulate_cashflow_paths([injected], initial, schedule, payment_indices,
                                    policy, RF_SIMPLE_DAILY, debug_identity=True)
    assert mixed.n_paths == 4
    assert mixed.wealth_at_payments[0] == pytest.approx(
        ledger['wealth_after'].to_numpy(), abs=1e-12)
    assert mixed.cash_at_payments[0] == pytest.approx(
        ledger['cash_after'].to_numpy(), abs=1e-12)
    assert mixed.final_wealth[0] == pytest.approx(single.final_wealth[0], abs=1e-12)
    assert bool(mixed.covered_all[0]) == bool(single.covered_all[0])
