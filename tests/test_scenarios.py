"""Tests of core/scenarios.py (design.md section 21, row ``test_scenarios.py``).

Covers CA-23 (the Monte Carlo and the deterministic engine agree), CA-24 (bitwise
reproducibility from the seed) and CA-25 (the four categories with their exact
thresholds).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.cashflow import (LiquidityPolicy, build_initial_position, build_schedule,
                           map_payment_indices, simulate_cashflow_paths)
from core.rng import child_rng
from core.scenarios import (CAPACITY_CATEGORIES, MAX_FLOATS, NO_SCHEDULED_OBLIGATIONS,
                            PERCENTILE_KEYS, SCENARIO_KEYS, CapacityMetrics,
                            CapacityThresholds, PathSimulationSpec,
                            aggregate_capacity_metrics, build_scenarios,
                            classify_payment_capacity, deterministic_path,
                            iter_path_batches, plan_path_simulation)
from core.stats import correlation_from_covariance, portfolio_volatility
from core.types import AssetStats, FrequencyConvention

PRICES = np.array([25.0, 48.0, 120.0, 310.0, 7.0])
EQUAL = np.full(5, 0.2)
VALUATION = pd.Timestamp('2024-01-31')
CONVENTION = FrequencyConvention(trading_days=252, rf_annual_effective=0.045)


def make_stats(mu_annual, sigma_annual, corr=None,
               convention: FrequencyConvention = CONVENTION) -> AssetStats:
    """An ``AssetStats`` with the moments the test wants, no estimation involved."""
    mu = np.asarray(mu_annual, dtype=float).ravel()
    sigma = np.asarray(sigma_annual, dtype=float).ravel()
    n = mu.size
    correlation = np.eye(n) if corr is None else np.asarray(corr, dtype=float)
    cov_annual = np.outer(sigma, sigma) * correlation
    tdy = float(convention.trading_days)
    return AssetStats(names=tuple(f'ACTIVO_{i}' for i in range(n)), n_obs=1_000,
                      mu_daily=mu / tdy, cov_daily=cov_annual / tdy, mu_annual=mu,
                      cov_annual=cov_annual, sigma_annual=sigma, corr=correlation,
                      sharpe_individual=np.zeros(n), min_return=np.zeros(n),
                      max_return=np.zeros(n), psd_repaired=False, ridge_epsilon=0.0,
                      convention=convention)


class Case:
    """A complete capacity run, assembled from a few knobs."""

    def __init__(self, capital=100_000.0, n_installments=12, amount=1000.0,
                 growth_rate=0.0, mu_annual=0.08, sigma_annual=0.20, corr=None,
                 n_paths=200, seed=42, k_sigma=1.0, pct_fee=0.0, fixed_fee=0.0,
                 method='gaussian_cholesky', block_size=21, historical_returns=None,
                 max_floats=MAX_FLOATS):
        self.stats = make_stats(np.full(5, mu_annual), np.full(5, sigma_annual), corr)
        self.policy = LiquidityPolicy(pct_fee=pct_fee, fixed_fee=fixed_fee)
        self.schedule = build_schedule(pd.Timestamp('2024-02-15'), 'monthly',
                                       n_installments, amount, growth_rate)
        self.payment_indices, self.horizon, self.effective = map_payment_indices(
            VALUATION, self.schedule)
        self.initial = build_initial_position(capital, EQUAL, PRICES, self.policy)
        self.scenarios = build_scenarios(self.stats, EQUAL, k_sigma)
        self.spec = plan_path_simulation(self.stats.cov_daily, self.horizon, n_paths,
                                         method=method, block_size=block_size,
                                         seed=seed, max_floats=max_floats)
        self.method = method
        self.block_size = block_size
        self.historical_returns = historical_returns
        self.n_paths = n_paths
        self.seed = seed
        self.max_floats = max_floats

    def batches(self):
        return iter_path_batches(self.stats.mu_daily, self.stats.cov_daily,
                                 self.horizon, self.n_paths, method=self.method,
                                 rng=child_rng(self.seed, 'paths'),
                                 block_size=self.block_size,
                                 historical_returns=self.historical_returns,
                                 max_floats=self.max_floats)

    def metrics(self) -> CapacityMetrics:
        return aggregate_capacity_metrics(
            self.stats, EQUAL, self.initial, self.schedule, self.payment_indices,
            self.policy, self.scenarios, self.batches(), self.spec, CONVENTION,
            effective_payment_dates=self.effective)


def make_metrics(p_cover_all, *, central=True, conservative=True, margin_ratio=1.0,
                 conservative_deficit_ratio=0.0, note_codes=()) -> CapacityMetrics:
    """A ``CapacityMetrics`` with only the fields the classifier reads."""
    spec = PathSimulationSpec(method='gaussian_cholesky', psd_repaired=False,
                              block_size=None, n_paths=10, horizon=10, n_assets=5,
                              batch_size=10, seed=42)
    return CapacityMetrics(initial_capital=1.0, n_installments=1, total_scheduled=1.0,
                           deterministic={}, central_ledger=pd.DataFrame(),
                           central_covers_all=central,
                           conservative_covers_all=conservative,
                           conservative_deficit_ratio=conservative_deficit_ratio,
                           margin_ratio=margin_ratio, p_cover_all=p_cover_all,
                           coverage_by_installment=np.ones(1),
                           mc_final_wealth=np.ones(10),
                           mc_covered_all=np.ones(10, dtype=bool),
                           final_wealth_percentiles={}, additional_capital_required=0.0,
                           path_spec=spec, note_codes=note_codes)


# --------------------------------------------------------------------------- #
# Deterministic scenarios (section 11.2)
# --------------------------------------------------------------------------- #
def test_build_scenarios_shifts_mu_by_k_sigma(stats):
    scenarios = build_scenarios(stats, EQUAL, 1.0)
    assert tuple(scenarios) == SCENARIO_KEYS
    mu = np.asarray(stats.mu_annual)
    sigma = np.asarray(stats.sigma_annual)
    assert np.asarray(scenarios['conservative'].mu_annual_by_asset) == pytest.approx(
        mu - sigma, rel=1e-12)
    assert np.asarray(scenarios['central'].mu_annual_by_asset) == pytest.approx(
        mu, rel=1e-12)
    assert np.asarray(scenarios['favorable'].mu_annual_by_asset) == pytest.approx(
        mu + sigma, rel=1e-12)
    reference = portfolio_volatility(EQUAL, stats.cov_annual)
    for key in SCENARIO_KEYS:
        # Taken from core.stats.portfolio_volatility, never a local formula, and
        # identical in the three scenarios because they only shift mu.
        assert scenarios[key].vol_portfolio_annual == pytest.approx(reference,
                                                                    rel=1e-12)
        assert not hasattr(scenarios[key], 'label')


def test_deterministic_path_spreads_the_return_uniformly(stats):
    scenarios = build_scenarios(stats, EQUAL, 1.0)
    mu = np.asarray(scenarios['favorable'].mu_annual_by_asset)
    path = deterministic_path(mu, 252, 30)
    assert path.shape == (1, 30, mu.size)
    assert path[0] == pytest.approx(np.repeat((mu / 252.0)[None, :], 30, axis=0),
                                    rel=1e-12)
    assert float(path[0].std()) > 0.0 or mu.std() == 0.0


def test_k_sigma_zero_collapses_scenarios(stats):
    scenarios = build_scenarios(stats, EQUAL, 0.0)
    central = np.asarray(scenarios['central'].mu_annual_by_asset)
    for key in SCENARIO_KEYS:
        assert np.asarray(scenarios[key].mu_annual_by_asset) == pytest.approx(
            central, rel=1e-12)
    metrics = Case(k_sigma=0.0, n_paths=100).metrics()
    assert metrics.conservative_covers_all == metrics.central_covers_all


# --------------------------------------------------------------------------- #
# Path Monte Carlo (section 11.3)
# --------------------------------------------------------------------------- #
def test_monte_carlo_uses_the_historical_moments(stats):
    batches = iter_path_batches(stats.mu_daily, stats.cov_daily, 60, 4_000,
                                method='gaussian_cholesky',
                                rng=child_rng(42, 'paths'))
    paths = np.concatenate(list(batches))
    empirical = paths.reshape(-1, paths.shape[2]).mean(axis=0)
    assert empirical == pytest.approx(np.asarray(stats.mu_daily), abs=3e-4)
    conservative = (np.asarray(stats.mu_annual)
                    - np.asarray(stats.sigma_annual)) / 252.0
    assert not np.allclose(empirical, conservative, atol=3e-4)


def test_paths_preserve_covariance(stats):
    """Fixed numbers of design.md section 21: no statistical hand-waving."""
    cov_daily = np.asarray(stats.cov_daily)
    n = cov_daily.shape[0]
    batches = iter_path_batches(stats.mu_daily, cov_daily, 252, 20_000,
                                method='gaussian_cholesky',
                                rng=child_rng(42, 'paths'))
    sample = np.concatenate(list(batches)).reshape(-1, n)
    empirical = np.cov(sample, rowvar=False, ddof=1)
    assert np.allclose(empirical, cov_daily, rtol=0.05, atol=1e-8)
    corr_empirical = correlation_from_covariance(empirical)
    off_diagonal = np.abs(corr_empirical - np.eye(n))[~np.eye(n, dtype=bool)]
    assert float(off_diagonal.max()) > 0.05
    assert np.allclose(corr_empirical, correlation_from_covariance(cov_daily),
                       atol=0.03)


def test_block_bootstrap_preserves_covariance(stats, aligned):
    history = np.asarray(aligned.log_returns)
    cov_daily = np.asarray(stats.cov_daily)
    n = cov_daily.shape[0]
    batches = iter_path_batches(stats.mu_daily, cov_daily, 252, 20_000,
                                method='block_bootstrap',
                                rng=child_rng(42, 'bootstrap'), block_size=21,
                                historical_returns=history)
    sample = np.concatenate(list(batches)).reshape(-1, n)
    empirical = np.cov(sample, rowvar=False, ddof=1)
    assert np.allclose(empirical, cov_daily, rtol=0.10, atol=1e-8)
    corr_empirical = correlation_from_covariance(empirical)
    off_diagonal = np.abs(corr_empirical - np.eye(n))[~np.eye(n, dtype=bool)]
    assert float(off_diagonal.max()) > 0.05
    assert np.allclose(corr_empirical, correlation_from_covariance(cov_daily),
                       atol=0.03)


def test_block_bootstrap_rows_come_from_the_history(stats, aligned):
    history = np.asarray(aligned.log_returns)
    batches = iter_path_batches(stats.mu_daily, stats.cov_daily, 50, 5,
                                method='block_bootstrap',
                                rng=child_rng(7, 'bootstrap'), block_size=10,
                                historical_returns=history)
    sample = np.concatenate(list(batches)).reshape(-1, history.shape[1])
    known = {tuple(row) for row in history.tolist()}
    assert all(tuple(row) in known for row in sample.tolist())


def test_block_bootstrap_requires_historical_returns(stats):
    with pytest.raises(ValueError) as excinfo:
        iter_path_batches(stats.mu_daily, stats.cov_daily, 10, 5,
                          method='block_bootstrap', rng=child_rng(1, 'bootstrap'))
    assert str(excinfo.value) == 'block_bootstrap_requires_historical_returns'


@pytest.mark.parametrize('block_size', [0, 1, 300, 'too_long'])
def test_block_size_out_of_range(stats, block_size):
    history = child_rng(3, 'bootstrap').normal(0.0, 0.01, size=(120, 5))
    size = history.shape[0] + 1 if block_size == 'too_long' else block_size
    with pytest.raises(ValueError) as excinfo:
        iter_path_batches(stats.mu_daily, stats.cov_daily, 60, 4,
                          method='block_bootstrap', rng=child_rng(1, 'bootstrap'),
                          block_size=size, historical_returns=history)
    assert str(excinfo.value) == 'block_size_out_of_range'


def test_unknown_path_method_raises(stats):
    with pytest.raises(ValueError) as excinfo:
        iter_path_batches(stats.mu_daily, stats.cov_daily, 10, 5, method='garch',
                          rng=child_rng(1, 'paths'))
    assert str(excinfo.value) == 'unknown_path_method'


def test_psd_repair_has_a_single_origin(singular_stats):
    """A singular Sigma is repaired through ``core.stats.cholesky_factor`` only."""
    cov_daily = np.asarray(singular_stats.cov_daily_raw)
    spec = plan_path_simulation(cov_daily, 20, 50, method='gaussian_cholesky',
                                block_size=21, seed=42)
    assert spec.psd_repaired is True
    batch = next(iter_path_batches(singular_stats.mu_daily, cov_daily, 20, 50,
                                   rng=child_rng(42, 'paths')))
    assert batch.shape == (50, 20, 2)
    assert np.all(np.isfinite(batch))


def test_plan_path_simulation_decides_the_batch_size(stats):
    spec = plan_path_simulation(stats.cov_daily, 1_260, 50_000,
                                method='gaussian_cholesky', block_size=21, seed=42,
                                max_floats=MAX_FLOATS)
    assert spec.batch_size == max(1, MAX_FLOATS // (1_260 * 5))
    assert spec.block_size is None
    assert spec.horizon == 1_260 and spec.n_paths == 50_000 and spec.n_assets == 5
    bootstrap = plan_path_simulation(stats.cov_daily, 60, 100,
                                     method='block_bootstrap', block_size=30, seed=7)
    assert bootstrap.block_size == 30


def test_path_spec_tuple_roundtrip(stats):
    spec = plan_path_simulation(stats.cov_daily, 60, 100, method='block_bootstrap',
                                block_size=30, seed=7)
    assert PathSimulationSpec.from_tuple(spec.to_tuple()) == spec
    gaussian = plan_path_simulation(stats.cov_daily, 60, 100,
                                    method='gaussian_cholesky', block_size=21, seed=7)
    assert PathSimulationSpec.from_tuple(gaussian.to_tuple()) == gaussian


def test_same_seed_is_bit_identical(stats):
    first = np.concatenate(list(iter_path_batches(
        stats.mu_daily, stats.cov_daily, 40, 300, rng=child_rng(42, 'paths'))))
    second = np.concatenate(list(iter_path_batches(
        stats.mu_daily, stats.cov_daily, 40, 300, rng=child_rng(42, 'paths'))))
    assert np.array_equal(first, second)


def test_batching_does_not_change_results():
    """Several batches and a single batch give bit-identical results."""
    case = Case(n_paths=300, max_floats=MAX_FLOATS)
    small = Case(n_paths=300, max_floats=40 * 5)      # forces many batches
    assert small.spec.batch_size < case.spec.batch_size

    def run(item):
        return simulate_cashflow_paths(item.batches(), item.initial, item.schedule,
                                       item.payment_indices, item.policy,
                                       CONVENTION.rf_simple_daily)

    one = run(case)
    many = run(small)
    assert one.n_paths == many.n_paths == 300
    for field in ('covered_all', 'n_defaults', 'first_default_idx', 'max_deficit',
                  'cumulative_deficit', 'final_wealth', 'min_wealth', 'total_costs',
                  'total_paid', 'total_returns', 'coverage_by_installment',
                  'wealth_at_payments', 'cash_at_payments'):
        assert np.array_equal(getattr(one, field), getattr(many, field)), field


# --------------------------------------------------------------------------- #
# Capacity metrics and classification (section 11.4)
# --------------------------------------------------------------------------- #
def test_mc_arrays_back_the_reported_figures():
    metrics = Case(n_paths=400).metrics()
    assert metrics.mc_final_wealth.shape == (400,)
    assert metrics.mc_covered_all.shape == (400,)
    assert metrics.p_cover_all == pytest.approx(
        float(metrics.mc_covered_all.mean()), abs=1e-12)
    for key, percentile in zip(PERCENTILE_KEYS, (5.0, 25.0, 50.0, 75.0, 95.0)):
        assert metrics.final_wealth_percentiles[key] == pytest.approx(
            float(np.percentile(metrics.mc_final_wealth, percentile)), abs=1e-12)
    assert metrics.path_spec.n_paths == 400
    assert metrics.central_ledger.shape[0] == metrics.n_installments
    assert set(SCENARIO_KEYS) == set(metrics.deterministic)


def test_no_scheduled_obligations_is_flagged():
    metrics = Case(amount=0.0, n_paths=100).metrics()
    assert metrics.total_scheduled == 0.0
    assert metrics.margin_ratio == float('inf')
    assert metrics.conservative_deficit_ratio == 0.0
    assert NO_SCHEDULED_OBLIGATIONS in metrics.note_codes
    verdict = classify_payment_capacity(metrics)
    assert verdict.category == 'sufficient'
    assert verdict.rule_fired == 4
    for criterion in verdict.criteria:
        assert not (isinstance(criterion.value, float) and np.isnan(criterion.value))


def test_criteria_mapping_is_fixed():
    thresholds = CapacityThresholds()
    metrics = make_metrics(0.80, conservative=False, margin_ratio=0.05,
                           conservative_deficit_ratio=0.3)
    verdict = classify_payment_capacity(metrics, thresholds)
    expected = (
        ('p_cover_all', '>=', thresholds.p_sufficient),
        ('central_covers_all', '==', True),
        ('conservative_covers_all', '==', True),
        ('margin_ratio', '>=', thresholds.margin_min),
        ('conservative_deficit_ratio', '<=',
         thresholds.conservative_deficit_tolerance),
    )
    assert len(verdict.criteria) == 5
    for criterion, (name, comparator, threshold) in zip(verdict.criteria, expected):
        assert criterion.name == name
        assert criterion.comparator == comparator
        assert criterion.threshold == threshold
        if comparator == '>=':
            assert criterion.passed == (criterion.value >= threshold)
        elif comparator == '<=':
            assert criterion.passed == (criterion.value <= threshold)
        else:
            assert criterion.passed == (criterion.value == threshold)


@pytest.mark.parametrize('metrics, category, rule', [
    (make_metrics(0.99, central=False), 'insufficient', 1),
    (make_metrics(0.49), 'insufficient', 1),
    (make_metrics(0.50), 'high_risk', 2),
    (make_metrics(0.74), 'high_risk', 2),
    (make_metrics(0.99, conservative_deficit_ratio=0.11), 'high_risk', 2),
    (make_metrics(0.75), 'tight', 3),
    (make_metrics(0.89), 'tight', 3),
    (make_metrics(0.99, conservative=False), 'tight', 3),
    (make_metrics(0.99, margin_ratio=0.09), 'tight', 3),
    (make_metrics(0.90, margin_ratio=0.10), 'sufficient', 4),
    (make_metrics(1.0, margin_ratio=2.0), 'sufficient', 4),
])
def test_classification_at_every_threshold_boundary(metrics, category, rule):
    verdict = classify_payment_capacity(metrics, CapacityThresholds())
    assert verdict.category == category
    assert verdict.rule_fired == rule
    assert verdict.category in CAPACITY_CATEGORIES
    assert not hasattr(verdict, 'narrative')


def test_thresholds_tuple_roundtrip(thresholds):
    assert CapacityThresholds.from_tuple(thresholds.to_tuple()) == thresholds
    custom = CapacityThresholds(0.95, 0.8, 0.6, 0.2, 0.05)
    assert CapacityThresholds.from_tuple(custom.to_tuple()) == custom
