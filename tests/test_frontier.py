"""Tests of core/frontier.py (design.md section 21, row ``test_frontier.py``).

Covers CA-9 (every frontier point feasible, returns non-decreasing above the
minimum-variance point), CA-10 (Monte Carlo weights sum to one and respect the
active bounds, with the acceptance rate reported) and CA-24 (reproducibility by
seed), plus the equality between the vectorized evaluator and the three scalar
formulas at 1e-12.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.frontier import (SAMPLING_METHODS, build_frontier, evaluate_weight_cloud,
                           sample_random_weights, _target_range)
from core.optimize import check_feasibility, solve_min_variance_target
from core.rng import child_rng
from core.stats import portfolio_return, portfolio_sharpe, portfolio_volatility
from core.types import Constraints


# --------------------------------------------------------------------------- #
# Frontier (CA-9)
# --------------------------------------------------------------------------- #
def test_frontier_points_are_feasible(stats, convention, constraints_long_only):
    frontier = build_frontier(stats.mu_annual, stats.cov_annual, constraints_long_only,
                              convention.rf_cc_annual, 20)
    assert frontier.computed > 0
    assert frontier.computed <= frontier.requested
    assert len(frontier.skipped_targets) == frontier.requested - frontier.computed
    weight_columns = [f'w_{i}' for i in range(len(stats.names))]
    weights = frontier.points[weight_columns].to_numpy(dtype=float)
    for row in weights:
        assert check_feasibility(row, constraints_long_only).ok is True


def test_frontier_returns_are_monotone_above_min_variance(stats, convention,
                                                          constraints_long_only):
    frontier = build_frontier(stats.mu_annual, stats.cov_annual, constraints_long_only,
                              convention.rf_cc_annual, 25)
    points = frontier.points.sort_values('vol').reset_index(drop=True)
    start = int(points['vol'].idxmin())
    upper = points.loc[start:, 'ret'].to_numpy(dtype=float)
    efficient = upper[upper >= upper[0] - 1e-12]
    assert np.all(np.diff(efficient) >= -1e-8)


def test_frontier_is_ordered_by_volatility(stats, convention, constraints_long_only):
    frontier = build_frontier(stats.mu_annual, stats.cov_annual, constraints_long_only,
                              convention.rf_cc_annual, 15)
    volatility = frontier.points['vol'].to_numpy(dtype=float)
    assert np.all(np.diff(volatility) >= -1e-12)


def test_binding_cap_reports_skipped_targets(stats, convention):
    constraints = Constraints(max_weight=0.3)
    frontier = build_frontier(stats.mu_annual, stats.cov_annual, constraints,
                              convention.rf_cc_annual, 20)
    assert frontier.computed < frontier.requested
    assert len(frontier.skipped_targets) > 0
    weight_columns = [f'w_{i}' for i in range(len(stats.names))]
    weights = frontier.points[weight_columns].to_numpy(dtype=float)
    assert (weights <= 0.3 + 1e-6).all()


def test_continuation_matches_independent_multistart(stats, convention,
                                                     constraints_long_only):
    """The warm-started curve is the same curve a cold multistart produces (1e-8)."""
    n_points = 12
    frontier = build_frontier(stats.mu_annual, stats.cov_annual, constraints_long_only,
                              convention.rf_cc_annual, n_points)
    low, high = _target_range(np.asarray(stats.mu_annual, dtype=float),
                             constraints_long_only)
    cold = []
    for target in np.linspace(low, high, n_points):
        result = solve_min_variance_target(stats.mu_annual, stats.cov_annual,
                                           float(target), constraints_long_only)
        if result.success:
            cold.append(portfolio_volatility(result.weights, stats.cov_annual))
    assert len(cold) == frontier.computed
    assert np.allclose(np.sort(np.array(cold)),
                       frontier.points['vol'].to_numpy(dtype=float), atol=1e-8)


# --------------------------------------------------------------------------- #
# Weight cloud (CA-10, CA-24)
# --------------------------------------------------------------------------- #
def test_cloud_without_cap_is_dirichlet(constraints_long_only):
    sample = sample_random_weights(5, 2000, constraints_long_only,
                                   child_rng(42, 'weights'))
    assert sample.method == 'dirichlet'
    assert sample.acceptance_rate == 1.0
    assert sample.weights.shape == (2000, 5)
    assert np.allclose(sample.weights.sum(axis=1), 1.0, atol=1e-10)
    assert (sample.weights >= 0.0).all()


def test_cloud_with_cap_respects_bounds_and_reports_acceptance():
    constraints = Constraints(max_weight=0.3)
    sample = sample_random_weights(5, 1500, constraints, child_rng(42, 'weights'))
    assert sample.method in SAMPLING_METHODS
    assert sample.message == ''
    assert sample.weights.shape[1] == 5
    assert sample.weights.shape[0] > 0
    assert 0.0 < sample.acceptance_rate <= 1.0
    assert np.all(np.abs(sample.weights.sum(axis=1) - 1.0) <= 1e-10)
    assert (sample.weights <= 0.3 + 1e-12).all()
    assert (sample.weights >= -1e-12).all()


def test_cloud_with_shorts_and_narrow_bounds_terminates():
    constraints = Constraints(allow_short=True, max_weight=0.4, short_lower_bound=-0.3)
    sample = sample_random_weights(5, 1000, constraints, child_rng(7, 'weights'))
    assert sample.method in SAMPLING_METHODS
    if sample.weights.shape[0] == 0:
        assert sample.weights.shape == (0, 5)
        assert sample.message != ''
    else:
        assert sample.message == ''
        assert np.all(np.abs(sample.weights.sum(axis=1) - 1.0) <= 1e-10)
        assert (sample.weights <= 0.4 + 1e-12).all()
        assert (sample.weights >= -0.3 - 1e-12).all()


def test_same_seed_produces_the_same_cloud():
    constraints = Constraints(max_weight=0.35)
    first = sample_random_weights(5, 800, constraints, child_rng(11, 'weights'))
    second = sample_random_weights(5, 800, constraints, child_rng(11, 'weights'))
    assert np.array_equal(first.weights, second.weights)
    assert first.method == second.method
    assert first.acceptance_rate == second.acceptance_rate


# --------------------------------------------------------------------------- #
# The vectorized counterpart is tied to the scalar formulas
# --------------------------------------------------------------------------- #
def test_vectorized_cloud_matches_scalar_formulas(stats, convention,
                                                  constraints_long_only):
    rf = convention.rf_cc_annual
    cloud = sample_random_weights(len(stats.names), 200, constraints_long_only,
                                  child_rng(3, 'weights')).weights
    vol, ret, sharpe = evaluate_weight_cloud(cloud, stats.mu_annual,
                                             stats.cov_annual, rf)
    scalar_vol = np.array([portfolio_volatility(w, stats.cov_annual) for w in cloud])
    scalar_ret = np.array([portfolio_return(w, stats.mu_annual) for w in cloud])
    scalar_sharpe = np.array([portfolio_sharpe(w, stats.mu_annual, stats.cov_annual, rf)
                              for w in cloud])
    assert np.allclose(vol, scalar_vol, atol=1e-12)
    assert np.allclose(ret, scalar_ret, atol=1e-12)
    assert np.allclose(sharpe, scalar_sharpe, atol=1e-12)

    frontier = build_frontier(stats.mu_annual, stats.cov_annual, constraints_long_only,
                              rf, 15)
    weight_columns = [f'w_{i}' for i in range(len(stats.names))]
    weights = frontier.points[weight_columns].to_numpy(dtype=float)
    assert np.allclose(frontier.points['vol'].to_numpy(dtype=float),
                       [portfolio_volatility(w, stats.cov_annual) for w in weights],
                       atol=1e-12)
    assert np.allclose(frontier.points['ret'].to_numpy(dtype=float),
                       [portfolio_return(w, stats.mu_annual) for w in weights],
                       atol=1e-12)
    assert np.allclose(frontier.points['sharpe'].to_numpy(dtype=float),
                       [portfolio_sharpe(w, stats.mu_annual, stats.cov_annual, rf)
                        for w in weights],
                       atol=1e-12)


def test_empty_request_returns_an_empty_cloud(constraints_long_only):
    sample = sample_random_weights(5, 0, constraints_long_only, child_rng(1, 'weights'))
    assert sample.weights.shape == (0, 5)
    assert sample.requested == 0


def test_evaluate_weight_cloud_accepts_a_single_vector(stats, convention):
    weights = np.full(len(stats.names), 1.0 / len(stats.names))
    vol, ret, sharpe = evaluate_weight_cloud(weights, stats.mu_annual,
                                             stats.cov_annual,
                                             convention.rf_cc_annual)
    assert vol.shape == (1,)
    assert ret[0] == pytest.approx(portfolio_return(weights, stats.mu_annual))
    assert sharpe[0] == pytest.approx(
        portfolio_sharpe(weights, stats.mu_annual, stats.cov_annual,
                         convention.rf_cc_annual))
