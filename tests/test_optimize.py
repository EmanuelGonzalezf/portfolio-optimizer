"""Tests of core/optimize.py (design.md section 21, row ``test_optimize.py``).

Covers CA-4 (feasibility), CA-5 (agreement with the closed forms when shorts are
allowed and no cap binds), CA-6 (minimum variance is the least volatile
portfolio) and CA-8 (a singular covariance is repaired instead of raising).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from core.frontier import build_frontier
from core.optimize import (check_feasibility, closed_form_min_variance,
                           closed_form_tangency, equal_weights, max_feasible_return,
                           solve_max_sharpe, solve_min_variance,
                           solve_min_variance_target)
from core.stats import (TANGENCY_UNDEFINED, portfolio_sharpe, portfolio_volatility,
                        summarize_portfolio)
from core.types import Constraints, FrequencyConvention, InfeasibleConstraintsError

# A small, well-conditioned problem whose closed forms sit inside [-1, 1].
TAME_COV = np.array([[0.04, 0.01, 0.00],
                     [0.01, 0.09, 0.02],
                     [0.00, 0.02, 0.16]])
TAME_MU = np.array([0.08, 0.11, 0.14])
TAME_RF = 0.04

WIDE_SHORT = Constraints(allow_short=True, max_weight=10.0, short_lower_bound=-10.0)


def _diagnostics(result):
    return {'message': result.message, 'n_restarts': result.n_restarts,
            'start_used': result.start_used}


def _brute_force_max_return(mu: np.ndarray, lower: float, upper: float) -> float:
    """Independent oracle: enumerate the vertices of the bounded simplex.

    A vertex of ``{sum(w)=1, lower <= w <= upper}`` has at most one coordinate
    strictly between the bounds, so fixing every other coordinate to a bound
    enumerates all of them.
    """
    n = len(mu)
    best = -np.inf
    for free in range(n):
        others = [i for i in range(n) if i != free]
        for assignment in itertools.product((lower, upper), repeat=n - 1):
            w = np.empty(n, dtype=float)
            for index, value in zip(others, assignment):
                w[index] = value
            w[free] = 1.0 - float(sum(assignment))
            if lower - 1e-12 <= w[free] <= upper + 1e-12:
                best = max(best, float(w @ np.asarray(mu, dtype=float)))
    return float(best)


# --------------------------------------------------------------------------- #
# Feasibility (CA-4)
# --------------------------------------------------------------------------- #
def test_min_variance_is_feasible(stats, constraints_long_only):
    result = solve_min_variance(stats.cov_annual, constraints_long_only)
    assert result.success is True
    assert result.feasibility.ok is True
    assert abs(result.weights.sum() - 1.0) <= 1e-6
    assert (result.weights >= -1e-9).all()
    assert (result.weights <= constraints_long_only.max_weight + 1e-9).all()


def test_max_sharpe_is_feasible(stats, convention, constraints_long_only):
    result = solve_max_sharpe(stats.mu_annual, stats.cov_annual,
                              convention.rf_cc_annual, constraints_long_only)
    assert result.success is True
    assert result.feasibility.ok is True
    assert abs(result.weights.sum() - 1.0) <= 1e-6
    assert (result.weights >= -1e-9).all()


def test_weight_cap_is_respected(stats, convention):
    constraints = Constraints(max_weight=0.3)
    for result in (solve_min_variance(stats.cov_annual, constraints),
                   solve_max_sharpe(stats.mu_annual, stats.cov_annual,
                                    convention.rf_cc_annual, constraints)):
        assert result.feasibility.ok is True
        assert (result.weights <= 0.3 + 1e-6).all()
        assert result.feasibility.max_bound_violation <= 1e-6


def test_max_weight_below_equal_share_raises(stats):
    with pytest.raises(InfeasibleConstraintsError):
        solve_min_variance(stats.cov_annual, Constraints(max_weight=0.1))


# --------------------------------------------------------------------------- #
# Closed forms (CA-5)
# --------------------------------------------------------------------------- #
def test_min_variance_matches_closed_form_with_shorts(constraints_short):
    numeric = solve_min_variance(TAME_COV, constraints_short)
    assert numeric.success is True
    assert np.allclose(numeric.weights, closed_form_min_variance(TAME_COV), atol=1e-6)


def test_max_sharpe_matches_closed_form_tangency_with_shorts(constraints_short):
    numeric = solve_max_sharpe(TAME_MU, TAME_COV, TAME_RF, constraints_short)
    assert numeric.success is True
    assert np.allclose(numeric.weights,
                       closed_form_tangency(TAME_MU, TAME_COV, TAME_RF), atol=1e-6)


def test_closed_forms_match_on_the_synthetic_universe(stats, convention):
    """Same check on the five-asset universe, where the optimum sits in a flat valley.

    With ``ftol=1e-12`` on a variance of order 1e-2 SLSQP stops once the
    objective stops improving, which leaves the weights up to a few 1e-6 away
    from the closed form along the flat directions. The documented criterion
    here is therefore the achieved objective (1e-9 relative), with the weights
    compared at 1e-4; the 1e-6 agreement of CA-5 is asserted on the
    well-conditioned problem above.
    """
    min_variance = solve_min_variance(stats.cov_annual, WIDE_SHORT)
    tangency = solve_max_sharpe(stats.mu_annual, stats.cov_annual,
                                convention.rf_cc_annual, WIDE_SHORT)
    closed_mv = closed_form_min_variance(stats.cov_annual)
    closed_tan = closed_form_tangency(stats.mu_annual, stats.cov_annual,
                                      convention.rf_cc_annual)
    assert np.allclose(min_variance.weights, closed_mv, atol=1e-4)
    assert np.allclose(tangency.weights, closed_tan, atol=1e-4)
    assert portfolio_volatility(min_variance.weights, stats.cov_annual) == pytest.approx(
        portfolio_volatility(closed_mv, stats.cov_annual), rel=1e-9)
    numeric_sharpe = portfolio_sharpe(tangency.weights, stats.mu_annual,
                                      stats.cov_annual, convention.rf_cc_annual)
    closed_sharpe = portfolio_sharpe(closed_tan, stats.mu_annual, stats.cov_annual,
                                     convention.rf_cc_annual)
    assert numeric_sharpe == pytest.approx(closed_sharpe, rel=1e-9)


# --------------------------------------------------------------------------- #
# Minimum variance is the least volatile portfolio (CA-6)
# --------------------------------------------------------------------------- #
def test_min_variance_volatility_below_equal_weight(stats, constraints_long_only):
    result = solve_min_variance(stats.cov_annual, constraints_long_only)
    optimal = portfolio_volatility(result.weights, stats.cov_annual)
    benchmark = portfolio_volatility(equal_weights(len(stats.names)), stats.cov_annual)
    assert optimal <= benchmark + 1e-12


def test_min_variance_volatility_below_every_frontier_point(stats, convention,
                                                            constraints_long_only):
    result = solve_min_variance(stats.cov_annual, constraints_long_only)
    optimal = portfolio_volatility(result.weights, stats.cov_annual)
    frontier = build_frontier(stats.mu_annual, stats.cov_annual, constraints_long_only,
                              convention.rf_cc_annual, 12)
    assert frontier.computed > 0
    assert optimal <= float(frontier.points['vol'].min()) + 1e-8


# --------------------------------------------------------------------------- #
# Multistart and certification
# --------------------------------------------------------------------------- #
def test_multistart_certifies_the_well_conditioned_case(stats, convention,
                                                        constraints_long_only):
    result = solve_max_sharpe(stats.mu_annual, stats.cov_annual,
                              convention.rf_cc_annual, constraints_long_only)
    summary = summarize_portfolio('max_sharpe', result.weights, stats.mu_annual,
                                  stats.cov_annual, convention.rf_cc_annual,
                                  converged=result.success,
                                  feasibility=result.feasibility,
                                  diagnostics=_diagnostics(result))
    assert summary.certified is True
    assert 1 <= result.n_restarts <= 8


def test_degenerate_covariance_does_not_break_the_solvers(singular_stats, convention,
                                                          constraints_long_only):
    result = solve_min_variance(singular_stats.cov_annual_raw, constraints_long_only)
    summary = summarize_portfolio('min_variance', result.weights,
                                  singular_stats.mu_annual,
                                  singular_stats.stats.cov_annual,
                                  convention.rf_cc_annual,
                                  converged=result.success,
                                  feasibility=result.feasibility,
                                  diagnostics=_diagnostics(result))
    assert np.all(np.isfinite(result.weights))
    assert summary.certified == bool(result.success and result.feasibility.ok)


def test_target_return_solver_hits_the_target(stats, constraints_long_only):
    target = float(np.mean(stats.mu_annual))
    result = solve_min_variance_target(stats.mu_annual, stats.cov_annual, target,
                                       constraints_long_only)
    assert result.success is True
    assert abs(float(result.weights @ stats.mu_annual) - target) <= 1e-6


def test_continuation_start_is_accepted(stats, constraints_long_only):
    target = float(np.mean(stats.mu_annual))
    base = solve_min_variance_target(stats.mu_annual, stats.cov_annual, target,
                                     constraints_long_only)
    continued = solve_min_variance_target(stats.mu_annual, stats.cov_annual, target,
                                          constraints_long_only, x0=base.weights)
    assert continued.success is True
    assert continued.start_used == 'continuation'
    assert np.allclose(continued.weights, base.weights, atol=1e-6)


# --------------------------------------------------------------------------- #
# Degenerate tangency guard
# --------------------------------------------------------------------------- #
def test_max_feasible_return_closed_form():
    mu = np.array([0.10, 0.07, 0.03, -0.02])
    for max_weight in (1.0, 0.6, 0.4, 0.3):
        constraints = Constraints(max_weight=max_weight)
        assert max_feasible_return(mu, constraints) == pytest.approx(
            _brute_force_max_return(mu, 0.0, max_weight), abs=1e-12)
    with_shorts = Constraints(allow_short=True, max_weight=0.6, short_lower_bound=-0.4)
    assert max_feasible_return(mu, with_shorts) == pytest.approx(
        _brute_force_max_return(mu, -0.4, 0.6), abs=1e-12)
    assert max_feasible_return(mu, Constraints(max_weight=1.0)) == pytest.approx(
        float(mu.max()), abs=1e-12)


def test_max_sharpe_with_rf_above_all_assets(stats):
    # Case (a): the risk-free rate is above max(mu_annual) and no cap binds.
    convention = FrequencyConvention(trading_days=252, rf_annual_effective=1.0)
    assert convention.rf_cc_annual > float(stats.mu_annual.max())
    constraints = Constraints(max_weight=1.0)
    result = solve_max_sharpe(stats.mu_annual, stats.cov_annual,
                              convention.rf_cc_annual, constraints)
    expected = solve_min_variance(stats.cov_annual, constraints).weights
    assert result.message == TANGENCY_UNDEFINED
    assert result.start_used == 'min_variance_fallback'
    assert result.success is True
    assert result.n_restarts == 0
    assert np.isnan(result.objective_value)
    assert np.allclose(result.weights, expected, atol=1e-12)
    assert result.feasibility.ok is True
    summary = summarize_portfolio('max_sharpe', result.weights, stats.mu_annual,
                                  stats.cov_annual, convention.rf_cc_annual,
                                  converged=result.success,
                                  feasibility=result.feasibility,
                                  diagnostics=_diagnostics(result))
    assert summary.certified is False


def test_max_sharpe_with_cap_keeping_every_portfolio_below_rf():
    # Case (b): max(mu) > rf but the binding cap keeps max_feasible_return below rf.
    mu = np.array([0.10, 0.0, 0.0, 0.0, 0.0])
    cov = np.diag([0.04, 0.03, 0.05, 0.02, 0.06])
    constraints = Constraints(max_weight=0.25)
    rf_cc_annual = 0.044
    assert float(mu.max()) > rf_cc_annual
    assert max_feasible_return(mu, constraints) == pytest.approx(0.025, abs=1e-12)
    result = solve_max_sharpe(mu, cov, rf_cc_annual, constraints)
    expected = solve_min_variance(cov, constraints).weights
    assert result.message == TANGENCY_UNDEFINED
    assert result.start_used == 'min_variance_fallback'
    assert result.success is True
    assert np.allclose(result.weights, expected, atol=1e-12)
    assert result.feasibility.ok is True
    summary = summarize_portfolio('max_sharpe', result.weights, mu, cov, rf_cc_annual,
                                  converged=result.success,
                                  feasibility=result.feasibility,
                                  diagnostics=_diagnostics(result))
    assert summary.certified is False


def test_check_feasibility_reports_each_violation(constraints_long_only):
    assert check_feasibility(np.array([0.5, 0.5]), constraints_long_only).ok is True
    broken_sum = check_feasibility(np.array([0.5, 0.2]), constraints_long_only)
    assert broken_sum.sum_ok is False
    assert broken_sum.sum_error == pytest.approx(0.3)
    broken_bounds = check_feasibility(np.array([1.5, -0.5]), Constraints(max_weight=1.0))
    assert broken_bounds.bounds_ok is False
    assert broken_bounds.max_bound_violation == pytest.approx(0.5)
    not_finite = check_feasibility(np.array([np.nan, 1.0]), constraints_long_only)
    assert not_finite.finite_ok is False
    assert not_finite.ok is False
