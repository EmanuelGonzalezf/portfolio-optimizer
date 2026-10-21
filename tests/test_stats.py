"""Tests for core/stats.py plus the two structural invariants of core/.

Covers CA-1, CA-2, CA-3, CA-7 and CA-8 (design.md section 21).
"""

from __future__ import annotations

import ast
import math
import pathlib
import re

import numpy as np
import pytest

from core.stats import (cholesky_factor, compute_asset_stats,
                        compute_asset_stats_from_returns, compute_log_returns,
                        correlation_from_covariance, ensure_psd, estimate_moments,
                        gross_from_log, portfolio_log_return_series, portfolio_return,
                        portfolio_sharpe, portfolio_simple_return_series,
                        portfolio_variance, portfolio_volatility, realized_metrics,
                        simple_from_log, summarize_portfolio)
from core.types import FeasibilityCheck, FrequencyConvention

CORE_DIR = pathlib.Path(__file__).resolve().parents[1] / 'core'
SPANISH_CHARS = set('\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1\u00bf\u00a1'
                    '\u00c1\u00c9\u00cd\u00d3\u00da\u00d1')
SPANISH_WORDS = re.compile(r'(?:^|\s)(de|la|el|los|las|del|una|por|con|que|no)(?:\s|$)')


# --------------------------------------------------------------------------- #
# Returns and moments against hand values (CA-1, CA-2)
# --------------------------------------------------------------------------- #
def test_log_returns_match_hand_values(tiny_series):
    returns = compute_log_returns(tiny_series.prices)
    assert returns.shape == (3, 2)
    assert np.allclose(returns, tiny_series.expected_log_returns, atol=1e-10)


def test_simple_and_gross_are_derived_from_log(tiny_series):
    returns = compute_log_returns(tiny_series.prices)
    assert np.allclose(gross_from_log(returns), np.exp(returns), atol=1e-15)
    assert np.allclose(simple_from_log(returns), np.exp(returns) - 1.0, atol=1e-15)


def test_moments_match_hand_computed_covariance(tiny_series):
    returns = tiny_series.expected_log_returns
    mu_daily, cov_daily, mu_annual, cov_annual = estimate_moments(returns, 252)

    t, n = returns.shape
    hand_mu = [sum(returns[k, i] for k in range(t)) / t for i in range(n)]
    hand_cov = [[sum((returns[k, i] - hand_mu[i]) * (returns[k, j] - hand_mu[j])
                     for k in range(t)) / (t - 1) for j in range(n)] for i in range(n)]

    assert np.allclose(mu_daily, hand_mu, atol=1e-12)
    assert np.allclose(cov_daily, hand_cov, atol=1e-12)
    assert np.allclose(mu_annual, np.asarray(hand_mu) * 252, atol=1e-10)
    assert np.allclose(cov_annual, np.asarray(hand_cov) * 252, atol=1e-10)


def test_annualization_is_coherent(stats, convention):
    assert np.allclose(stats.mu_annual, stats.mu_daily * convention.trading_days, atol=1e-12)
    assert np.allclose(stats.sigma_annual, np.sqrt(np.diag(stats.cov_annual)), atol=1e-12)
    assert np.allclose(stats.sigma_annual,
                       np.sqrt(np.diag(stats.cov_daily)) * math.sqrt(convention.trading_days),
                       atol=1e-12)


# --------------------------------------------------------------------------- #
# Correlation (CA-3)
# --------------------------------------------------------------------------- #
def test_correlation_diagonal_and_range(stats):
    assert np.allclose(np.diag(stats.corr), 1.0, atol=1e-12)
    assert stats.corr.min() >= -1.0
    assert stats.corr.max() <= 1.0
    assert np.allclose(stats.corr, stats.corr.T, atol=1e-12)


def test_correlation_invariant_to_annualization(aligned):
    mu_daily, cov_daily, mu_annual, cov_annual = estimate_moments(aligned.log_returns, 252)
    assert np.allclose(correlation_from_covariance(cov_daily),
                       correlation_from_covariance(cov_annual), atol=1e-12)


# --------------------------------------------------------------------------- #
# Covariance conditioning (CA-8)
# --------------------------------------------------------------------------- #
def test_ensure_psd_repairs_singular_covariance(singular_stats):
    raw = singular_stats.cov_annual_raw
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(raw)
    fixed, repaired, epsilon = ensure_psd(raw)
    assert repaired is True
    assert epsilon > 0.0
    np.linalg.cholesky(fixed)
    assert np.allclose(fixed, raw, atol=1e-6)
    assert singular_stats.stats.psd_repaired is True
    assert singular_stats.stats.ridge_epsilon > 0.0


def test_ensure_psd_leaves_a_good_matrix_untouched(stats):
    fixed, repaired, epsilon = ensure_psd(stats.cov_annual)
    assert repaired is False
    assert epsilon == 0.0
    assert np.array_equal(fixed, stats.cov_annual)


def test_cholesky_factor_reproduces_the_repaired_matrix(singular_stats, stats):
    factor, repaired = cholesky_factor(singular_stats.cov_annual_raw)
    assert repaired is True
    expected, _, _ = ensure_psd(singular_stats.cov_annual_raw)
    assert np.allclose(factor @ factor.T, expected, atol=1e-12)

    factor_ok, repaired_ok = cholesky_factor(stats.cov_annual)
    assert repaired_ok is False
    assert np.allclose(factor_ok @ factor_ok.T, stats.cov_annual, atol=1e-12)


# --------------------------------------------------------------------------- #
# Risk-free conversions and Sharpe (CA-7)
# --------------------------------------------------------------------------- #
def test_risk_free_conversions_match_the_formulas():
    convention = FrequencyConvention(trading_days=252, rf_annual_effective=0.045)
    assert convention.rf_cc_annual == pytest.approx(math.log(1.045), abs=1e-15)
    assert convention.rf_cc_daily == pytest.approx(math.log(1.045) / 252, abs=1e-15)
    assert convention.rf_simple_daily == pytest.approx(1.045 ** (1 / 252) - 1, abs=1e-15)
    assert convention.rf_cc_annual < 0.045


def test_portfolio_metrics_are_consistent(stats):
    weights = np.full(len(stats.names), 0.2)
    variance = portfolio_variance(weights, stats.cov_annual)
    assert portfolio_volatility(weights, stats.cov_annual) == pytest.approx(
        math.sqrt(variance), abs=1e-15)
    expected = (portfolio_return(weights, stats.mu_annual)
                - stats.convention.rf_cc_annual) / math.sqrt(variance)
    assert portfolio_sharpe(weights, stats.mu_annual, stats.cov_annual,
                            stats.convention.rf_cc_annual) == pytest.approx(expected, abs=1e-15)


def test_portfolio_variance_is_floored():
    assert portfolio_variance(np.zeros(3), np.eye(3)) == pytest.approx(1e-16, abs=0.0)


def test_individual_sharpe_uses_portfolio_sharpe(stats):
    direct = (stats.mu_annual - stats.convention.rf_cc_annual) / stats.sigma_annual
    assert np.allclose(stats.sharpe_individual, direct, atol=1e-12)


def test_sharpe_log_series_matches_quadratic_form(aligned, stats, convention):
    assert stats.psd_repaired is False
    weights = np.full(len(stats.names), 0.2)
    series = portfolio_log_return_series(aligned.log_returns, weights)
    tdy = convention.trading_days
    mu_series = series.mean() * tdy
    vol_series = series.std(ddof=1) * math.sqrt(tdy)
    sharpe_series = (mu_series - convention.rf_cc_annual) / vol_series
    quadratic = portfolio_sharpe(weights, stats.mu_annual, stats.cov_annual,
                                 convention.rf_cc_annual)
    assert sharpe_series == pytest.approx(quadratic, abs=1e-10)


def test_jensen_gap_is_documented(aligned, stats, convention):
    """The log/money gap is an EXPECTED difference, not a failure.

    For a single asset the gap is sigma^2/2. For a fixed-weight portfolio the
    correct closed form is sum_i w_i sigma_i^2 / 2, which exceeds sigma_p^2 / 2
    by the diversification term; both are checked with the documented tolerance
    0.25 * sigma_p^2.
    """
    tdy = convention.trading_days
    n = len(stats.names)

    single = np.zeros(n)
    single[0] = 1.0
    gap_single = (portfolio_simple_return_series(aligned.log_returns, single).mean() * tdy
                  - portfolio_log_return_series(aligned.log_returns, single).mean() * tdy)
    variance_single = portfolio_variance(single, stats.cov_annual)
    assert gap_single == pytest.approx(variance_single / 2.0, abs=0.25 * variance_single)

    weights = np.full(n, 1.0 / n)
    gap_portfolio = (portfolio_simple_return_series(aligned.log_returns, weights).mean() * tdy
                     - portfolio_log_return_series(aligned.log_returns, weights).mean() * tdy)
    variance_portfolio = portfolio_variance(weights, stats.cov_annual)
    per_asset_gap = float(np.sum(weights * np.diag(stats.cov_annual)) / 2.0)
    assert gap_portfolio == pytest.approx(per_asset_gap, abs=0.25 * variance_portfolio)
    assert gap_portfolio > variance_portfolio / 2.0


# --------------------------------------------------------------------------- #
# Realised metrics
# --------------------------------------------------------------------------- #
def test_realized_metrics_against_handmade_equity(convention):
    equity = np.array([100.0, 110.0, 121.0, 108.9])
    metrics = realized_metrics(equity, convention)
    tdy = convention.trading_days
    n_days = 3
    steps = np.log(equity[1:] / equity[:-1])

    assert metrics['degenerate'] == 0.0
    assert metrics['total_return'] == pytest.approx(108.9 / 100.0 - 1.0, abs=1e-12)
    assert metrics['cagr'] == pytest.approx((108.9 / 100.0) ** (tdy / n_days) - 1.0, rel=1e-12)
    assert metrics['mu_cc_real'] == pytest.approx(
        math.log(108.9 / 100.0) * tdy / n_days, abs=1e-12)
    assert metrics['vol_real'] == pytest.approx(steps.std(ddof=1) * math.sqrt(tdy), abs=1e-12)
    assert metrics['sharpe_real'] == pytest.approx(
        (metrics['mu_cc_real'] - convention.rf_cc_annual) / metrics['vol_real'], abs=1e-12)


def test_realized_metrics_derives_n_days_from_a_sliced_equity(convention):
    equity = np.array([100.0, 110.0, 121.0, 108.9])
    sub = equity[1:]
    metrics = realized_metrics(sub, convention)
    tdy = convention.trading_days
    assert metrics['mu_cc_real'] == pytest.approx(math.log(sub[-1] / sub[0]) * tdy / 2, abs=1e-12)
    assert metrics['cagr'] == pytest.approx((sub[-1] / sub[0]) ** (tdy / 2) - 1.0, rel=1e-12)


def test_realized_metrics_on_degenerate_equity(convention):
    exhausted = realized_metrics(np.array([100.0, 60.0, 0.0, 0.0]), convention)
    assert exhausted['degenerate'] == 1.0
    for key in ('total_return', 'cagr', 'mu_cc_real', 'vol_real', 'sharpe_real'):
        assert math.isnan(exhausted[key])

    single = realized_metrics(np.array([100.0]), convention)
    assert single['degenerate'] == 1.0
    assert math.isnan(single['cagr'])

    empty = realized_metrics(np.array([]), convention)
    assert empty['degenerate'] == 1.0


# --------------------------------------------------------------------------- #
# Aggregates
# --------------------------------------------------------------------------- #
def test_compute_asset_stats_matches_the_array_entry_point(aligned, convention):
    adapted = compute_asset_stats(aligned, convention)
    direct = compute_asset_stats_from_returns(aligned.log_returns, aligned.names,
                                              len(aligned.log_returns), convention)
    assert adapted.names == direct.names
    assert adapted.n_obs == direct.n_obs
    assert adapted.psd_repaired == direct.psd_repaired
    assert adapted.ridge_epsilon == direct.ridge_epsilon
    assert adapted.convention == direct.convention
    for field in ('mu_daily', 'cov_daily', 'mu_annual', 'cov_annual', 'sigma_annual',
                  'corr', 'sharpe_individual', 'min_return', 'max_return'):
        assert np.array_equal(getattr(adapted, field), getattr(direct, field))


def test_summarize_portfolio_certifies_only_feasible_convergence(stats):
    weights = np.full(len(stats.names), 0.2)
    good = FeasibilityCheck(True, True, True, 0.0, 0.0)
    summary = summarize_portfolio('equal_weight', weights, stats.mu_annual,
                                  stats.cov_annual, stats.convention.rf_cc_annual,
                                  converged=True, feasibility=good, diagnostics={})
    assert summary.certified is True
    assert summary.n_positive == len(stats.names)
    assert summary.sharpe == pytest.approx(
        portfolio_sharpe(weights, stats.mu_annual, stats.cov_annual,
                         stats.convention.rf_cc_annual), abs=1e-15)

    bad = FeasibilityCheck(False, True, True, 1e-3, 0.0)
    assert summarize_portfolio('max_sharpe', weights, stats.mu_annual, stats.cov_annual,
                               stats.convention.rf_cc_annual, converged=True,
                               feasibility=bad, diagnostics={}).certified is False


def test_summarize_portfolio_never_certifies_the_tangency_fallback(stats):
    weights = np.full(len(stats.names), 0.2)
    good = FeasibilityCheck(True, True, True, 0.0, 0.0)
    summary = summarize_portfolio(
        'max_sharpe', weights, stats.mu_annual, stats.cov_annual,
        stats.convention.rf_cc_annual, converged=True, feasibility=good,
        diagnostics={'message': 'tangency_undefined_rf_above_all_assets'})
    assert summary.converged is True
    assert summary.feasibility.ok is True
    assert summary.certified is False


# --------------------------------------------------------------------------- #
# Structural invariants of core/
# --------------------------------------------------------------------------- #
def _core_modules():
    return sorted(path for path in CORE_DIR.glob('*.py'))


def _docstring_nodes(tree: ast.AST) -> set:
    marked = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, 'body', [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                marked.add(id(body[0].value))
    return marked


def test_no_spanish_literals_in_core():
    offenders = []
    for path in _core_modules():
        tree = ast.parse(path.read_text(encoding='utf-8'))
        docstrings = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node) in docstrings:
                continue
            text = node.value
            if SPANISH_CHARS & set(text) or SPANISH_WORDS.search(text.lower()):
                offenders.append(f'{path.name}:{node.lineno}:{text!r}')
    assert offenders == []


def test_core_has_no_streamlit_dependency():
    offenders = []
    for path in _core_modules():
        source = path.read_text(encoding='utf-8')
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                offenders += [f'{path.name}:{alias.name}' for alias in node.names
                              if alias.name.split('.')[0] in ('streamlit', 'plotly')]
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split('.')[0]
                if root in ('streamlit', 'plotly', 'ui'):
                    offenders.append(f'{path.name}:{node.module}')
        assert 'import streamlit' not in source
    assert offenders == []
