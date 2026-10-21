"""Tests of core/backtest.py (design.md section 21, row ``test_backtest.py``).

Covers CA-11 (no look-ahead), CA-12 (costs over turnover against a hand-made
case), CA-13 (zero costs and daily rebalancing reproduce the cumulative return
of the series), CA-14 (shared span, shared capital, expected rebalance count)
and RF-BT7 (a window without a verified optimum keeps the previous weights).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.backtest import (REBALANCE_RULES, BacktestConfig, compute_drawdown,
                           generate_rebalance_indices, performance_metrics,
                           run_backtest)
from core.rng import child_rng
from core.stats import portfolio_simple_return_series, realized_metrics
from core.types import AlignedData, DataQualityReport, PortfolioError

ALL_RULES = tuple(REBALANCE_RULES) + ('21d',)


def make_aligned(returns: np.ndarray, names=None, start: str = '2019-01-02') -> AlignedData:
    """Build an ``AlignedData`` from a returns matrix, with consistent prices."""
    matrix = np.asarray(returns, dtype=float)
    rows, n = matrix.shape
    dates = pd.bdate_range(start, periods=rows + 1)
    log_levels = np.vstack([np.zeros((1, n), dtype=float), np.cumsum(matrix, axis=0)])
    prices = 100.0 * np.exp(log_levels)
    quality = DataQualityReport(per_asset=pd.DataFrame(), common_dates=rows + 1,
                                missing_policy='drop', filled_cells={},
                                non_adjusted_assets=(),
                                returns_spanning_multiple_days=0,
                                max_return_span_days=1)
    return AlignedData(names=tuple(names) if names else tuple(f'A{i}' for i in range(n)),
                       dates=dates, prices=prices, return_dates=dates[1:],
                       log_returns=matrix, quality=quality)


# --------------------------------------------------------------------------- #
# Rebalance calendar
# --------------------------------------------------------------------------- #
@pytest.mark.filterwarnings('error::FutureWarning')
def test_every_rule_generates_indices_without_future_warnings():
    dates = pd.bdate_range('2015-01-02', periods=1500)
    for rule in ALL_RULES:
        indices = generate_rebalance_indices(dates, rule, 252)
        assert indices.size > 0
        assert indices.min() >= 251
        assert indices.max() <= len(dates) - 2
        assert np.array_equal(indices, np.unique(indices))


def test_unknown_rule_raises_before_touching_pandas():
    dates = pd.bdate_range('2020-01-01', periods=50)
    with pytest.raises(ValueError, match='unknown_rebalance_rule'):
        generate_rebalance_indices(dates, 'ME', 10)
    with pytest.raises(ValueError, match='unknown_rebalance_rule'):
        generate_rebalance_indices(dates, 'monthly', 10)
    with pytest.raises(ValueError, match='rebalance_step_not_positive'):
        generate_rebalance_indices(dates, '0d', 10)


@pytest.mark.filterwarnings('error::FutureWarning')
def test_rebalance_indices_with_zero_window():
    dates = pd.bdate_range('2021-01-04', periods=800)
    for rule in ALL_RULES:
        indices = generate_rebalance_indices(dates, rule, 0)
        assert indices.size > 0
        assert indices.min() >= 1
        assert -1 not in indices
        assert indices.max() <= len(dates) - 1


def test_step_rule_counts_positions():
    dates = pd.bdate_range('2021-01-04', periods=100)
    indices = generate_rebalance_indices(dates, '10d', 21)
    assert indices[0] == 20
    assert np.array_equal(np.diff(indices), np.full(indices.size - 1, 10))


# --------------------------------------------------------------------------- #
# No look-ahead (CA-11)
# --------------------------------------------------------------------------- #
def test_no_look_ahead(aligned, convention, constraints_long_only):
    base = np.asarray(aligned.log_returns, dtype=float)[:400].copy()
    config = BacktestConfig(window=120, rebalance_rule='A', initial_capital=100_000.0,
                            cost_rate=0.001, strategies=('min_variance', 'max_sharpe'))
    data = make_aligned(base, aligned.names)
    indices = generate_rebalance_indices(data.return_dates, config.rebalance_rule,
                                         config.window)
    decision = int(indices[0])

    perturbed = base.copy()
    perturbed[decision + 1:] *= 3.0
    assert not np.array_equal(base, perturbed)

    untouched = run_backtest(data, convention, constraints_long_only, config)
    shifted = run_backtest(make_aligned(perturbed, aligned.names), convention,
                           constraints_long_only, config)
    for strategy in config.strategies:
        before = untouched.weights_history[strategy].iloc[0].to_numpy(dtype=float)
        after = shifted.weights_history[strategy].iloc[0].to_numpy(dtype=float)
        assert np.array_equal(before, after)


# --------------------------------------------------------------------------- #
# Costs over turnover (CA-12)
# --------------------------------------------------------------------------- #
def test_costs_over_turnover_match_a_handmade_case(convention, constraints_long_only):
    returns = np.array([[0.0, 0.0],
                        [np.log(2.0), 0.0],
                        [0.0, 0.0]])
    data = make_aligned(returns)
    config = BacktestConfig(window=1, rebalance_rule='1d', initial_capital=100.0,
                            cost_rate=0.001, strategies=('equal_weight',))
    result = run_backtest(data, convention, constraints_long_only, config)
    equity = result.equity['equal_weight'].to_numpy(dtype=float)

    # Entry from cash: turnover = 1, cost = 100 * 0,001 = 0,1 -> 99,9 invested.
    assert equity[0] == pytest.approx(99.9, abs=1e-12)
    # Asset A doubles: 49,95 -> 99,9 and the portfolio is worth 149,85 before the
    # second rebalance, whose turnover is 1/3 and whose cost is 0,04995.
    assert equity[1] == pytest.approx(149.85 - 149.85 / 3.0 * 0.001, abs=1e-12)
    assert equity[2] == pytest.approx(equity[1], abs=1e-12)
    assert result.metrics.loc['equal_weight', 'total_costs'] == pytest.approx(
        0.1 + 149.85 / 3.0 * 0.001, abs=1e-12)
    assert result.metrics.loc['equal_weight', 'n_rebalances'] == 2.0
    assert result.metrics.loc['equal_weight', 'final_capital'] == pytest.approx(
        equity[-1], abs=1e-12)


# --------------------------------------------------------------------------- #
# Zero costs and daily rebalancing (CA-13)
# --------------------------------------------------------------------------- #
def test_equal_weight_without_costs_matches_the_cumulative_series(convention,
                                                                  constraints_long_only):
    rng = child_rng(5, 'paths')
    returns = rng.normal(0.0004, 0.01, size=(60, 3))
    data = make_aligned(returns)
    config = BacktestConfig(window=2, rebalance_rule='1d', initial_capital=100_000.0,
                            cost_rate=0.0, strategies=('equal_weight',))
    result = run_backtest(data, convention, constraints_long_only, config)
    equity = result.equity['equal_weight'].to_numpy(dtype=float)

    weights = np.full(3, 1.0 / 3.0)
    simple = portfolio_simple_return_series(returns, weights)
    expected = 100_000.0 * np.cumprod(1.0 + simple[2:])
    assert equity[0] == pytest.approx(100_000.0, abs=1e-9)
    assert np.allclose(equity[1:], expected, atol=1e-8)
    assert result.metrics.loc['equal_weight', 'total_costs'] == 0.0


# --------------------------------------------------------------------------- #
# Shared span and rebalance count (CA-14)
# --------------------------------------------------------------------------- #
def test_strategies_share_span_capital_and_calendar(aligned, convention,
                                                    constraints_long_only):
    data = make_aligned(np.asarray(aligned.log_returns, dtype=float)[:400],
                        aligned.names)
    config = BacktestConfig(window=120, rebalance_rule='Q', initial_capital=100_000.0,
                            cost_rate=0.001)
    result = run_backtest(data, convention, constraints_long_only, config)
    expected_dates = data.return_dates[
        generate_rebalance_indices(data.return_dates, 'Q', 120)]
    assert list(result.rebalance_dates) == list(expected_dates)

    for strategy in config.strategies:
        assert strategy in result.equity.columns
        assert len(result.equity[strategy]) == len(result.equity.index)
        assert result.equity[strategy].iloc[0] == pytest.approx(
            100_000.0 * (1.0 - config.cost_rate), abs=1e-6)
        kept = sum(1 for event in result.events
                   if event.startswith(f'rebalance_kept_previous_weights:{strategy}:'))
        assert result.metrics.loc[strategy, 'n_rebalances'] == float(
            len(expected_dates) - kept)
        assert result.weights_history[strategy].shape == (len(expected_dates),
                                                          len(data.names))
    for name in data.names:
        assert result.equity[name].iloc[0] == pytest.approx(100_000.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Drawdown
# --------------------------------------------------------------------------- #
def test_compute_drawdown_on_a_constructed_series():
    equity = np.array([100.0, 120.0, 60.0, 90.0, 150.0])
    drawdown = compute_drawdown(equity)
    assert np.allclose(drawdown, [0.0, 0.0, -0.5, -0.25, 0.0])
    assert drawdown.min() == pytest.approx(-0.5)


def test_backtest_drawdown_is_non_positive(aligned, convention, constraints_long_only):
    data = make_aligned(np.asarray(aligned.log_returns, dtype=float)[:300],
                        aligned.names)
    config = BacktestConfig(window=120, rebalance_rule='A', initial_capital=100_000.0,
                            strategies=('equal_weight',))
    result = run_backtest(data, convention, constraints_long_only, config)
    assert (result.drawdown.to_numpy(dtype=float) <= 1e-12).all()
    assert result.metrics.loc['equal_weight', 'max_drawdown'] == pytest.approx(
        float(result.drawdown['equal_weight'].min()))


# --------------------------------------------------------------------------- #
# RF-BT7: no verified optimum keeps the previous weights
# --------------------------------------------------------------------------- #
def test_window_without_certified_optimum_keeps_previous_weights(convention,
                                                                 constraints_long_only):
    rng = child_rng(9, 'paths')
    returns = -0.001 + rng.normal(0.0, 0.004, size=(60, 3))
    data = make_aligned(returns)
    config = BacktestConfig(window=20, rebalance_rule='21d', initial_capital=100_000.0,
                            cost_rate=0.001, strategies=('max_sharpe',))
    result = run_backtest(data, convention, constraints_long_only, config)

    history = result.weights_history['max_sharpe'].to_numpy(dtype=float)
    assert history.shape[0] >= 2
    assert np.allclose(history, np.full_like(history, 1.0 / 3.0), atol=1e-12)
    assert any(event.startswith('first_rebalance_equal_weights:max_sharpe:')
               for event in result.events)
    assert any(event.startswith('rebalance_kept_previous_weights:max_sharpe:')
               for event in result.events)
    # The skipped rebalances neither trade nor pay: only the entry costs.
    assert result.metrics.loc['max_sharpe', 'n_rebalances'] == 1.0
    assert result.metrics.loc['max_sharpe', 'total_costs'] == pytest.approx(100.0,
                                                                            abs=1e-9)


# --------------------------------------------------------------------------- #
# Metrics assembler
# --------------------------------------------------------------------------- #
def test_performance_metrics_only_assembles(aligned, convention, constraints_long_only):
    data = make_aligned(np.asarray(aligned.log_returns, dtype=float)[:300],
                        aligned.names)
    config = BacktestConfig(window=120, rebalance_rule='A', initial_capital=100_000.0,
                            strategies=('equal_weight',))
    result = run_backtest(data, convention, constraints_long_only, config)
    equity = result.equity['equal_weight'].to_numpy(dtype=float)
    reference = realized_metrics(equity, convention)
    assembled = performance_metrics(equity, result.equity.index, convention, 3, 12.5)
    for key in ('total_return', 'cagr', 'mu_cc_real', 'vol_real', 'sharpe_real'):
        assert assembled[key] == reference[key]
        assert result.metrics.loc['equal_weight', key] == pytest.approx(reference[key])
    assert assembled['n_rebalances'] == 3.0
    assert assembled['total_costs'] == 12.5
    assert assembled['max_drawdown'] == pytest.approx(float(compute_drawdown(equity).min()))


def test_window_out_of_range_raises(aligned, convention, constraints_long_only):
    data = make_aligned(np.asarray(aligned.log_returns, dtype=float)[:50],
                        aligned.names)
    with pytest.raises(PortfolioError):
        run_backtest(data, convention, constraints_long_only,
                     BacktestConfig(window=50, rebalance_rule='M'))
    with pytest.raises(PortfolioError):
        run_backtest(data, convention, constraints_long_only,
                     BacktestConfig(window=0, rebalance_rule='M'))
