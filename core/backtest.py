"""Walk-forward backtesting without look-ahead (design.md section 9).

Anti-look-ahead contract, stated over the ``(T, n)`` array ``log_returns`` whose
row ``t`` is realised on ``return_dates[t]``:

    at the decision index ``i`` the estimation uses EXCLUSIVELY
    log_returns[i-window+1 : i+1], and the resulting weights are applied to
    log_returns[i+1 : j+1], with ``j`` the next rebalance index. No weight in
    force at ``t`` depends on log_returns[t'] with t' >= t.

``_estimation_slice`` is the only place that cuts the window, so there is a
single point to audit.

Accounting (section 9.2): positions are carried as per-asset VALUES, the three
strategies are always fully invested (``cash(t) = 0``) and the transaction cost
is ``turnover * V * cost_rate`` without the 0,5 factor, because ``turnover`` as a
sum of absolute weight changes already counts both legs.

This module never imports Streamlit and emits no user-facing text: ``events``
carries ``code:strategy:date`` triples and ui/labels.py renders them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.optimize import check_feasibility, equal_weights, solve_max_sharpe, solve_min_variance
from core.stats import (TANGENCY_UNDEFINED, ensure_psd, estimate_moments, gross_from_log,
                        realized_metrics)
from core.types import AlignedData, Constraints, FrequencyConvention, PortfolioError

__all__ = [
    'BacktestConfig',
    'BacktestResult',
    'REBALANCE_RULES',
    'STRATEGIES',
    'STEP_RULE_PATTERN',
    'generate_rebalance_indices',
    'run_backtest',
    'compute_drawdown',
    'performance_metrics',
]

REBALANCE_RULES = {'W': 'W-FRI', 'M': 'ME', 'Q': 'QE', 'S': '2QE', 'A': 'YE'}
STEP_RULE_PATTERN = re.compile(r'\d+d')
STRATEGIES = ('max_sharpe', 'min_variance', 'equal_weight')
EQUAL_WEIGHT = 'equal_weight'


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest settings.

    ``cash_interest`` does not exist: the three strategies are always fully
    invested, so there is no cash balance to remunerate (section 9.2).
    """

    window: int = 252
    rebalance_rule: str = 'M'
    initial_capital: float = 100_000.0
    cost_rate: float = 0.001
    strategies: tuple = STRATEGIES

    def to_tuple(self) -> tuple:
        return (int(self.window), str(self.rebalance_rule), float(self.initial_capital),
                float(self.cost_rate), tuple(str(name) for name in self.strategies))

    @classmethod
    def from_tuple(cls, data: tuple) -> 'BacktestConfig':
        window, rule, capital, cost_rate, strategies = data
        return cls(window=int(window), rebalance_rule=str(rule),
                   initial_capital=float(capital), cost_rate=float(cost_rate),
                   strategies=tuple(str(name) for name in strategies))


@dataclass(frozen=True)
class BacktestResult:
    """Equity paths, weight history, metrics and the events worth reporting."""

    equity: pd.DataFrame
    weights_history: dict
    metrics: pd.DataFrame
    drawdown: pd.DataFrame
    events: tuple
    rebalance_dates: pd.DatetimeIndex


# --------------------------------------------------------------------------- #
# Rebalance calendar
# --------------------------------------------------------------------------- #
def generate_rebalance_indices(dates: pd.DatetimeIndex, rule: str,
                               window: int) -> np.ndarray:
    """Positions of ``dates`` on which the portfolio is recomposed.

    The internal keys of ``REBALANCE_RULES`` are translated to pandas aliases
    only here, right before ``resample``: ``'S'`` is not a pandas alias and
    ``'M'``, ``'Q'``, ``'A'`` are deprecated in pandas 2.2, so an unvalidated key
    would surface as a ``FutureWarning`` or an exception. An unknown key raises
    ``ValueError`` BEFORE touching pandas. ``'Nd'`` means every N positions and
    never goes through ``resample``.

    With ``window > 0`` the first rebalance is ``window - 1`` and the range is
    ``[window-1, len(dates)-2]`` (a decision at ``i`` needs returns at ``i+1``).

    With ``window == 0`` -- the installment horizon of section 11.3, where the
    weights are given and there is no estimation window -- there is NO start
    rebalance: ``i = window - 1`` would be ``-1``, which NumPy reads as the last
    day, so the rule is explicit and the indices are bounded to
    ``1 <= i <= len(dates) - 1``. Index 0 is excluded because recomposing the
    portfolio the day it was built would only pay fees.
    """
    index = pd.DatetimeIndex(dates)
    size = int(len(index))
    key = str(rule)
    step = None
    alias = None
    if key in REBALANCE_RULES:
        alias = REBALANCE_RULES[key]
    elif STEP_RULE_PATTERN.fullmatch(key):
        step = int(key[:-1])
        if step <= 0:
            raise ValueError('rebalance_step_not_positive')
    else:
        raise ValueError('unknown_rebalance_rule')

    span = int(window)
    if span < 0:
        raise ValueError('rebalance_window_negative')
    if size == 0:
        return np.empty(0, dtype=int)

    if span == 0:
        low, high, start = 1, size - 1, 0
    else:
        low, high, start = span - 1, size - 2, span - 1

    if step is not None:
        positions = np.arange(start, size, step, dtype=int)
    else:
        marks = pd.Series(np.arange(size), index=index).resample(alias).last().dropna()
        positions = marks.to_numpy(dtype=int)
        if span > 0:
            positions = np.append(positions, span - 1)

    positions = positions[(positions >= low) & (positions <= high)]
    return np.unique(positions.astype(int))


def _estimation_slice(i: int, window: int) -> slice:
    """The ONLY window cut of the module: returns already realised at ``i``."""
    start = max(0, int(i) - int(window) + 1)
    return slice(start, int(i) + 1)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_drawdown(equity: np.ndarray) -> np.ndarray:
    """``V / cummax(V) - 1``, always <= 0."""
    values = np.asarray(equity, dtype=float).ravel()
    if values.size == 0:
        return values
    peak = np.maximum.accumulate(values)
    with np.errstate(divide='ignore', invalid='ignore'):
        drawdown = np.where(peak > 0.0, values / peak - 1.0, 0.0)
    return drawdown


def performance_metrics(equity: np.ndarray, dates, convention: FrequencyConvention,
                        n_rebalances: int, total_costs: float) -> dict:
    """Assembler only: ``realized_metrics(equity, convention)`` plus four items.

    No ``n_days`` argument is forwarded: ``realized_metrics`` always derives it as
    ``len(equity) - 1`` (section 3), so a sub-period is evaluated by slicing
    ``equity`` before calling. ``dates`` is kept in the signature for
    traceability of the evaluated span; the figures do not depend on it.
    """
    values = np.asarray(equity, dtype=float).ravel()
    metrics = dict(realized_metrics(values, convention))
    drawdown = compute_drawdown(values)
    metrics['max_drawdown'] = float(drawdown.min()) if drawdown.size else float('nan')
    metrics['final_capital'] = float(values[-1]) if values.size else float('nan')
    metrics['n_rebalances'] = float(n_rebalances)
    metrics['total_costs'] = float(total_costs)
    metrics['n_observations'] = float(values.size)
    return metrics


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
def _strategy_weights(strategy: str, moments, constraints: Constraints,
                      convention: FrequencyConvention, n: int) -> tuple:
    """Return ``(weights, certified)`` for one strategy on one estimation window."""
    if strategy == EQUAL_WEIGHT:
        weights = equal_weights(n)
        return weights, bool(check_feasibility(weights, constraints).ok)
    if moments is None:
        return None, False
    mu_annual, cov_annual = moments
    try:
        if strategy == 'min_variance':
            result = solve_min_variance(cov_annual, constraints)
        elif strategy == 'max_sharpe':
            result = solve_max_sharpe(mu_annual, cov_annual, convention.rf_cc_annual,
                                      constraints)
        else:
            raise PortfolioError('unknown_strategy', strategy=str(strategy))
    except PortfolioError:
        return None, False
    certified = bool(result.success and result.feasibility.ok
                     and result.message != TANGENCY_UNDEFINED)
    return result.weights, certified


def _plan_weights(returns: np.ndarray, dates: pd.DatetimeIndex, indices: np.ndarray,
                  config: BacktestConfig, convention: FrequencyConvention,
                  constraints: Constraints) -> tuple:
    """Chronological pass that fixes the weights of every rebalance.

    Returns ``(plan, events)`` with ``plan[strategy][i] = (weights, traded)``.
    When no verified optimum is available the previous weights are kept and
    nothing is traded (RF-BT7); on the first rebalance the equal-weight portfolio
    is used instead, since the portfolio has to be built.
    """
    n = returns.shape[1]
    strategies = tuple(config.strategies)
    needs_moments = any(name != EQUAL_WEIGHT for name in strategies)
    plan = {name: {} for name in strategies}
    previous = {name: None for name in strategies}
    events = []

    for position in indices:
        step = int(position)
        stamp = pd.Timestamp(dates[step]).date().isoformat()
        moments = None
        if needs_moments:
            try:
                _, _, mu_annual, cov_annual = estimate_moments(
                    returns[_estimation_slice(step, config.window)],
                    convention.trading_days)
                cov_annual, _, _ = ensure_psd(cov_annual)
                moments = (mu_annual, cov_annual)
            except PortfolioError:
                moments = None
        for name in strategies:
            weights, certified = _strategy_weights(name, moments, constraints,
                                                   convention, n)
            if certified:
                plan[name][step] = (np.asarray(weights, dtype=float), True)
                previous[name] = np.asarray(weights, dtype=float)
                continue
            if previous[name] is None:
                fallback = equal_weights(n)
                plan[name][step] = (fallback, True)
                previous[name] = fallback
                events.append(f'first_rebalance_equal_weights:{name}:{stamp}')
            else:
                plan[name][step] = (previous[name], False)
                events.append(f'rebalance_kept_previous_weights:{name}:{stamp}')
    return plan, tuple(events)


def _run_strategy(gross: np.ndarray, plan: dict, first: int, config: BacktestConfig
                  ) -> tuple:
    """Per-asset value accounting of one strategy; returns ``(equity, costs, n_trades)``."""
    total, n = gross.shape
    position = np.zeros(n, dtype=float)
    equity = np.empty(total - first, dtype=float)
    costs = 0.0
    trades = 0
    for step in range(first, total):
        if step > first:
            position = position * gross[step]
        if step in plan:
            weights, traded = plan[step]
            value = float(config.initial_capital) if step == first else float(position.sum())
            if traded and value > 0.0:
                current = (np.zeros(n, dtype=float) if step == first
                           else position / value)
                turnover = float(np.abs(weights - current).sum())
                cost = turnover * value * float(config.cost_rate)
                value -= cost
                costs += cost
                trades += 1
                position = weights * value
        equity[step - first] = float(position.sum())
    return equity, float(costs), int(trades)


def run_backtest(aligned: AlignedData, convention: FrequencyConvention,
                 constraints: Constraints, config: BacktestConfig) -> BacktestResult:
    """Run the walk-forward backtest of the configured strategies.

    All strategies share the evaluated span and the initial capital (CA-14); the
    individual asset columns of ``equity`` are buy and hold from
    ``initial_capital`` over the same span (RF-BT4).
    """
    returns = np.asarray(aligned.log_returns, dtype=float)
    if returns.ndim != 2 or returns.shape[0] < 2:
        raise PortfolioError('backtest_too_few_returns', n=int(returns.shape[0]))
    total, n = returns.shape
    dates = pd.DatetimeIndex(aligned.return_dates)
    names = tuple(aligned.names)
    window = int(config.window)
    if window < 1 or window >= total:
        raise PortfolioError('backtest_window_out_of_range', window=window,
                             n_returns=int(total))
    constraints.validate(n)
    strategies = tuple(config.strategies)
    if not strategies:
        raise PortfolioError('backtest_no_strategies')

    indices = generate_rebalance_indices(dates, config.rebalance_rule, window)
    if indices.size == 0:
        raise PortfolioError('backtest_no_rebalance_dates', window=window,
                             rule=str(config.rebalance_rule), n_returns=int(total))

    plan, events = _plan_weights(returns, dates, indices, config, convention, constraints)
    gross = gross_from_log(returns)
    first = int(indices[0])
    equity_index = dates[first:]

    columns = {}
    metrics_rows = {}
    weights_history = {}
    for name in strategies:
        equity, costs, trades = _run_strategy(gross, plan[name], first, config)
        columns[name] = equity
        metrics_rows[name] = performance_metrics(equity, equity_index, convention,
                                                 trades, costs)
        weights_history[name] = pd.DataFrame(
            np.vstack([plan[name][int(step)][0] for step in indices]),
            index=dates[indices], columns=list(names))

    factors = np.vstack([np.ones((1, n), dtype=float),
                         np.cumprod(gross[first + 1:total], axis=0)])
    buy_and_hold = float(config.initial_capital) * factors
    for position, name in enumerate(names):
        columns[name] = buy_and_hold[:, position]

    equity_frame = pd.DataFrame(columns, index=equity_index)
    drawdown_frame = pd.DataFrame(
        {column: compute_drawdown(equity_frame[column].to_numpy(dtype=float))
         for column in equity_frame.columns},
        index=equity_index)
    metrics_frame = pd.DataFrame.from_dict(metrics_rows, orient='index')

    return BacktestResult(equity=equity_frame, weights_history=weights_history,
                          metrics=metrics_frame, drawdown=drawdown_frame,
                          events=events, rebalance_dates=dates[indices])
