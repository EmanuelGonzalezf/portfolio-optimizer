"""Cache boundaries (design.md section 13.2).

``core/`` stays free of Streamlit; the caching lives in these thin wrappers.
Their arguments are hashable primitives (bytes, tuples, floats, NumPy arrays)
and they return immutable dataclasses. No wrapper receives a mutable
``DataFrame``, and the sections copy every output frame before styling it.

Two things are deliberately **not** cached: the purchase balance and
``classify_payment_capacity``. The first costs microseconds; the second depends
on thresholds the user edits live.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import streamlit as st

from core.backtest import BacktestConfig, generate_rebalance_indices, run_backtest
from core.cashflow import InitialPosition, InstallmentSchedule, LiquidityPolicy
from core.data_loader import (ColumnMapping, MissingPolicy, ParsePreferences,
                             detect_columns, list_sheets, load_assets,
                             missing_policy_preview, parse_files, read_tabular)
from core.frontier import build_frontier, evaluate_weight_cloud, sample_random_weights
from core.optimize import (check_feasibility, equal_weights, solve_max_sharpe,
                           solve_min_variance)
from core.rng import child_rng
from core.scenarios import (aggregate_capacity_metrics, build_scenarios,
                            iter_path_batches, plan_path_simulation)
from core.stats import (compute_asset_stats_from_moments,
                        compute_asset_stats_from_returns, summarize_portfolio)
from core.types import AlignedData, Constraints, DataQualityReport, FrequencyConvention

__all__ = [
    'build_overrides_key',
    'build_prefs_key',
    'cached_inspect_file',
    'cached_missing_preview',
    'cached_load_assets',
    'cached_asset_stats',
    'cached_optimization',
    'cached_frontier',
    'cached_weight_cloud',
    'cached_backtest',
    'cached_capacity',
]

logger = logging.getLogger('portfolio_optimizer')


# --------------------------------------------------------------------------- #
# Serialisation of the two per-file mappings
# --------------------------------------------------------------------------- #
def build_overrides_key(overrides: dict | None) -> str:
    """Canonical JSON of the manual column overrides.

    ``cached_load_assets`` takes this as the ``overrides_key`` argument of the
    design signature: a single ``str`` that is both the cache discriminator and
    the payload, so the wrapper needs no unhashable ``dict``.
    """
    payload = {}
    for name, mapping in (overrides or {}).items():
        payload[str(name)] = {
            'file_name': str(mapping.file_name),
            'date_column': str(mapping.date_column),
            'price_columns': [str(column) for column in mapping.price_columns],
            'detected_format': str(mapping.detected_format),
            'adjusted_flags': {str(key): bool(value)
                               for key, value in (mapping.adjusted_flags or {}).items()},
        }
    return json.dumps(payload, sort_keys=True, separators=(',', ':'))


def build_prefs_key(prefs: dict | None) -> str:
    """Canonical JSON of the per-file parsing preferences."""
    payload = {str(name): list(item.to_tuple())
               for name, item in (prefs or {}).items()}
    return json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str)


def _overrides_from_key(overrides_key: str) -> dict:
    payload = json.loads(overrides_key or '{}')
    return {name: ColumnMapping(file_name=item['file_name'],
                                date_column=item['date_column'],
                                price_columns=tuple(item['price_columns']),
                                detected_format=item['detected_format'],
                                adjusted_flags=dict(item['adjusted_flags']))
            for name, item in payload.items()}


def _prefs_from_key(prefs_key: str) -> dict:
    payload = json.loads(prefs_key or '{}')
    return {name: ParsePreferences.from_tuple(tuple(item))
            for name, item in payload.items()}


# --------------------------------------------------------------------------- #
# 1. Data loading
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner='Procesando archivos…', max_entries=8)
def cached_load_assets(files: tuple, overrides_key: str, prefs_key: str,
                       policy: tuple, start: str | None, end: str | None) -> tuple:
    """``(AlignedData, tuple[ParseReport, ...])`` from the uploaded bytes."""
    return load_assets(list(files),
                       overrides=_overrides_from_key(overrides_key),
                       prefs=_prefs_from_key(prefs_key),
                       policy=MissingPolicy.from_tuple(policy),
                       start=start, end=end)


@st.cache_data(show_spinner=False, max_entries=32)
def cached_inspect_file(file_name: str, file_bytes: bytes, prefs_tuple: tuple) -> tuple:
    """``(frame, report, detected_mapping, sheet_names)`` for the mapping panel."""
    prefs = ParsePreferences.from_tuple(prefs_tuple)
    frame, report = read_tabular(file_bytes, file_name, prefs)
    mapping = detect_columns(frame, file_name)
    return frame, report, mapping, tuple(list_sheets(file_bytes, file_name))


@st.cache_data(show_spinner=False, max_entries=8)
def cached_missing_preview(files: tuple, overrides_key: str, prefs_key: str,
                           start: str | None, end: str | None,
                           limit_days: int) -> pd.DataFrame:
    """The three-row table of RF-D7: effect of each missing-value policy."""
    series, _ = parse_files(list(files),
                            overrides=_overrides_from_key(overrides_key),
                            prefs=_prefs_from_key(prefs_key))
    return missing_policy_preview(series, start, end, int(limit_days))


# --------------------------------------------------------------------------- #
# 2. Asset statistics
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def cached_asset_stats(log_returns: np.ndarray, names: tuple, n_obs: int,
                       trading_days: int, rf_annual_effective: float):
    """``AssetStats`` over the aligned returns."""
    return compute_asset_stats_from_returns(
        log_returns, tuple(names), int(n_obs),
        FrequencyConvention(int(trading_days), float(rf_annual_effective)))


# --------------------------------------------------------------------------- #
# 3. Optimization
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner='Optimizando…')
def cached_optimization(mu_annual: np.ndarray, cov_annual: np.ndarray, rf_cc: float,
                        allow_short: bool, max_weight: float, lb: float,
                        seed: int) -> dict:
    """``dict[strategy, PortfolioSummary]`` for the three strategies."""
    mu = np.asarray(mu_annual, dtype=float)
    cov = np.asarray(cov_annual, dtype=float)
    n = mu.size
    constraints = Constraints(bool(allow_short), float(max_weight), float(lb))
    constraints.validate(n)

    summaries: dict = {}
    min_var = solve_min_variance(cov, constraints, seed=int(seed))
    summaries['min_variance'] = summarize_portfolio(
        'min_variance', min_var.weights, mu, cov, float(rf_cc),
        converged=bool(min_var.success), feasibility=min_var.feasibility,
        diagnostics={'message': min_var.message, 'n_restarts': min_var.n_restarts,
                     'start_used': min_var.start_used})

    tangency = solve_max_sharpe(mu, cov, float(rf_cc), constraints, seed=int(seed))
    summaries['max_sharpe'] = summarize_portfolio(
        'max_sharpe', tangency.weights, mu, cov, float(rf_cc),
        converged=bool(tangency.success), feasibility=tangency.feasibility,
        diagnostics={'message': tangency.message, 'n_restarts': tangency.n_restarts,
                     'start_used': tangency.start_used})

    equal = equal_weights(n)
    summaries['equal_weight'] = summarize_portfolio(
        'equal_weight', equal, mu, cov, float(rf_cc), converged=True,
        feasibility=check_feasibility(equal, constraints),
        diagnostics={'message': '', 'n_restarts': 0, 'start_used': 'equal'})
    return summaries


# --------------------------------------------------------------------------- #
# 4. Frontier
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner='Calculando frontera…')
def cached_frontier(mu_annual: np.ndarray, cov_annual: np.ndarray, rf_cc: float,
                    allow_short: bool, max_weight: float, lb: float, n_points: int,
                    seed: int):
    """``FrontierResult`` with ``n_points`` requested targets."""
    constraints = Constraints(bool(allow_short), float(max_weight), float(lb))
    return build_frontier(np.asarray(mu_annual, dtype=float),
                          np.asarray(cov_annual, dtype=float), constraints,
                          float(rf_cc), int(n_points), seed=int(seed))


# --------------------------------------------------------------------------- #
# 5. Random weight cloud
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner='Simulando portafolios aleatorios…')
def cached_weight_cloud(mu_annual: np.ndarray, cov_annual: np.ndarray, rf_cc: float,
                        allow_short: bool, max_weight: float, lb: float, n_sims: int,
                        seed: int) -> tuple:
    """``(WeightSample, vol, ret, sharpe)``; the metrics come from the one
    vectorised counterpart of the scalar formulas (``evaluate_weight_cloud``)."""
    mu = np.asarray(mu_annual, dtype=float)
    cov = np.asarray(cov_annual, dtype=float)
    constraints = Constraints(bool(allow_short), float(max_weight), float(lb))
    sample = sample_random_weights(mu.size, int(n_sims), constraints,
                                   child_rng(int(seed), 'weights'))
    if sample.weights.size == 0:
        empty = np.zeros(0, dtype=float)
        return sample, empty, empty, empty
    vol, ret, sharpe = evaluate_weight_cloud(sample.weights, mu, cov, float(rf_cc))
    return sample, vol, ret, sharpe


# --------------------------------------------------------------------------- #
# 6. Backtesting
# --------------------------------------------------------------------------- #
def _aligned_for_backtest(log_returns: np.ndarray, return_dates_i8: np.ndarray,
                          names: tuple) -> AlignedData:
    """Rebuild the ``AlignedData`` that ``run_backtest`` consumes.

    ``run_backtest`` reads only ``log_returns``, ``return_dates`` and ``names``,
    which is why the design signature crosses the cache boundary with just
    those three. The price matrix is reconstructed from the returns so the
    dataclass stays internally consistent; the backtest never reads it.
    """
    returns = np.asarray(log_returns, dtype=float)
    return_dates = pd.DatetimeIndex(np.asarray(return_dates_i8, dtype='int64')
                                    .astype('datetime64[ns]'))
    factors = np.vstack([np.ones((1, returns.shape[1]), dtype=float),
                         np.exp(np.cumsum(returns, axis=0))])
    first = return_dates[0] - pd.Timedelta(days=1)
    dates = pd.DatetimeIndex([first]).append(return_dates)
    quality = DataQualityReport(per_asset=pd.DataFrame(), common_dates=len(dates),
                                missing_policy='drop', filled_cells={},
                                non_adjusted_assets=(),
                                returns_spanning_multiple_days=0,
                                max_return_span_days=1)
    return AlignedData(names=tuple(names), dates=dates, prices=100.0 * factors,
                       return_dates=return_dates, log_returns=returns, quality=quality)


@st.cache_data(show_spinner='Ejecutando backtesting…')
def cached_backtest(log_returns: np.ndarray, return_dates_i8: np.ndarray, names: tuple,
                    config_tuple: tuple, convention_tuple: tuple,
                    constraints_tuple: tuple):
    """``BacktestResult`` of the walk-forward run."""
    aligned = _aligned_for_backtest(log_returns, return_dates_i8, tuple(names))
    return run_backtest(aligned,
                        FrequencyConvention.from_tuple(convention_tuple),
                        Constraints.from_tuple(constraints_tuple),
                        BacktestConfig.from_tuple(config_tuple))


# --------------------------------------------------------------------------- #
# 7. Payment capacity
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner='Simulando trayectorias…')
def cached_capacity(mu_daily: np.ndarray, cov_daily: np.ndarray, weights: np.ndarray,
                    initial_tuple: tuple, schedule_tuple: tuple,
                    payment_indices: np.ndarray, calendar_i8: np.ndarray,
                    effective_payment_dates_i8: np.ndarray, policy_tuple: tuple,
                    convention_tuple: tuple, k_sigma: float, n_paths: int,
                    method: str, block_size: int,
                    historical_returns: np.ndarray | None, rebalance_rule: str | None,
                    seed: int):
    """``CapacityMetrics`` for the whole section 7.

    Reconstruction order, documented because it is load bearing:

    0. rebuild the two ``DatetimeIndex`` from their ``int64`` nanoseconds and
       derive ``horizon = len(calendar_i8) - 1`` (the calendar is the single
       source of the horizon, so no second number can disagree with it);
    1. ``rebalance_indices = generate_rebalance_indices(calendar, rule, window=0)``
       when a rule is active, ``None`` otherwise;
    2. ``from_tuple`` on ``InitialPosition``, ``InstallmentSchedule``,
       ``LiquidityPolicy`` and ``FrequencyConvention``;
    3. ``build_scenarios`` on the annual space rebuilt by ``core.stats``;
    4. ``plan_path_simulation`` (deterministic, consumes no randomness);
    5. ``iter_path_batches`` with ``child_rng(seed, 'paths')`` and
       ``historical_returns`` whenever the method is ``block_bootstrap``;
    6. ``aggregate_capacity_metrics``.
    """
    calendar = pd.DatetimeIndex(np.asarray(calendar_i8, dtype='int64')
                                .astype('datetime64[ns]'))
    horizon = int(len(calendar) - 1)
    effective_dates = pd.DatetimeIndex(
        np.asarray(effective_payment_dates_i8, dtype='int64').astype('datetime64[ns]'))

    rebalance_indices = None
    target_weights = None
    if rebalance_rule:
        rebalance_indices = generate_rebalance_indices(calendar, str(rebalance_rule), 0)
        target_weights = np.asarray(weights, dtype=float).ravel()

    initial = InitialPosition.from_tuple(initial_tuple)
    schedule = InstallmentSchedule.from_tuple(schedule_tuple)
    policy = LiquidityPolicy.from_tuple(policy_tuple)
    convention = FrequencyConvention.from_tuple(convention_tuple)

    mu = np.asarray(mu_daily, dtype=float).ravel()
    cov = np.asarray(cov_daily, dtype=float)
    names = tuple(f'a{i}' for i in range(mu.size))
    stats = compute_asset_stats_from_moments(mu, cov, names, horizon, convention)

    scenarios = build_scenarios(stats, weights, float(k_sigma))

    spec = plan_path_simulation(cov, horizon, int(n_paths), method=str(method),
                                block_size=int(block_size), seed=int(seed))
    history = np.asarray(historical_returns, dtype=float) \
        if str(method) == 'block_bootstrap' else None
    batches = iter_path_batches(mu, cov, horizon, int(n_paths), method=str(method),
                               rng=child_rng(int(seed), 'paths'),
                               block_size=int(block_size), historical_returns=history)

    logger.info('capacity simulation: method=%s n_paths=%d horizon=%d',
                method, int(n_paths), horizon)
    return aggregate_capacity_metrics(
        stats, weights, initial, schedule, np.asarray(payment_indices, dtype=int),
        policy, scenarios, batches, spec, convention,
        rebalance_indices=rebalance_indices, target_weights=target_weights,
        effective_payment_dates=effective_dates)
