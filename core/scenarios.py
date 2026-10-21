"""Deterministic scenarios, path Monte Carlo and payment-capacity category.

Design references: section 11.2 (scenarios), 11.3 (path simulation, decision D-6)
and 11.4 (metrics and interpretation, decision D-4).

Which moments the Monte Carlo uses, stated explicitly: ALWAYS the historical
``mu_daily`` and ``cov_daily`` of the selected period, that is the CENTRAL
scenario. ``iter_path_batches`` takes no scenario and there is no "conservative
Monte Carlo": the conservative and favorable scenarios are purely deterministic
(section 11.2). Hence ``p_cover_all`` measures dispersion around the central case
while ``conservative_covers_all`` is a separate parallel stress, and the rules of
section 11.4 combine both precisely because they measure different things.

This module never imports Streamlit and emits no user-facing text: categories,
criteria and note codes are English keys that ui/labels.py renders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from core.cashflow import (InitialPosition, InstallmentSchedule, LiquidityPolicy,
                           required_additional_capital, simulate_cashflow_paths,
                           simulate_cashflow_single)
from core.stats import cholesky_factor, portfolio_return, portfolio_volatility
from core.types import AssetStats, FrequencyConvention, PortfolioError, freeze_array

__all__ = [
    'SCENARIO_KEYS',
    'PATH_METHODS',
    'CAPACITY_CATEGORIES',
    'PERCENTILE_KEYS',
    'BLOCK_SIZE_RANGE',
    'MAX_FLOATS',
    'NO_SCHEDULED_OBLIGATIONS',
    'ScenarioAssumption',
    'PathSimulationSpec',
    'CapacityMetrics',
    'CapacityThresholds',
    'CriterionEval',
    'CapacityVerdict',
    'build_scenarios',
    'deterministic_path',
    'plan_path_simulation',
    'iter_path_batches',
    'aggregate_capacity_metrics',
    'classify_payment_capacity',
]

SCENARIO_KEYS = ('conservative', 'central', 'favorable')
PATH_METHODS = ('gaussian_cholesky', 'block_bootstrap')
CAPACITY_CATEGORIES = ('sufficient', 'tight', 'high_risk', 'insufficient')
PERCENTILE_KEYS = ('p5', 'p25', 'p50', 'p75', 'p95')
PERCENTILE_VALUES = (5.0, 25.0, 50.0, 75.0, 95.0)
BLOCK_SIZE_RANGE = (2, 252)
MAX_FLOATS = 20_000_000
NO_SCHEDULED_OBLIGATIONS = 'no_scheduled_obligations'


# --------------------------------------------------------------------------- #
# Deterministic scenarios (section 11.2)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScenarioAssumption:
    """One parallel shift of the expected return, ``mu_annual + k*sigma_annual``.

    There is no ``label`` field: presentation is not a concern of core/ (FD-3,
    RNF-2); ui/labels.py::SCENARIO_LABELS holds the visible names.

    ``vol_portfolio_annual`` is the HISTORICAL reference volatility that
    generates the shift, taken from ``core.stats.portfolio_volatility``. It is
    identical in the three scenarios and it is NOT the volatility applied to the
    deterministic path, which is zero because the path spreads the return
    uniformly (the UI table states this in four columns).
    """

    key: str
    k_sigma: float
    mu_annual_by_asset: np.ndarray
    mu_portfolio_annual: float
    vol_portfolio_annual: float

    def __post_init__(self) -> None:
        object.__setattr__(self, 'mu_annual_by_asset',
                           freeze_array(self.mu_annual_by_asset).ravel())


def build_scenarios(stats: AssetStats, weights: np.ndarray,
                    k_sigma: float = 1.0) -> dict:
    """``mu_annual_by_asset(k) = mu_annual + k*sigma_annual`` for the three keys.

    ``k`` takes ``-k_sigma``, ``0``, ``+k_sigma``. With ``k_sigma = 0`` the three
    scenarios collapse into the central one, which the UI warns about: the stress
    content of rule 4 of section 11.4 would be empty.
    """
    w = np.asarray(weights, dtype=float).ravel()
    mu_annual = np.asarray(stats.mu_annual, dtype=float)
    sigma_annual = np.asarray(stats.sigma_annual, dtype=float)
    cov_annual = np.asarray(stats.cov_annual, dtype=float)
    if w.size != mu_annual.size:
        raise PortfolioError('scenario_inputs_length_mismatch', n_weights=int(w.size),
                             n_assets=int(mu_annual.size))
    shift = float(k_sigma)
    volatility = portfolio_volatility(w, cov_annual)
    scenarios = {}
    for key, factor in zip(SCENARIO_KEYS, (-shift, 0.0, shift)):
        mu_by_asset = mu_annual + factor * sigma_annual
        scenarios[key] = ScenarioAssumption(
            key=key, k_sigma=factor, mu_annual_by_asset=mu_by_asset,
            mu_portfolio_annual=portfolio_return(w, mu_by_asset),
            vol_portfolio_annual=volatility)
    return scenarios


def deterministic_path(mu_annual_by_asset: np.ndarray, trading_days: int,
                       horizon: int) -> np.ndarray:
    """Shape ``(1, horizon, n)``: a one-path batch.

    ``r[t,i] = mu_annual_by_asset[i] / trading_days`` every day, so the log
    return is spread uniformly and the applied volatility is zero.
    ``simulate_cashflow_single`` and ``required_additional_capital`` consume the
    2D form, i.e. ``deterministic_path(...)[0]`` of shape ``(horizon, n)``; row
    ``j`` of that array moves ``calendar[j] -> calendar[j+1]`` (section 10.1).
    """
    mu = np.asarray(mu_annual_by_asset, dtype=float).ravel()
    steps = int(horizon)
    if steps < 1:
        raise PortfolioError('cashflow_horizon_not_positive', horizon=steps)
    daily = mu / float(trading_days)
    return np.array(np.broadcast_to(daily, (1, steps, mu.size)), dtype=float, copy=True)


# --------------------------------------------------------------------------- #
# Path Monte Carlo (section 11.3)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PathSimulationSpec:
    """What is stored in session and exported; never the simulated matrices."""

    method: str
    psd_repaired: bool
    block_size: int
    n_paths: int
    horizon: int
    n_assets: int
    batch_size: int
    seed: int

    def to_tuple(self) -> tuple:
        return (str(self.method), bool(self.psd_repaired),
                -1 if self.block_size is None else int(self.block_size),
                int(self.n_paths), int(self.horizon), int(self.n_assets),
                int(self.batch_size), int(self.seed))

    @classmethod
    def from_tuple(cls, data: tuple) -> 'PathSimulationSpec':
        method, repaired, block_size, n_paths, horizon, n_assets, batch, seed = data
        size = int(block_size)
        return cls(method=str(method), psd_repaired=bool(repaired),
                   block_size=None if size < 0 else size, n_paths=int(n_paths),
                   horizon=int(horizon), n_assets=int(n_assets),
                   batch_size=int(batch), seed=int(seed))


def _batch_size(horizon: int, n_assets: int, max_floats: int) -> int:
    """``max(1, max_floats // (horizon*n_assets))``: a deterministic function of
    the parameters only, so seed reproducibility never depends on machine memory.
    """
    per_path = max(int(horizon) * int(n_assets), 1)
    return int(max(1, int(max_floats) // per_path))


def plan_path_simulation(cov_daily: np.ndarray, horizon: int, n_paths: int, *,
                         method: str, block_size, seed: int,
                         max_floats: int = MAX_FLOATS) -> PathSimulationSpec:
    """Pure planning step: decides ``batch_size`` and whether Sigma needed repair.

    Draws not a single random number, so it can feed the session and the export
    sheet without touching the generator that ``iter_path_batches`` consumes.
    """
    key = str(method)
    if key not in PATH_METHODS:
        raise ValueError('unknown_path_method')
    matrix = np.atleast_2d(np.asarray(cov_daily, dtype=float))
    n_assets = int(matrix.shape[0])
    _, repaired = cholesky_factor(matrix)
    size = None if key != 'block_bootstrap' else int(block_size)
    return PathSimulationSpec(method=key, psd_repaired=bool(repaired), block_size=size,
                              n_paths=int(n_paths), horizon=int(horizon),
                              n_assets=n_assets,
                              batch_size=_batch_size(horizon, n_assets, max_floats),
                              seed=int(seed))


def iter_path_batches(mu_daily: np.ndarray, cov_daily: np.ndarray, horizon: int,
                      n_paths: int, *, method: str = 'gaussian_cholesky', rng,
                      block_size: int = 21, historical_returns=None,
                      max_floats: int = MAX_FLOATS) -> Iterator[np.ndarray]:
    """Yield consecutive batches of shape ``(batch, horizon, n_assets)``.

    Every validation runs eagerly, BEFORE any batch is produced, so an invalid
    call raises at call time and not on the first ``next()``.

    ``gaussian_cholesky`` (default): ``L, psd_repaired = cholesky_factor(cov_daily)``
    and, per batch, ``r = mu_daily + Z @ L.T`` with ``Z ~ N(0,1)``, which
    preserves Sigma exactly because ``cov(r) = L L' = Sigma``. The factorisation
    ALWAYS goes through ``core.stats.cholesky_factor``, never through
    ``np.linalg.cholesky`` directly, so ``psd_repaired`` has a single origin and
    there are not two repair heuristics. Simulating each asset separately is
    forbidden.

    ``block_bootstrap``: uniform start indices and blocks of ``block_size``
    COMPLETE historical rows with circular wrapping, WITHOUT recentring, so the
    implied drift is the realised historical one and may differ from
    ``mu_daily * horizon``.

    Each batch is drawn from the SAME generator in order, so the random sequence
    (and therefore the result) is identical however many batches are needed.
    """
    key = str(method)
    if key not in PATH_METHODS:
        raise ValueError('unknown_path_method')
    steps = int(horizon)
    total = int(n_paths)
    if steps < 1:
        raise ValueError('horizon_not_positive')
    if total < 1:
        raise ValueError('n_paths_not_positive')

    mu = np.asarray(mu_daily, dtype=float).ravel()
    matrix = np.atleast_2d(np.asarray(cov_daily, dtype=float))
    n_assets = int(mu.size)
    if matrix.shape != (n_assets, n_assets):
        raise ValueError('covariance_shape_mismatch')

    history = None
    size = None
    if key == 'block_bootstrap':
        if historical_returns is None:
            raise ValueError('block_bootstrap_requires_historical_returns')
        history = np.atleast_2d(np.asarray(historical_returns, dtype=float))
        if history.shape[1] != n_assets:
            raise ValueError('covariance_shape_mismatch')
        size = int(block_size)
        low, high = BLOCK_SIZE_RANGE
        if size < low or size > high or size > history.shape[0]:
            raise ValueError('block_size_out_of_range')

    factor = None
    if key == 'gaussian_cholesky':
        factor, _ = cholesky_factor(matrix)

    batch = _batch_size(steps, n_assets, max_floats)

    def generate() -> Iterator[np.ndarray]:
        remaining = total
        while remaining > 0:
            count = int(min(batch, remaining))
            if key == 'gaussian_cholesky':
                noise = rng.standard_normal((count, steps, n_assets))
                yield mu + noise @ factor.T
            else:
                rows = history.shape[0]
                n_blocks = int(np.ceil(steps / size))
                starts = rng.integers(0, rows, size=(count, n_blocks))
                offsets = np.arange(size, dtype=np.int64)
                positions = (starts[:, :, None] + offsets[None, None, :]) % rows
                positions = positions.reshape(count, n_blocks * size)[:, :steps]
                yield history[positions]
            remaining -= count

    return generate()


# --------------------------------------------------------------------------- #
# Capacity metrics and classification (section 11.4)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CapacityThresholds:
    """The five editable thresholds. Rules reference them by name, never by value."""

    p_sufficient: float = 0.90
    p_tight: float = 0.75
    p_high_risk: float = 0.50
    margin_min: float = 0.10
    conservative_deficit_tolerance: float = 0.10

    def to_tuple(self) -> tuple:
        return (float(self.p_sufficient), float(self.p_tight), float(self.p_high_risk),
                float(self.margin_min), float(self.conservative_deficit_tolerance))

    @classmethod
    def from_tuple(cls, data: tuple) -> 'CapacityThresholds':
        p_sufficient, p_tight, p_high_risk, margin_min, tolerance = data
        return cls(p_sufficient=float(p_sufficient), p_tight=float(p_tight),
                   p_high_risk=float(p_high_risk), margin_min=float(margin_min),
                   conservative_deficit_tolerance=float(tolerance))


@dataclass(frozen=True)
class CapacityMetrics:
    """Everything the section 7 blocks and the export need, already computed."""

    initial_capital: float
    n_installments: int
    total_scheduled: float
    deterministic: dict
    central_ledger: pd.DataFrame
    central_covers_all: bool
    conservative_covers_all: bool
    conservative_deficit_ratio: float
    margin_ratio: float
    p_cover_all: float
    coverage_by_installment: np.ndarray
    mc_final_wealth: np.ndarray
    mc_covered_all: np.ndarray
    final_wealth_percentiles: dict
    additional_capital_required: float
    path_spec: PathSimulationSpec
    note_codes: tuple = ()


@dataclass(frozen=True)
class CriterionEval:
    """One row of the five-row criteria table the UI paints (RF-C22)."""

    name: str
    value: float
    threshold: float
    comparator: str
    passed: bool


@dataclass(frozen=True)
class CapacityVerdict:
    """Category plus the full criteria table and which rule decided.

    No ``narrative`` field: the prose of RF-C20 is presentation and lives in
    ui/narrative.py::build_narrative.
    """

    category: str
    criteria: tuple
    rule_fired: int


def aggregate_capacity_metrics(stats: AssetStats, weights: np.ndarray,
                               initial: InitialPosition, schedule: InstallmentSchedule,
                               payment_indices: np.ndarray, policy: LiquidityPolicy,
                               scenarios: dict, path_batches: Iterable[np.ndarray],
                               path_spec: PathSimulationSpec,
                               convention: FrequencyConvention, *,
                               rebalance_indices=None, target_weights=None,
                               effective_payment_dates: pd.DatetimeIndex = None
                               ) -> CapacityMetrics:
    """Run the three deterministic scenarios and the MC batches through the SAME engine.

    The central scenario goes through ``simulate_cashflow_single`` so its
    row-per-installment ledger (indexed by ``effective_payment_dates``) is kept;
    the other two and the Monte Carlo go through ``simulate_cashflow_paths``.
    There is no second bookkeeping implementation anywhere.

    Degenerate calendar (``total_scheduled <= 0``): the two ratios would be
    ``inf`` and ``nan``, and a ``nan`` would make rules 2 and 3 false so the
    verdict would come out ``sufficient`` by arithmetic accident. Treated
    explicitly: ``margin_ratio = inf``, ``conservative_deficit_ratio = 0.0`` and
    ``note_codes += ('no_scheduled_obligations',)``; section 7 then omits the
    verdict instead of answering a question nobody asked.
    """
    w = np.asarray(weights, dtype=float).ravel()
    if w.size != len(stats.names):
        raise PortfolioError('scenario_inputs_length_mismatch', n_weights=int(w.size),
                             n_assets=int(len(stats.names)))
    missing = [key for key in SCENARIO_KEYS if key not in scenarios]
    if missing:
        raise PortfolioError('scenario_key_missing', key=str(missing[0]))

    horizon = int(path_spec.horizon)
    trading_days = int(convention.trading_days)
    rf_simple_daily = convention.rf_simple_daily
    note_codes = set()

    deterministic = {}
    central_ledger = None
    central_path = None
    for key in SCENARIO_KEYS:
        path = deterministic_path(scenarios[key].mu_annual_by_asset, trading_days,
                                  horizon)
        if key == 'central':
            central_path = path[0]
            central_ledger, outcome = simulate_cashflow_single(
                central_path, initial, schedule, payment_indices, policy,
                rf_simple_daily, rebalance_indices, target_weights,
                effective_payment_dates)
        else:
            outcome = simulate_cashflow_paths([path], initial, schedule,
                                              payment_indices, policy,
                                              rf_simple_daily, rebalance_indices,
                                              target_weights)
        note_codes |= set(outcome.note_codes)
        deterministic[key] = outcome

    monte_carlo = simulate_cashflow_paths(path_batches, initial, schedule,
                                          payment_indices, policy, rf_simple_daily,
                                          rebalance_indices, target_weights)
    note_codes |= set(monte_carlo.note_codes)

    total_scheduled = schedule.total_scheduled
    central = deterministic['central']
    conservative = deterministic['conservative']
    central_covers_all = bool(central.covered_all[0])
    conservative_covers_all = bool(conservative.covered_all[0])
    if total_scheduled <= 0.0:
        margin_ratio = float('inf')
        conservative_deficit_ratio = 0.0
        note_codes.add(NO_SCHEDULED_OBLIGATIONS)
    else:
        margin_ratio = float(central.final_wealth[0] / total_scheduled)
        conservative_deficit_ratio = float(conservative.cumulative_deficit[0]
                                           / total_scheduled)

    final_wealth = np.asarray(monte_carlo.final_wealth, dtype=float)
    percentiles = {name: float(np.percentile(final_wealth, value))
                   for name, value in zip(PERCENTILE_KEYS, PERCENTILE_VALUES)}

    additional = required_additional_capital(
        initial.capital, w, initial.prices, schedule, payment_indices, central_path,
        policy, rf_simple_daily, rebalance_indices=rebalance_indices,
        target_weights=target_weights, integer_units=initial.integer_units)

    return CapacityMetrics(
        initial_capital=float(initial.capital),
        n_installments=int(schedule.n_installments),
        total_scheduled=float(total_scheduled),
        deterministic=deterministic,
        central_ledger=central_ledger,
        central_covers_all=central_covers_all,
        conservative_covers_all=conservative_covers_all,
        conservative_deficit_ratio=float(conservative_deficit_ratio),
        margin_ratio=float(margin_ratio),
        p_cover_all=float(np.mean(np.asarray(monte_carlo.covered_all, dtype=float))),
        coverage_by_installment=np.asarray(monte_carlo.coverage_by_installment),
        mc_final_wealth=np.asarray(monte_carlo.final_wealth),
        mc_covered_all=np.asarray(monte_carlo.covered_all),
        final_wealth_percentiles=percentiles,
        additional_capital_required=float(additional),
        path_spec=path_spec,
        note_codes=tuple(sorted(note_codes)),
    )


def _evaluate(name: str, value, threshold, comparator: str) -> CriterionEval:
    """Build one ``CriterionEval`` applying ``comparator`` to ``(value, threshold)``."""
    if comparator == '>=':
        passed = value >= threshold
    elif comparator == '<=':
        passed = value <= threshold
    elif comparator == '>':
        passed = value > threshold
    elif comparator == '<':
        passed = value < threshold
    elif comparator == '==':
        passed = value == threshold
    else:
        raise PortfolioError('unknown_comparator', comparator=str(comparator))
    return CriterionEval(name=name, value=value, threshold=threshold,
                         comparator=comparator, passed=bool(passed))


def classify_payment_capacity(metrics: CapacityMetrics,
                              thresholds: CapacityThresholds = CapacityThresholds()
                              ) -> CapacityVerdict:
    """Apply the four rules of section 11.4 in order; the first match wins.

    1. ``insufficient`` if not ``central_covers_all`` OR ``p_cover_all < p_high_risk``.
    2. ``high_risk`` if ``p_high_risk <= p_cover_all < p_tight`` OR
       ``conservative_deficit_ratio > conservative_deficit_tolerance``.
    3. ``tight`` if ``p_tight <= p_cover_all < p_sufficient``, OR
       ``p_cover_all >= p_sufficient`` but not ``conservative_covers_all``, OR
       ``margin_ratio < margin_min``.
    4. ``sufficient`` only if all of: ``p_cover_all >= p_sufficient``,
       ``central_covers_all``, ``conservative_covers_all`` and
       ``margin_ratio >= margin_min``.

    The five ``CriterionEval`` are built ALWAYS in the order and with the
    comparators of the fixed mapping table, whichever rule fired: the table
    informs the user about every criterion, and ``rule_fired`` says which one
    decided.
    """
    th = thresholds
    p_cover = float(metrics.p_cover_all)
    margin = float(metrics.margin_ratio)
    deficit_ratio = float(metrics.conservative_deficit_ratio)
    central_ok = bool(metrics.central_covers_all)
    conservative_ok = bool(metrics.conservative_covers_all)

    criteria = (
        _evaluate('p_cover_all', p_cover, float(th.p_sufficient), '>='),
        _evaluate('central_covers_all', central_ok, True, '=='),
        _evaluate('conservative_covers_all', conservative_ok, True, '=='),
        _evaluate('margin_ratio', margin, float(th.margin_min), '>='),
        _evaluate('conservative_deficit_ratio', deficit_ratio,
                  float(th.conservative_deficit_tolerance), '<='),
    )

    if (not central_ok) or p_cover < th.p_high_risk:
        return CapacityVerdict(category='insufficient', criteria=criteria, rule_fired=1)
    if (th.p_high_risk <= p_cover < th.p_tight) or deficit_ratio > th.conservative_deficit_tolerance:
        return CapacityVerdict(category='high_risk', criteria=criteria, rule_fired=2)
    if (th.p_tight <= p_cover < th.p_sufficient) \
            or (p_cover >= th.p_sufficient and not conservative_ok) \
            or margin < th.margin_min:
        return CapacityVerdict(category='tight', criteria=criteria, rule_fired=3)
    return CapacityVerdict(category='sufficient', criteria=criteria, rule_fired=4)
