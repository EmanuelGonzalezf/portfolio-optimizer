"""Shared dataclasses and coded exceptions for the whole calculation core.

Design references: design.md section 2.1 (idiomatic boundary: core/ emits English
snake_case codes only), section 4 (data model) and section 3 (frequency convention).

Every dataclass here is ``frozen=True``. Numpy arrays are normalised with
``np.asarray(..., dtype=float)`` and marked non-writeable, so a frozen instance
cannot be mutated through one of its arrays either.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    'PortfolioError',
    'DataLoadError',
    'AlignmentError',
    'InfeasibleConstraintsError',
    'ScheduleError',
    'ExportError',
    'NOTE_CODES',
    'AssetSeries',
    'DataQualityReport',
    'AlignedData',
    'FrequencyConvention',
    'AssetStats',
    'Constraints',
    'FeasibilityCheck',
    'PortfolioSummary',
    'freeze_array',
]


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class PortfolioError(Exception):
    """Base error. Carries an English code plus numeric context.

    The code is stable and testable; ``ui/labels.py`` is the only place that
    turns it into user-facing prose. The ``Exception`` message built here is a
    technical log line, never something rendered to the user.
    """

    def __init__(self, code: str, **ctx: Any) -> None:
        self.code = code
        self.ctx = ctx
        super().__init__(f'{code} {ctx}')


class DataLoadError(PortfolioError):
    """Raised while reading, detecting or parsing an input file."""


class AlignmentError(PortfolioError):
    """Raised when the aligned universe cannot support the analysis."""


class InfeasibleConstraintsError(PortfolioError):
    """Raised when the constraint set admits no feasible weight vector."""


class ScheduleError(PortfolioError):
    """Raised when an installment schedule cannot be built."""


class ExportError(PortfolioError):
    """Raised when the workbook cannot be assembled."""


# Ledger note codes (design.md section 2.1). ui/labels.py::NOTE_LABELS translates them.
NOTE_CODES = (
    'sale_not_executed_costs_exceed_proceeds',
    'rebalance_skipped_no_market_value',
    'rebalance_skipped_cost_exceeds_value',
    'price_above_assignable_budget',
    'two_installments_same_business_day',
    'no_scheduled_obligations',
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def freeze_array(values: Any) -> np.ndarray:
    """Return ``values`` as a contiguous float array that cannot be written to."""
    arr = np.asarray(values, dtype=float)
    if arr.base is not None or arr.flags.writeable:
        arr = np.array(arr, dtype=float, copy=True)
    arr.flags.writeable = False
    return arr


def _freeze_index(values: Any) -> pd.DatetimeIndex:
    """Return ``values`` as a ``pd.DatetimeIndex`` (immutable by construction)."""
    if isinstance(values, pd.DatetimeIndex):
        return values
    return pd.DatetimeIndex(values)


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AssetSeries:
    """One cleaned price series: sorted dates, unique, strictly positive prices."""

    name: str
    source_file: str
    price_column: str
    is_adjusted: bool
    dates: pd.DatetimeIndex
    prices: np.ndarray
    rows_raw: int
    dropped_invalid: int
    dropped_duplicates: int

    def __post_init__(self) -> None:
        object.__setattr__(self, 'dates', _freeze_index(self.dates))
        object.__setattr__(self, 'prices', freeze_array(self.prices))


@dataclass(frozen=True)
class DataQualityReport:
    """Per-asset and per-universe quality counters produced by the loader.

    ``per_asset`` columns are English (asset, rows_raw, invalid, duplicates,
    first, last, max_gap_days, lost_in_alignment); the UI renames them with
    ui/labels.py::COLUMN_LABELS.
    """

    per_asset: pd.DataFrame
    common_dates: int
    missing_policy: str
    filled_cells: dict
    non_adjusted_assets: tuple
    returns_spanning_multiple_days: int
    max_return_span_days: int
    warning_codes: tuple = ()


@dataclass(frozen=True)
class AlignedData:
    """Aligned price matrix plus the log returns realised on ``return_dates``.

    ``return_dates == dates[1:]`` and ``log_returns[t] = ln(prices[t+1]/prices[t])``,
    so the return at row ``t`` is realised on ``return_dates[t] = dates[t+1]``.
    """

    names: tuple
    dates: pd.DatetimeIndex
    prices: np.ndarray
    return_dates: pd.DatetimeIndex
    log_returns: np.ndarray
    quality: DataQualityReport

    def __post_init__(self) -> None:
        object.__setattr__(self, 'names', tuple(self.names))
        object.__setattr__(self, 'dates', _freeze_index(self.dates))
        object.__setattr__(self, 'return_dates', _freeze_index(self.return_dates))
        object.__setattr__(self, 'prices', freeze_array(self.prices))
        object.__setattr__(self, 'log_returns', freeze_array(self.log_returns))


@dataclass(frozen=True)
class FrequencyConvention:
    """Single owner of the three risk-free conversions (design.md section 3).

        rf_cc_annual    = ln(1 + rf_annual_effective)
        rf_cc_daily     = rf_cc_annual / trading_days
        rf_simple_daily = (1 + rf_annual_effective)**(1/trading_days) - 1

    ``rf_cc_annual`` is the only rate used by Sharpe; ``rf_simple_daily`` is the
    only rate used to remunerate a money balance; ``rf_cc_daily`` is exposed for
    diagnostics.
    """

    trading_days: int = 252
    rf_annual_effective: float = 0.045

    @property
    def rf_cc_annual(self) -> float:
        return float(math.log1p(self.rf_annual_effective))

    @property
    def rf_cc_daily(self) -> float:
        return float(self.rf_cc_annual / self.trading_days)

    @property
    def rf_simple_daily(self) -> float:
        return float((1.0 + self.rf_annual_effective) ** (1.0 / self.trading_days) - 1.0)

    def to_tuple(self) -> tuple:
        return (int(self.trading_days), float(self.rf_annual_effective))

    @classmethod
    def from_tuple(cls, data: tuple) -> 'FrequencyConvention':
        trading_days, rf_annual_effective = data
        return cls(trading_days=int(trading_days),
                   rf_annual_effective=float(rf_annual_effective))


@dataclass(frozen=True)
class AssetStats:
    """Estimated moments in both spaces plus the per-asset descriptive figures."""

    names: tuple
    n_obs: int
    mu_daily: np.ndarray
    cov_daily: np.ndarray
    mu_annual: np.ndarray
    cov_annual: np.ndarray
    sigma_annual: np.ndarray
    corr: np.ndarray
    sharpe_individual: np.ndarray
    min_return: np.ndarray
    max_return: np.ndarray
    psd_repaired: bool
    ridge_epsilon: float
    convention: FrequencyConvention

    def __post_init__(self) -> None:
        object.__setattr__(self, 'names', tuple(self.names))
        for name in ('mu_daily', 'cov_daily', 'mu_annual', 'cov_annual',
                     'sigma_annual', 'corr', 'sharpe_individual',
                     'min_return', 'max_return'):
            object.__setattr__(self, name, freeze_array(getattr(self, name)))


@dataclass(frozen=True)
class Constraints:
    """Weight-space constraint set: budget plus box bounds."""

    allow_short: bool = False
    max_weight: float = 1.0
    short_lower_bound: float = -1.0

    def bounds(self, n: int) -> list:
        """Box bounds: (0, max_weight) long-only, (short_lower_bound, max_weight) with shorts."""
        lower = self.short_lower_bound if self.allow_short else 0.0
        return [(float(lower), float(self.max_weight))] * int(n)

    def validate(self, n: int) -> None:
        """Raise ``InfeasibleConstraintsError`` when no weight vector can satisfy the set."""
        if self.max_weight <= 0.0:
            raise InfeasibleConstraintsError('max_weight_not_positive',
                                             max_weight=float(self.max_weight))
        if self.max_weight * n < 1.0 - 1e-12:
            raise InfeasibleConstraintsError('max_weight_below_equal_share',
                                             max_weight=float(self.max_weight),
                                             n=int(n),
                                             minimum=1.0 / float(n))
        if self.allow_short and self.short_lower_bound >= 0.0:
            raise InfeasibleConstraintsError('short_lower_bound_not_negative',
                                             short_lower_bound=float(self.short_lower_bound))

    def to_tuple(self) -> tuple:
        return (bool(self.allow_short), float(self.max_weight), float(self.short_lower_bound))

    @classmethod
    def from_tuple(cls, data: tuple) -> 'Constraints':
        allow_short, max_weight, short_lower_bound = data
        return cls(allow_short=bool(allow_short),
                   max_weight=float(max_weight),
                   short_lower_bound=float(short_lower_bound))


@dataclass(frozen=True)
class FeasibilityCheck:
    """Audit of a weight vector against the budget equality and the box bounds."""

    sum_ok: bool
    bounds_ok: bool
    finite_ok: bool
    sum_error: float
    max_bound_violation: float

    @property
    def ok(self) -> bool:
        return bool(self.sum_ok and self.bounds_ok and self.finite_ok)


@dataclass(frozen=True)
class PortfolioSummary:
    """Ex-ante figures of one strategy, plus its certification flag.

    ``certified`` is the only field the UI reads to decide between the verified
    badge and the warning badge: an approximation is never shown as an optimum.
    """

    strategy: str
    weights: np.ndarray
    expected_return_annual: float
    volatility_annual: float
    sharpe: float
    n_positive: int
    converged: bool
    certified: bool
    feasibility: FeasibilityCheck
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'weights', freeze_array(self.weights))
