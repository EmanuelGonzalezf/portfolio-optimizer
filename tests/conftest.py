"""Shared fixtures (design.md section 21).

No fixture imports Streamlit and none reads a file from the user's workspace
(CA-38). The fixtures that depend on modules delivered by later features import
them lazily with ``pytest.importorskip``, so this file stays collectable while
those modules do not exist yet.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.data_loader import MissingPolicy, align_assets
from core.formatting import CurrencyFormat
from core.rng import child_rng
from core.stats import compute_asset_stats, compute_asset_stats_from_returns, estimate_moments
from core.types import AssetSeries, Constraints, FrequencyConvention
from data.generate_sample_data import generate_prices


def series_from_frame(frame: pd.DataFrame, *, adjusted: bool = True) -> list:
    """Turn a wide price frame into the ``AssetSeries`` list the loader produces."""
    result = []
    for column in frame.columns:
        values = frame[column].dropna()
        result.append(AssetSeries(
            name=str(column),
            source_file=f'{column}.csv',
            price_column='Adj Close' if adjusted else 'Close',
            is_adjusted=adjusted,
            dates=pd.DatetimeIndex(values.index),
            prices=values.to_numpy(dtype=float),
            rows_raw=int(len(values)),
            dropped_invalid=0,
            dropped_duplicates=0,
        ))
    return result


@pytest.fixture(scope='session')
def sample_prices() -> pd.DataFrame:
    """Five synthetic price series, seed 42 (data/generate_sample_data.py)."""
    return generate_prices(42)


@pytest.fixture(scope='session')
def convention() -> FrequencyConvention:
    """252 trading days, 4,5 % effective annual risk-free rate."""
    return FrequencyConvention(trading_days=252, rf_annual_effective=0.045)


@pytest.fixture(scope='session')
def aligned(sample_prices):
    """``AlignedData`` built from the synthetic frame, drop policy."""
    return align_assets(series_from_frame(sample_prices), MissingPolicy('drop'))


@pytest.fixture(scope='session')
def stats(aligned, convention):
    """``AssetStats`` estimated on the aligned synthetic universe."""
    return compute_asset_stats(aligned, convention)


@pytest.fixture
def constraints_long_only() -> Constraints:
    return Constraints(allow_short=False, max_weight=1.0)


@pytest.fixture
def constraints_short() -> Constraints:
    return Constraints(allow_short=True, max_weight=1.0, short_lower_bound=-1.0)


@pytest.fixture
def tiny_series():
    """Four dates and two assets whose log returns are exact by construction.

    Asset X ratios: 1.1, 1.1, 0.9. Asset Y ratios: 0.95, 1.1, 1.0.
    """
    dates = pd.DatetimeIndex(['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'])
    prices = np.array([[100.0, 200.0],
                       [110.0, 190.0],
                       [121.0, 209.0],
                       [108.9, 209.0]])
    ratios = np.array([[1.1, 0.95],
                       [1.1, 1.1],
                       [0.9, 1.0]])
    expected_log = np.log(ratios)
    return SimpleNamespace(dates=dates, prices=prices, names=('X', 'Y'),
                           expected_log_returns=expected_log)


@pytest.fixture
def singular_stats(convention):
    """Two identical assets, so the sample covariance is exactly singular.

    Exposes the raw (unrepaired) matrices plus the ``AssetStats`` that
    ``compute_asset_stats_from_returns`` produces after repairing them.
    """
    base = child_rng(7, 'paths').normal(0.0, 0.01, size=(300, 1))
    log_returns = np.hstack([base, base])
    mu_daily, cov_daily, mu_annual, cov_annual = estimate_moments(
        log_returns, convention.trading_days)
    repaired = compute_asset_stats_from_returns(log_returns, ('X', 'Y'),
                                                len(log_returns), convention)
    return SimpleNamespace(log_returns=log_returns, names=('X', 'Y'),
                           mu_daily=mu_daily, mu_annual=mu_annual,
                           cov_daily_raw=cov_daily, cov_annual_raw=cov_annual,
                           stats=repaired)


@pytest.fixture
def zero_return_path():
    """A (horizon, n) path of exactly zero log returns."""
    return np.zeros((252, 5), dtype=float)


@pytest.fixture
def fmt() -> CurrencyFormat:
    return CurrencyFormat()


# --------------------------------------------------------------------------- #
# Fixtures backed by modules that arrive with later features
# --------------------------------------------------------------------------- #
@pytest.fixture
def schedule_simple():
    """Twelve monthly installments of a fixed amount (core/cashflow.py).

    Signature taken literally from design.md section 10.1:
    ``build_schedule(first_date, frequency, n_installments, amount, growth_rate,
    external_income=0.0)``.
    """
    cashflow = pytest.importorskip('core.cashflow')
    return cashflow.build_schedule(pd.Timestamp('2024-02-01'), 'monthly', 12,
                                   1000.0, 0.0)


@pytest.fixture
def initial_position(sample_prices):
    """Capital 100.000 spread equally, no fees (core/cashflow.py).

    ``build_initial_position(capital, weights, prices, policy, *, integer_units,
    from_balance)`` per design.md section 10.2b: the fees travel inside the
    ``LiquidityPolicy``, not as loose arguments.
    """
    cashflow = pytest.importorskip('core.cashflow')
    prices = sample_prices.iloc[-1].to_numpy(dtype=float)
    policy = cashflow.LiquidityPolicy(pct_fee=0.0, fixed_fee=0.0,
                                      cash_reserve_initial=0.0)
    return cashflow.build_initial_position(100000.0, np.full(prices.size, 0.2),
                                           prices, policy, integer_units=False)


@pytest.fixture
def policy_zero_fees():
    """Liquidity policy without transaction costs (core/cashflow.py)."""
    cashflow = pytest.importorskip('core.cashflow')
    return cashflow.LiquidityPolicy(pct_fee=0.0, fixed_fee=0.0)


@pytest.fixture
def policy_with_fees():
    """Liquidity policy with both a percentage and a fixed fee (core/cashflow.py)."""
    cashflow = pytest.importorskip('core.cashflow')
    return cashflow.LiquidityPolicy(pct_fee=0.002, fixed_fee=5.0)


@pytest.fixture
def thresholds():
    """Default classification thresholds (core/scenarios.py)."""
    scenarios = pytest.importorskip('core.scenarios')
    return scenarios.CapacityThresholds()


# --------------------------------------------------------------------------- #
# Helpers shared by several test modules
# --------------------------------------------------------------------------- #
@pytest.fixture(scope='session')
def sqrt_trading_days(convention) -> float:
    return math.sqrt(convention.trading_days)
