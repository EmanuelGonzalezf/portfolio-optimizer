"""Returns, moments, annualisation and portfolio metrics.

Single implementation of every formula of design.md section 3, copied literally
in the docstrings below. Estimation space is daily log returns; the money space
is derived from them with ``gross_from_log``, never re-estimated.

    r[t,i] = ln(P[t,i] / P[t-1,i])                      t = 1..T,  i = 1..n
    mu_daily[i]    = mean_t r[t,i]
    cov_daily[i,j] = sample covariance, ddof = 1
    mu_annual      = mu_daily * TDY                      TDY = trading days/year (252)
    cov_annual     = cov_daily * TDY
    sigma_annual[i]= sqrt(cov_annual[i,i])
    corr[i,j]      = cov_annual[i,j] / (sigma_annual[i] * sigma_annual[j])

This module never imports Streamlit and emits no user-facing text.
"""

from __future__ import annotations

import math

import numpy as np

from core.types import (AlignedData, AssetStats, FeasibilityCheck, FrequencyConvention,
                        PortfolioError, PortfolioSummary)

__all__ = [
    'compute_log_returns',
    'simple_from_log',
    'gross_from_log',
    'estimate_moments',
    'correlation_from_covariance',
    'ensure_psd',
    'cholesky_factor',
    'portfolio_return',
    'portfolio_variance',
    'portfolio_volatility',
    'portfolio_sharpe',
    'portfolio_log_return_series',
    'portfolio_simple_return_series',
    'realized_metrics',
    'annualize_moments',
    'compute_asset_stats_from_returns',
    'compute_asset_stats_from_moments',
    'compute_asset_stats',
    'summarize_portfolio',
]

TANGENCY_UNDEFINED = 'tangency_undefined_rf_above_all_assets'
VARIANCE_FLOOR = 1e-16
POSITIVE_WEIGHT_TOL = 1e-6


# --------------------------------------------------------------------------- #
# Return spaces
# --------------------------------------------------------------------------- #
def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    """``r[t,i] = ln(P[t+1,i] / P[t,i])`` over a (T+1, n) price matrix."""
    values = np.asarray(prices, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.shape[0] < 2:
        return np.empty((0, values.shape[1]), dtype=float)
    return np.diff(np.log(values), axis=0)


def simple_from_log(log_returns: np.ndarray) -> np.ndarray:
    """``simple[t,i] = exp(r[t,i]) - 1`` -- money space, derived, never re-estimated."""
    return np.expm1(np.asarray(log_returns, dtype=float))


def gross_from_log(log_returns: np.ndarray) -> np.ndarray:
    """``gross[t,i] = exp(r[t,i])`` -- the compounding factor the engines apply per asset."""
    return np.exp(np.asarray(log_returns, dtype=float))


def estimate_moments(log_returns: np.ndarray, trading_days: int
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return exactly, in this order: ``(mu_daily, cov_daily, mu_annual, cov_annual)``.

        mu_daily[i]    = mean_t r[t,i]
        cov_daily[i,j] = sample covariance, ddof = 1
        mu_annual      = mu_daily * TDY
        cov_annual     = cov_daily * TDY
    """
    returns = np.asarray(log_returns, dtype=float)
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)
    if returns.shape[0] < 2:
        raise PortfolioError('too_few_returns_to_estimate', n=int(returns.shape[0]))
    mu_daily = returns.mean(axis=0)
    cov_daily = np.atleast_2d(np.cov(returns, rowvar=False, ddof=1))
    mu_annual, cov_annual = annualize_moments(mu_daily, cov_daily, trading_days)
    return mu_daily, cov_daily, mu_annual, cov_annual


def annualize_moments(mu_daily: np.ndarray, cov_daily: np.ndarray, trading_days: int
                      ) -> tuple[np.ndarray, np.ndarray]:
    """``(mu_daily * TDY, cov_daily * TDY)``: the single annualization site (FD-5)."""
    tdy = float(trading_days)
    return (np.asarray(mu_daily, dtype=float) * tdy,
            np.atleast_2d(np.asarray(cov_daily, dtype=float)) * tdy)


def correlation_from_covariance(cov: np.ndarray) -> np.ndarray:
    """``corr[i,j] = cov[i,j] / (sqrt(cov[i,i]) * sqrt(cov[j,j]))``.

    Scale invariant, so the correlation of the daily covariance and of the
    annualised one are the same matrix: the TDY factor cancels.
    """
    matrix = np.atleast_2d(np.asarray(cov, dtype=float))
    sigma = np.sqrt(np.clip(np.diag(matrix), 0.0, None))
    denominator = np.outer(sigma, sigma)
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.where(denominator > 0.0, matrix / denominator, 0.0)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


# --------------------------------------------------------------------------- #
# Covariance conditioning
# --------------------------------------------------------------------------- #
def ensure_psd(cov: np.ndarray, ridge_scale: float = 1e-8
               ) -> tuple[np.ndarray, bool, float]:
    """Return ``(cov_fixed, repaired, epsilon)``.

    Cascade: try a Cholesky factorisation; on failure add a ridge of
    ``ridge_scale * trace(cov)`` to the diagonal and retry; on a second failure
    decompose with ``eigh``, clip the eigenvalues to
    ``max(lambda, 1e-12 * trace / n)``, rebuild and symmetrise with ``(A + A')/2``.
    """
    matrix = np.atleast_2d(np.asarray(cov, dtype=float)).copy()
    n = matrix.shape[0]
    try:
        np.linalg.cholesky(matrix)
        return matrix, False, 0.0
    except np.linalg.LinAlgError:
        pass

    trace = float(np.trace(matrix))
    epsilon = float(ridge_scale * trace)
    ridged = matrix + epsilon * np.eye(n)
    try:
        np.linalg.cholesky(ridged)
        return ridged, True, epsilon
    except np.linalg.LinAlgError:
        pass

    floor = float(1e-12 * trace / n) if n else 0.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    clipped = np.maximum(eigenvalues, floor)
    rebuilt = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    rebuilt = 0.5 * (rebuilt + rebuilt.T)
    return rebuilt, True, floor


def cholesky_factor(cov: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return ``(L, repaired)`` with ``L L' == cov_fixed``.

    Only place in the project that both repairs and factorises a covariance
    matrix, so ``psd_repaired`` has a single origin.
    """
    matrix = np.atleast_2d(np.asarray(cov, dtype=float))
    try:
        return np.linalg.cholesky(matrix), False
    except np.linalg.LinAlgError:
        pass
    fixed, _, _ = ensure_psd(matrix)
    try:
        return np.linalg.cholesky(fixed), True
    except np.linalg.LinAlgError as exc:
        raise PortfolioError('covariance_not_factorizable', n=int(matrix.shape[0])) from exc


# --------------------------------------------------------------------------- #
# Portfolio metrics (ex ante)
# --------------------------------------------------------------------------- #
def portfolio_return(w: np.ndarray, mu: np.ndarray) -> float:
    """``w' mu``."""
    return float(np.asarray(w, dtype=float) @ np.asarray(mu, dtype=float))


def portfolio_variance(w: np.ndarray, cov: np.ndarray) -> float:
    """``max(w' Sigma w, 1e-16)`` -- floored so the Sharpe denominator stays finite."""
    weights = np.asarray(w, dtype=float)
    matrix = np.atleast_2d(np.asarray(cov, dtype=float))
    return float(max(weights @ matrix @ weights, VARIANCE_FLOOR))


def portfolio_volatility(w: np.ndarray, cov: np.ndarray) -> float:
    """``sqrt(w' Sigma w)``."""
    return float(math.sqrt(portfolio_variance(w, cov)))


def portfolio_sharpe(w: np.ndarray, mu: np.ndarray, cov: np.ndarray,
                     rf_cc_annual: float) -> float:
    """Single Sharpe definition of the project:

        sharpe(w) = (w' mu_annual - rf_cc_annual) / sqrt(w' cov_annual w)
    """
    excess = portfolio_return(w, mu) - float(rf_cc_annual)
    return float(excess / portfolio_volatility(w, cov))


def portfolio_log_return_series(log_returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``R @ w`` -- log space. Consumed only by the ex-ante consistency check."""
    returns = np.atleast_2d(np.asarray(log_returns, dtype=float))
    return returns @ np.asarray(weights, dtype=float)


def portfolio_simple_return_series(log_returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``(exp(R) - 1) @ w`` -- money space of a fixed-weight portfolio.

    Consumed only by the cumulative-return chart and its verification; no engine
    reads it, since the engines compound per asset with ``gross_from_log``.
    """
    returns = np.atleast_2d(np.asarray(log_returns, dtype=float))
    return np.expm1(returns) @ np.asarray(weights, dtype=float)


# --------------------------------------------------------------------------- #
# Realised (ex post) metrics
# --------------------------------------------------------------------------- #
def realized_metrics(equity: np.ndarray, convention: FrequencyConvention) -> dict:
    """Single ex-post implementation: total_return, cagr, mu_cc_real, vol_real,
    sharpe_real, plus 'degenerate' (1.0 or 0.0) -- see design.md section 3.

        n_days     = len(equity) - 1
        CAGR       = (V_T / V_0)**(TDY / n_days) - 1
        mu_cc_real = ln(V_T / V_0) * TDY / n_days
        vol_real   = std(diff(ln(equity)), ddof=1) * sqrt(TDY)
        sharpe_real= (mu_cc_real - rf_cc_annual) / vol_real

    ``n_days`` is ALWAYS ``len(equity) - 1``; callers slice the equity array to
    evaluate a sub-period. One trading day per row (see the alignment caveat in
    design.md section 3). If ``equity.min() <= 0`` or ``n_days <= 0``, returns
    every metric as ``float('nan')`` plus ``'degenerate': 1.0``, and NEVER
    raises. Callers must not render nan metrics (the UI prints an em dash).

    Only consumer: ``core/backtest.performance_metrics``.
    """
    series = np.asarray(equity, dtype=float).ravel()
    n_days = series.size - 1
    nan = float('nan')
    if series.size == 0 or n_days <= 0 or not np.all(np.isfinite(series)) or series.min() <= 0.0:
        return {'total_return': nan, 'cagr': nan, 'mu_cc_real': nan,
                'vol_real': nan, 'sharpe_real': nan, 'degenerate': 1.0}

    tdy = float(convention.trading_days)
    total_return = float(series[-1] / series[0] - 1.0)
    cagr = float((series[-1] / series[0]) ** (tdy / n_days) - 1.0)
    mu_cc_real = float(math.log(series[-1] / series[0]) * tdy / n_days)
    log_steps = np.diff(np.log(series))
    vol_real = float(log_steps.std(ddof=1) * math.sqrt(tdy)) if n_days > 1 else nan
    sharpe_real = nan
    if math.isfinite(vol_real) and vol_real > 0.0:
        sharpe_real = float((mu_cc_real - convention.rf_cc_annual) / vol_real)
    return {'total_return': total_return, 'cagr': cagr, 'mu_cc_real': mu_cc_real,
            'vol_real': vol_real, 'sharpe_real': sharpe_real, 'degenerate': 0.0}


# --------------------------------------------------------------------------- #
# Aggregates
# --------------------------------------------------------------------------- #
def compute_asset_stats_from_returns(log_returns: np.ndarray, names: tuple,
                                     n_obs: int,
                                     convention: FrequencyConvention) -> AssetStats:
    """Estimate every per-asset figure from a (T, n) matrix of daily log returns.

    ``sharpe_individual[i] = portfolio_sharpe(e_i, mu_annual, cov_annual, rf)``
    with ``e_i`` the canonical vector: the per-asset Sharpe has no second
    implementation.
    """
    returns = np.atleast_2d(np.asarray(log_returns, dtype=float))
    mu_daily, cov_daily, _, _ = estimate_moments(returns, convention.trading_days)
    return _assemble_asset_stats(mu_daily, cov_daily, tuple(names), int(n_obs),
                                 convention, returns.min(axis=0),
                                 returns.max(axis=0))


def compute_asset_stats_from_moments(mu_daily: np.ndarray, cov_daily: np.ndarray,
                                     names: tuple, n_obs: int,
                                     convention: FrequencyConvention) -> AssetStats:
    """Assemble ``AssetStats`` from daily moments that were already estimated.

    Entry point for the caching layer: ``ui/compute.py`` crosses the cache
    boundary with ``mu_daily``/``cov_daily`` (design.md section 13.2) and still
    needs the annual space that the scenario builder reads. Annualization and
    the per-asset Sharpe stay here, so no formula is duplicated outside
    ``core/`` (RNF-2, FD-5). ``min_return`` and ``max_return`` are ``nan``:
    they are sample descriptives that this entry point does not receive.
    """
    mu = np.asarray(mu_daily, dtype=float).ravel()
    empty = np.full(mu.size, np.nan)
    return _assemble_asset_stats(mu, np.atleast_2d(np.asarray(cov_daily, dtype=float)),
                                 tuple(names), int(n_obs), convention, empty, empty)


def _assemble_asset_stats(mu_daily: np.ndarray, cov_daily: np.ndarray, names: tuple,
                          n_obs: int, convention: FrequencyConvention,
                          min_return: np.ndarray,
                          max_return: np.ndarray) -> AssetStats:
    """Shared tail of both entry points: repair, annualize and score.

    ``sharpe_individual[i] = portfolio_sharpe(e_i, mu_annual, cov_annual, rf)``
    with ``e_i`` the canonical vector: the per-asset Sharpe has no second
    implementation.
    """
    mu_annual, cov_annual = annualize_moments(mu_daily, cov_daily,
                                              convention.trading_days)
    cov_annual_fixed, repaired, epsilon = ensure_psd(cov_annual)
    if repaired:
        cov_daily = cov_annual_fixed / float(convention.trading_days)

    sigma_annual = np.sqrt(np.clip(np.diag(cov_annual_fixed), 0.0, None))
    corr = correlation_from_covariance(cov_annual_fixed)

    n = int(mu_annual.size)
    rf_cc_annual = convention.rf_cc_annual
    sharpe_individual = np.empty(n, dtype=float)
    for i in range(n):
        canonical = np.zeros(n, dtype=float)
        canonical[i] = 1.0
        sharpe_individual[i] = portfolio_sharpe(canonical, mu_annual,
                                                cov_annual_fixed, rf_cc_annual)

    return AssetStats(names=tuple(names), n_obs=int(n_obs),
                      mu_daily=mu_daily, cov_daily=cov_daily,
                      mu_annual=mu_annual, cov_annual=cov_annual_fixed,
                      sigma_annual=sigma_annual, corr=corr,
                      sharpe_individual=sharpe_individual,
                      min_return=min_return, max_return=max_return,
                      psd_repaired=bool(repaired), ridge_epsilon=float(epsilon),
                      convention=convention)


def compute_asset_stats(aligned: AlignedData,
                        convention: FrequencyConvention) -> AssetStats:
    """Thin adapter over ``compute_asset_stats_from_returns``."""
    return compute_asset_stats_from_returns(aligned.log_returns, aligned.names,
                                            len(aligned.log_returns), convention)


def summarize_portfolio(strategy: str, weights: np.ndarray, mu_annual: np.ndarray,
                        cov_annual: np.ndarray, rf_cc_annual: float, *,
                        converged: bool, feasibility: FeasibilityCheck,
                        diagnostics: dict) -> PortfolioSummary:
    """Assemble a ``PortfolioSummary`` from the three arrays plus the scalar rate.

    ``certified = converged AND feasibility.ok``, forced to ``False`` when the
    solver reported ``tangency_undefined_rf_above_all_assets``: that result is
    reproducible and feasible, but it is not the tangency portfolio.
    """
    w = np.asarray(weights, dtype=float)
    diagnostics = dict(diagnostics or {})
    certified = bool(converged and feasibility.ok)
    if diagnostics.get('message') == TANGENCY_UNDEFINED:
        certified = False
    return PortfolioSummary(
        strategy=str(strategy),
        weights=w,
        expected_return_annual=portfolio_return(w, mu_annual),
        volatility_annual=portfolio_volatility(w, cov_annual),
        sharpe=portfolio_sharpe(w, mu_annual, cov_annual, rf_cc_annual),
        n_positive=int(np.sum(w > POSITIVE_WEIGHT_TOL)),
        converged=bool(converged),
        certified=certified,
        feasibility=feasibility,
        diagnostics=diagnostics,
    )
