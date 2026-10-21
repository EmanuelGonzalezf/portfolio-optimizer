"""Efficient frontier and constraint-aware weight cloud (design.md sections 7.2, 7.3).

Two public producers plus the single vectorized evaluator:

* ``build_frontier`` solves ``n_points`` target-return problems with continuation
  (each point starts from the previous solution) and fills ``vol``, ``ret`` and
  ``sharpe`` by CALLING ``evaluate_weight_cloud`` over the resolved weight
  matrix, never with local arithmetic.
* ``sample_random_weights`` draws random portfolios that satisfy the active
  constraints, with four schemes and finite fallback cascades.
* ``evaluate_weight_cloud`` is the only vectorized counterpart of
  ``core.stats.portfolio_volatility`` / ``portfolio_return`` / ``portfolio_sharpe``
  in the project (FD-5, design.md section 5).

This module never imports Streamlit and emits no user-facing text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.optimize import project_to_bounded_simplex, solve_min_variance_target
from core.stats import VARIANCE_FLOOR
from core.types import Constraints, freeze_array

__all__ = [
    'FrontierResult',
    'WeightSample',
    'SAMPLING_METHODS',
    'MIN_USABLE_SAMPLES',
    'build_frontier',
    'sample_random_weights',
    'evaluate_weight_cloud',
]

SAMPLING_METHODS = ('dirichlet', 'dirichlet_rejection', 'reprojection',
                    'normal_hyperplane', 'normal_hyperplane_narrow')
OVERSAMPLE_FACTORS = (1.5, 3.0, 6.0, 12.0)
NARROW_FACTORS = (6.0, 12.0)
WIDE_SD = 0.5
NARROW_SD = 0.25
MIN_USABLE_SAMPLES = 100
SUM_TOL = 1e-10
BOUND_TOL = 1e-12
SAMPLING_FAILED = 'weight_sampling_failed_relax_weight_cap_or_short_bound'


@dataclass(frozen=True)
class FrontierResult:
    """Resolved frontier points plus the bookkeeping of what could not be solved.

    ``points`` carries ``vol``, ``ret``, ``sharpe`` and ``w_0 .. w_{n-1}``,
    ordered by ``vol``. ``skipped_targets`` holds the target returns SLSQP could
    not reach under the active bounds, which is what the UI reports as
    'N of M points computed'.
    """

    points: pd.DataFrame
    requested: int
    computed: int
    skipped_targets: tuple


@dataclass(frozen=True)
class WeightSample:
    """A cloud of feasible random weights and how it was produced.

    ``weights`` is ``(n_samples, n)``, or ``(0, n)`` when the whole cascade could
    not deliver at least ``MIN_USABLE_SAMPLES`` feasible vectors; in that case
    ``message`` carries an actionable code and the UI draws the frontier without
    a cloud. ``method`` and ``acceptance_rate`` are always reported, because a
    ``method`` other than ``'dirichlet'`` biases the density towards the bounds.
    """

    weights: np.ndarray
    method: str
    acceptance_rate: float
    requested: int
    message: str = ''

    def __post_init__(self) -> None:
        object.__setattr__(self, 'weights', freeze_array(self.weights))


# --------------------------------------------------------------------------- #
# The single vectorized counterpart of the three scalar formulas
# --------------------------------------------------------------------------- #
def evaluate_weight_cloud(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray,
                          rf: float) -> tuple:
    """Return exactly, in this order: ``(vol, ret, sharpe)``, each of shape ``(m,)``.

    THE single vectorized counterpart of ``core.stats.portfolio_volatility`` /
    ``portfolio_return`` / ``portfolio_sharpe`` (design.md section 5):

        vol    = sqrt(einsum('ij,jk,ik->i', W, cov, W))
        ret    = W @ mu
        sharpe = (ret - rf) / vol

    Every consumer of a weight MATRIX goes through here, ``build_frontier``
    included, so the frontier columns and the Monte Carlo cloud cannot diverge.
    The variance is floored at ``core.stats.VARIANCE_FLOOR``, exactly as the
    scalar route does, which is what keeps the equality at 1e-12.
    """
    matrix = np.atleast_2d(np.asarray(weights, dtype=float))
    cov_matrix = np.atleast_2d(np.asarray(cov, dtype=float))
    mu_vector = np.asarray(mu, dtype=float).ravel()
    quadratic = np.einsum('ij,jk,ik->i', matrix, cov_matrix, matrix)
    vol = np.sqrt(np.maximum(quadratic, VARIANCE_FLOOR))
    ret = matrix @ mu_vector
    sharpe = (ret - float(rf)) / vol
    return vol, ret, sharpe


# --------------------------------------------------------------------------- #
# Frontier
# --------------------------------------------------------------------------- #
def _target_range(mu: np.ndarray, constraints: Constraints) -> tuple:
    """Target-return range of RF-F2."""
    low, high = float(mu.min()), float(mu.max())
    if constraints.allow_short:
        span = max(high - low, 0.01)
        return low - 0.5 * span, high + 0.5 * span
    return low + 1e-6, high - 1e-6


def _frontier_columns(n: int) -> list:
    return ['vol', 'ret', 'sharpe'] + [f'w_{i}' for i in range(n)]


def build_frontier(mu: np.ndarray, cov: np.ndarray, constraints: Constraints,
                   rf: float, n_points: int, *, seed=None) -> FrontierResult:
    """Solve the frontier by continuation and evaluate it in a single pass.

    Each target is solved with ``solve_min_variance_target(..., x0=previous)``;
    the first point uses the standard multistart. Targets SLSQP cannot reach
    under the active bounds land in ``skipped_targets`` instead of producing a
    point that is not on the frontier.
    """
    mu_vector = np.asarray(mu, dtype=float).ravel()
    cov_matrix = np.atleast_2d(np.asarray(cov, dtype=float))
    n = mu_vector.size
    requested = int(n_points)
    constraints.validate(n)

    low, high = _target_range(mu_vector, constraints)
    targets = np.linspace(low, high, requested) if requested > 1 else np.array([high])

    resolved = []
    skipped = []
    previous = None
    for target in targets:
        result = solve_min_variance_target(mu_vector, cov_matrix, float(target),
                                           constraints, x0=previous, seed=seed)
        if result.success:
            resolved.append(np.asarray(result.weights, dtype=float))
            previous = result.weights
        else:
            skipped.append(float(target))

    if resolved:
        weight_matrix = np.vstack(resolved)
        vol, ret, sharpe = evaluate_weight_cloud(weight_matrix, mu_vector,
                                                 cov_matrix, rf)
        frame = pd.DataFrame({'vol': vol, 'ret': ret, 'sharpe': sharpe})
        for index in range(n):
            frame[f'w_{index}'] = weight_matrix[:, index]
        frame = frame.sort_values('vol').reset_index(drop=True)
    else:
        frame = pd.DataFrame(columns=_frontier_columns(n), dtype=float)

    return FrontierResult(points=frame, requested=requested,
                          computed=int(len(resolved)),
                          skipped_targets=tuple(skipped))


# --------------------------------------------------------------------------- #
# Weight cloud
# --------------------------------------------------------------------------- #
def _feasible_rows(candidates: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Keep the rows that satisfy ``|sum(w)-1| <= 1e-10`` and the box bounds (CA-10)."""
    if candidates.size == 0:
        return candidates.reshape(0, candidates.shape[-1])
    keep = (np.abs(candidates.sum(axis=1) - 1.0) <= SUM_TOL)
    keep &= np.all(candidates >= lower - BOUND_TOL, axis=1)
    keep &= np.all(candidates <= upper + BOUND_TOL, axis=1)
    return candidates[keep]


def _hyperplane_normal(rng, n: int, size: int, sd: float) -> np.ndarray:
    """``w ~ N(1/n, sd)`` projected onto ``sum(w) = 1`` (the prototype's projectHyperplane)."""
    draw = rng.normal(1.0 / n, sd, size=(size, n))
    return draw - (draw.sum(axis=1, keepdims=True) - 1.0) / n


def sample_random_weights(n_assets: int, n_samples: int, constraints: Constraints,
                          rng) -> WeightSample:
    """Draw ``n_samples`` random weight vectors that respect the active constraints.

    Four schemes, every cascade finite (design.md section 7.3):

    1. long-only without a binding cap: Dirichlet, acceptance 1,0;
    2. long-only with a cap: oversample and reject, 4 rounds with factors
       1.5 / 3 / 6 / 12;
    3. inefficient rejection: iterative reprojection (clip plus redistribution),
       at most 50 iterations, ``method='reprojection'``;
    4. with shorts: normal draws projected onto the budget hyperplane and
       rejected against the bounds, 4 wide rounds, then 2 narrow rounds, then
       reprojection.

    Fewer than ``MIN_USABLE_SAMPLES`` feasible vectors after the whole cascade
    returns an empty sample plus an actionable code.
    """
    n = int(n_assets)
    requested = int(n_samples)
    lower, upper = constraints.bounds(n)[0]
    alpha = np.ones(n, dtype=float)

    if requested <= 0:
        return WeightSample(weights=np.empty((0, n), dtype=float), method='dirichlet',
                            acceptance_rate=1.0, requested=requested)

    if not constraints.allow_short and upper >= 1.0:
        cloud = rng.dirichlet(alpha, requested)
        return WeightSample(weights=cloud, method='dirichlet', acceptance_rate=1.0,
                            requested=requested)

    chunks = []
    kept = 0
    generated = 0
    method = 'dirichlet_rejection' if not constraints.allow_short else 'normal_hyperplane'

    def collect(candidates: np.ndarray) -> None:
        nonlocal kept, generated
        generated += int(candidates.shape[0])
        good = _feasible_rows(candidates, lower, upper)
        if good.shape[0]:
            chunks.append(good)
            kept += int(good.shape[0])

    if not constraints.allow_short:
        for factor in OVERSAMPLE_FACTORS:
            if kept >= requested:
                break
            size = max(1, math.ceil((requested - kept) * factor))
            collect(rng.dirichlet(alpha, size))
    else:
        for factor in OVERSAMPLE_FACTORS:
            if kept >= requested:
                break
            size = max(1, math.ceil((requested - kept) * factor))
            collect(_hyperplane_normal(rng, n, size, WIDE_SD))
        if kept < requested:
            method = 'normal_hyperplane_narrow'
            for factor in NARROW_FACTORS:
                if kept >= requested:
                    break
                size = max(1, math.ceil((requested - kept) * factor))
                collect(_hyperplane_normal(rng, n, size, NARROW_SD))

    if kept < requested:
        method = 'reprojection'
        size = max(1, (requested - kept) * 2)
        if constraints.allow_short:
            raw = _hyperplane_normal(rng, n, size, NARROW_SD)
        else:
            raw = rng.dirichlet(alpha, size)
        collect(project_to_bounded_simplex(raw, lower, upper))

    cloud = np.vstack(chunks) if chunks else np.empty((0, n), dtype=float)
    acceptance = float(kept / generated) if generated else 0.0
    if cloud.shape[0] < MIN_USABLE_SAMPLES:
        return WeightSample(weights=np.empty((0, n), dtype=float), method=method,
                            acceptance_rate=acceptance, requested=requested,
                            message=SAMPLING_FAILED)
    return WeightSample(weights=cloud[:requested], method=method,
                        acceptance_rate=acceptance, requested=requested)
