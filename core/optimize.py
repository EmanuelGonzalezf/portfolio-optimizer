"""Portfolio optimization with SLSQP and analytic jacobians (design.md section 7.1).

Three solvers, one shared multistart and one shared feasibility audit:

    minimum variance        f = w'Sigma w                grad = 2 Sigma w
    target return           same f plus w'mu = target    jac  = mu
    maximum Sharpe          f = -(w'mu - rf)/s           grad = -(mu/s - e Sigma w/s^3)
                            with s = sqrt(w'Sigma w) and e = w'mu - rf

The closed forms are not the production route: they are warm starts and the
oracle the CA-5 tests compare against. Every solver returns an
``OptimizationResult`` whose ``feasibility`` is the only certificate the UI
reads, so an approximation is never presented as an optimum.

This module never imports Streamlit and emits no user-facing text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from core.rng import child_rng
from core.stats import TANGENCY_UNDEFINED, VARIANCE_FLOOR, ensure_psd
from core.types import (Constraints, FeasibilityCheck, InfeasibleConstraintsError,
                        PortfolioError, freeze_array)

__all__ = [
    'OptimizationResult',
    'SLSQP_OPTIONS',
    'START_LABELS',
    'FEASIBILITY_TOL',
    'equal_weights',
    'max_feasible_return',
    'closed_form_min_variance',
    'closed_form_tangency',
    'check_feasibility',
    'project_to_bounded_simplex',
    'solve_min_variance',
    'solve_max_sharpe',
    'solve_min_variance_target',
]

SLSQP_OPTIONS = {'maxiter': 500, 'ftol': 1e-12}
START_LABELS = ('equal', 'min_variance', 'closed_form',
                'random_1', 'random_2', 'random_3', 'random_4', 'random_5')
FEASIBILITY_TOL = 1e-6
TARGET_TOL = 1e-6
PROJECTION_ITERATIONS = 50


@dataclass(frozen=True)
class OptimizationResult:
    """Outcome of one solve.

    ``objective_value`` is the value of the MINIMISED objective at ``weights``
    (so it is the variance for the two quadratic problems and the negated Sharpe
    ratio for the tangency problem), ``n_restarts`` the number of SLSQP runs
    actually executed and ``start_used`` the label of the accepted start.
    """

    weights: np.ndarray
    success: bool
    message: str
    objective_value: float
    n_restarts: int
    feasibility: FeasibilityCheck
    start_used: str

    def __post_init__(self) -> None:
        object.__setattr__(self, 'weights', freeze_array(self.weights))


# --------------------------------------------------------------------------- #
# Closed forms and feasibility
# --------------------------------------------------------------------------- #
def equal_weights(n: int) -> np.ndarray:
    """``1/n`` in every position: the benchmark portfolio and the first warm start."""
    count = int(n)
    if count <= 0:
        raise PortfolioError('asset_count_not_positive', n=count)
    return np.full(count, 1.0 / count, dtype=float)


def max_feasible_return(mu: np.ndarray, constraints: Constraints) -> float:
    """Closed-form max of ``w'mu`` over ``{sum(w)=1, lb <= w <= ub}``.

    Greedy over ``mu`` descending, saturating the upper bound from the lower
    bound. The solution of this linear program over the bounded simplex is exact,
    so no optimizer is involved. Used by the degenerate guard of
    ``solve_max_sharpe``: comparing ``rf`` against ``max(mu)`` would miss the
    case where a binding ``max_weight`` keeps every feasible portfolio below the
    risk-free rate.
    """
    values = np.asarray(mu, dtype=float).ravel()
    n = values.size
    lower, upper = constraints.bounds(n)[0]
    order = np.argsort(values)[::-1]
    w = np.full(n, lower, dtype=float)
    remaining = 1.0 - w.sum()
    for i in order:
        add = min(upper - lower, remaining)
        w[i] += add
        remaining -= add
        if remaining <= 1e-15:
            break
    return float(w @ values)


def closed_form_min_variance(cov: np.ndarray) -> np.ndarray:
    """``Sigma^-1 1 / (1' Sigma^-1 1)`` -- unconstrained minimum variance."""
    matrix = np.atleast_2d(np.asarray(cov, dtype=float))
    fixed, _, _ = ensure_psd(matrix)
    n = fixed.shape[0]
    raw = np.linalg.solve(fixed, np.ones(n, dtype=float))
    total = float(raw.sum())
    if abs(total) < 1e-300:
        raise PortfolioError('closed_form_min_variance_undefined', n=int(n))
    return raw / total


def closed_form_tangency(mu: np.ndarray, cov: np.ndarray, rf: float) -> np.ndarray:
    """``Sigma^-1 (mu - rf) / (1' Sigma^-1 (mu - rf))`` -- unconstrained tangency.

    When the denominator vanishes the tangency portfolio is undefined; the
    equal-weight vector is returned as the warm start (same fallback the
    prototype applied), and the caller decides what to report.
    """
    excess = np.asarray(mu, dtype=float).ravel() - float(rf)
    matrix = np.atleast_2d(np.asarray(cov, dtype=float))
    fixed, _, _ = ensure_psd(matrix)
    raw = np.linalg.solve(fixed, excess)
    total = float(raw.sum())
    if abs(total) < 1e-12:
        return equal_weights(fixed.shape[0])
    return raw / total


def check_feasibility(w: np.ndarray, constraints: Constraints,
                      tol: float = FEASIBILITY_TOL) -> FeasibilityCheck:
    """Audit a weight vector against the budget equality and the box bounds."""
    weights = np.asarray(w, dtype=float).ravel()
    lower, upper = constraints.bounds(weights.size)[0]
    finite_ok = bool(np.all(np.isfinite(weights)))
    sum_error = float(abs(weights.sum() - 1.0)) if finite_ok else float('inf')
    violation = 0.0
    if finite_ok and weights.size:
        violation = float(max(0.0, float(np.max(lower - weights)),
                              float(np.max(weights - upper))))
    return FeasibilityCheck(sum_ok=bool(finite_ok and sum_error <= tol),
                            bounds_ok=bool(finite_ok and violation <= tol),
                            finite_ok=finite_ok,
                            sum_error=sum_error,
                            max_bound_violation=violation)


def project_to_bounded_simplex(weights: np.ndarray, lower: float, upper: float,
                               max_iter: int = PROJECTION_ITERATIONS) -> np.ndarray:
    """Clip to ``[lower, upper]`` and spread the residual over the free components.

    Shared by the multistart of this module and by the reprojection cascade of
    ``core/frontier.sample_random_weights`` (design.md section 7.3), so there is
    one projection rule and not two. Accepts a single vector or a matrix of row
    vectors and always returns the same shape. The loop is bounded, so a point
    that cannot be projected comes back infeasible instead of hanging; the caller
    checks it with ``check_feasibility``.
    """
    raw = np.asarray(weights, dtype=float)
    single = raw.ndim == 1
    matrix = np.atleast_2d(raw).astype(float, copy=True)
    for _ in range(int(max_iter)):
        np.clip(matrix, lower, upper, out=matrix)
        residual = 1.0 - matrix.sum(axis=1)
        if np.all(np.abs(residual) <= 1e-12):
            break
        headroom = np.where(residual[:, None] > 0.0,
                            matrix < upper - 1e-15,
                            matrix > lower + 1e-15)
        free = headroom.sum(axis=1)
        share = np.where(free > 0, residual / np.maximum(free, 1), 0.0)
        matrix = matrix + headroom * share[:, None]
    np.clip(matrix, lower, upper, out=matrix)
    return matrix[0] if single else matrix


# --------------------------------------------------------------------------- #
# Shared SLSQP machinery
# --------------------------------------------------------------------------- #
def _budget_constraint(n: int) -> dict:
    ones = np.ones(int(n), dtype=float)
    return {'type': 'eq', 'fun': lambda w: float(np.sum(w) - 1.0), 'jac': lambda w: ones}


def _run_slsqp(fun, jac, x0, bounds, constraint_set):
    return minimize(fun, np.asarray(x0, dtype=float), jac=jac, method='SLSQP',
                    bounds=bounds, constraints=constraint_set,
                    options=dict(SLSQP_OPTIONS))


def _starting_points(n: int, lower: float, upper: float, cov: np.ndarray,
                     closed_form: np.ndarray, seed) -> list:
    """The eight starts of RF-O4, in order, each projected to the feasible box."""
    rng = child_rng(0 if seed is None else int(seed), 'optimizer')
    raw = [('equal', equal_weights(n)),
           ('min_variance', closed_form_min_variance(cov)),
           ('closed_form', np.asarray(closed_form, dtype=float))]
    for index in range(1, 6):
        raw.append((f'random_{index}', rng.dirichlet(np.ones(n, dtype=float))))
    return [(label, project_to_bounded_simplex(vector, lower, upper))
            for label, vector in raw]


def _multistart_solve(fun, jac, constraints: Constraints, n: int, *,
                      closed_form: np.ndarray, cov: np.ndarray, seed,
                      extra_constraints: tuple = (), accept=None) -> OptimizationResult:
    """Try the starts in order and accept the FIRST feasible converged result.

    If none converges, the best feasible objective value is returned with
    ``success=False`` and the solver message, so ``certified=False`` propagates
    to the UI. If not even a starting point is feasible, the constraint set has
    no admissible vector and ``InfeasibleConstraintsError`` is raised.
    """
    constraints.validate(n)
    bounds = constraints.bounds(n)
    lower, upper = bounds[0]
    constraint_set = [_budget_constraint(n), *extra_constraints]
    admissible = accept if accept is not None else (lambda w: True)

    starts = [(label, vector) for label, vector
              in _starting_points(n, lower, upper, cov, closed_form, seed)
              if check_feasibility(vector, constraints).ok]
    if not starts:
        raise InfeasibleConstraintsError('no_feasible_starting_point', n=int(n),
                                         max_weight=float(constraints.max_weight),
                                         allow_short=bool(constraints.allow_short))

    best = None
    runs = 0
    for label, x0 in starts:
        runs += 1
        result = _run_slsqp(fun, jac, x0, bounds, constraint_set)
        candidate = np.asarray(result.x, dtype=float)
        feasibility = check_feasibility(candidate, constraints)
        if not (feasibility.ok and admissible(candidate)):
            continue
        if result.success:
            return OptimizationResult(weights=candidate, success=True,
                                      message=str(result.message),
                                      objective_value=float(result.fun),
                                      n_restarts=runs, feasibility=feasibility,
                                      start_used=label)
        value = float(result.fun)
        if best is None or value < best[0]:
            best = (value, candidate, str(result.message), label)

    if best is not None:
        return OptimizationResult(weights=best[1], success=False, message=best[2],
                                  objective_value=best[0], n_restarts=runs,
                                  feasibility=check_feasibility(best[1], constraints),
                                  start_used=best[3])

    values = [float(fun(vector)) for _, vector in starts]
    index = int(np.argmin(values))
    label, vector = starts[index]
    return OptimizationResult(weights=vector, success=False,
                              message='solver_did_not_return_a_feasible_point',
                              objective_value=values[index], n_restarts=runs,
                              feasibility=check_feasibility(vector, constraints),
                              start_used=label)


# --------------------------------------------------------------------------- #
# Solvers
# --------------------------------------------------------------------------- #
def solve_min_variance(cov: np.ndarray, constraints: Constraints, *,
                       seed=None) -> OptimizationResult:
    """Minimise ``w' Sigma w`` subject to ``sum(w) = 1`` and the box bounds."""
    matrix = np.atleast_2d(np.asarray(cov, dtype=float))
    n = matrix.shape[0]

    def fun(w: np.ndarray) -> float:
        return float(w @ matrix @ w)

    def jac(w: np.ndarray) -> np.ndarray:
        return 2.0 * (matrix @ w)

    return _multistart_solve(fun, jac, constraints, n,
                             closed_form=closed_form_min_variance(matrix),
                             cov=matrix, seed=seed)


def solve_max_sharpe(mu: np.ndarray, cov: np.ndarray, rf_cc_annual: float,
                     constraints: Constraints, *, seed=None) -> OptimizationResult:
    """Maximise ``(w'mu - rf)/sqrt(w' Sigma w)`` over the same feasible set.

    Degenerate guard, evaluated BEFORE optimising: when no feasible portfolio
    beats the risk-free rate the Sharpe ratio improves by adding volatility and
    the optimum runs away to a vertex, so the minimum-variance weights are
    returned with ``message='tangency_undefined_rf_above_all_assets'``.
    ``summarize_portfolio`` turns that message into ``certified=False``.
    """
    mu_vector = np.asarray(mu, dtype=float).ravel()
    matrix = np.atleast_2d(np.asarray(cov, dtype=float))
    n = mu_vector.size
    rf = float(rf_cc_annual)

    if max_feasible_return(mu_vector, constraints) <= rf:
        fallback = solve_min_variance(matrix, constraints, seed=seed).weights
        return OptimizationResult(weights=fallback, success=True,
                                  message=TANGENCY_UNDEFINED,
                                  objective_value=float('nan'), n_restarts=0,
                                  feasibility=check_feasibility(fallback, constraints),
                                  start_used='min_variance_fallback')

    def fun(w: np.ndarray) -> float:
        s = math.sqrt(max(float(w @ matrix @ w), VARIANCE_FLOOR))
        return float(-(float(w @ mu_vector) - rf) / s)

    def jac(w: np.ndarray) -> np.ndarray:
        s = math.sqrt(max(float(w @ matrix @ w), VARIANCE_FLOOR))
        excess = float(w @ mu_vector) - rf
        return -(mu_vector / s - excess * (matrix @ w) / s ** 3)

    return _multistart_solve(fun, jac, constraints, n,
                             closed_form=closed_form_tangency(mu_vector, matrix, rf),
                             cov=matrix, seed=seed)


def solve_min_variance_target(mu: np.ndarray, cov: np.ndarray, target_return: float,
                              constraints: Constraints, *,
                              x0: np.ndarray = None, seed=None) -> OptimizationResult:
    """Minimise ``w' Sigma w`` subject to ``w'mu = target_return`` and the box bounds.

    ``x0`` is the explicit start the frontier continuation passes (the weights of
    the previous point); with ``x0=None`` the standard multistart applies. A
    result is accepted only when the return equality is met as well, so an
    infeasible target comes back with ``success=False`` and ``build_frontier``
    can report it as skipped instead of drawing a point that is not on the
    frontier.
    """
    mu_vector = np.asarray(mu, dtype=float).ravel()
    matrix = np.atleast_2d(np.asarray(cov, dtype=float))
    n = mu_vector.size
    target = float(target_return)
    tolerance = TARGET_TOL * max(1.0, abs(target))

    def fun(w: np.ndarray) -> float:
        return float(w @ matrix @ w)

    def jac(w: np.ndarray) -> np.ndarray:
        return 2.0 * (matrix @ w)

    return_constraint = {'type': 'eq',
                         'fun': lambda w: float(w @ mu_vector - target),
                         'jac': lambda w: mu_vector}

    def accept(w: np.ndarray) -> bool:
        return bool(abs(float(w @ mu_vector) - target) <= tolerance)

    if x0 is None:
        return _multistart_solve(fun, jac, constraints, n,
                                 closed_form=closed_form_min_variance(matrix),
                                 cov=matrix, seed=seed,
                                 extra_constraints=(return_constraint,),
                                 accept=accept)

    constraints.validate(n)
    bounds = constraints.bounds(n)
    lower, upper = bounds[0]
    start = project_to_bounded_simplex(np.asarray(x0, dtype=float).ravel(), lower, upper)
    result = _run_slsqp(fun, jac, start, bounds,
                        [_budget_constraint(n), return_constraint])
    candidate = np.asarray(result.x, dtype=float)
    feasibility = check_feasibility(candidate, constraints)
    success = bool(result.success and feasibility.ok and accept(candidate))
    return OptimizationResult(weights=candidate, success=success,
                              message=str(result.message),
                              objective_value=float(result.fun), n_restarts=1,
                              feasibility=feasibility, start_used='continuation')
