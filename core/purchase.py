"""Purchase balance: optimal weights translated into concrete orders (design.md section 8).

Budget model, valid with and without current positions (section 8.2). ``capital``
is always NEW money; whatever the user already holds enters separately:

    current_value    = sum(current_units_i * price_i)
    total_investable = capital + current_value - cash_reserve
    n_active         = #{i : w_i > 1e-12}
    notional         = compute_notional_budget(budget, pct_fee, fixed_fee, n_active)
    target_value_i   = w_i * notional
    sell_proceeds    = max(sell_gross - sell_costs, 0)        -- NET of its own costs
    invested         = sum(target_units_i * price_i)          -- VALUE of the portfolio
    leftover_cash    = capital + sell_proceeds - buy_gross - buy_costs

Cash invariant enforced inside the allocation loop:

    buy_gross + buy_costs <= capital + sell_proceeds - cash_reserve + 1e-9

With ``current_positions=None`` there are no sales and it collapses into the
literal form of CA-15, ``invested + total_costs <= capital - cash_reserve``.

This module never imports Streamlit and emits no user-facing text: ``action``
holds 'buy' / 'sell' / 'hold' and ui/labels.py renders them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.types import InfeasibleConstraintsError, PortfolioError

__all__ = [
    'PurchaseConfig',
    'PurchaseBalance',
    'ACTIONS',
    'ROW_COLUMNS',
    'ACTIVE_WEIGHT_TOL',
    'CASH_INVARIANT_TOL',
    'compute_notional_budget',
    'allocate_integer_units',
    'build_purchase_balance',
]

ACTIONS = ('buy', 'sell', 'hold')
ROW_COLUMNS = ('asset', 'target_weight', 'target_value', 'price', 'current_units',
               'current_weight', 'target_units', 'delta_units', 'action',
               'gross_amount', 'costs', 'final_value', 'final_weight',
               'deviation', 'note_code')
ACTIVE_WEIGHT_TOL = 1e-12
CASH_INVARIANT_TOL = 1e-9
VALUE_FLOOR = 1e-12


@dataclass(frozen=True)
class PurchaseConfig:
    """User-facing settings of the purchase balance.

    ``weight_tolerance`` is an ABSOLUTE weight fraction: an asset is classified
    'hold' when ``|current_weight - target_weight| <= weight_tolerance``. There
    is no ``min_trade_amount``: no requirement asks for a minimum order size, and
    a declared-but-unused field would leave the reallocation rule undefined.
    """

    capital: float
    cash_reserve: float = 0.0
    pct_fee: float = 0.0
    fixed_fee: float = 0.0
    integer_units: bool = True
    weight_tolerance: float = 0.01

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.weight_tolerance) <= 0.5:
            raise ValueError('weight_tolerance_out_of_range')
        if float(self.pct_fee) >= 1.0:
            raise InfeasibleConstraintsError('pct_fee_not_below_one',
                                             pct_fee=float(self.pct_fee))
        if float(self.pct_fee) < 0.0 or float(self.fixed_fee) < 0.0:
            raise ValueError('fee_is_negative')


@dataclass(frozen=True)
class PurchaseBalance:
    """Per-asset orders plus the twelve aggregates of design.md section 8.1."""

    rows: pd.DataFrame
    capital: float
    current_value: float
    total_investable: float
    notional: float
    invested: float
    buy_gross: float
    sell_proceeds: float
    total_costs: float
    leftover_cash: float
    max_deviation: float
    feasible: bool
    note_codes: tuple = ()


# --------------------------------------------------------------------------- #
# Notional budget: the single implementation
# --------------------------------------------------------------------------- #
def compute_notional_budget(investable: float, pct_fee: float, fixed_fee: float,
                            n_active: int) -> float:
    """``max((investable - fixed_fee*n_active) / (1 + pct_fee), 0.0)``.

    Single implementation of the notional budget; ``core.cashflow`` imports this
    function instead of repeating the arithmetic (FD-5), so section 5 and
    section 7 of the app cannot diverge.
    """
    reserved = float(investable) - float(fixed_fee) * int(n_active)
    return float(max(reserved / (1.0 + float(pct_fee)), 0.0))


# --------------------------------------------------------------------------- #
# Integer allocation
# --------------------------------------------------------------------------- #
def allocate_integer_units(target_values: np.ndarray, prices: np.ndarray, budget: float,
                           pct_fee: float, fixed_fee: float,
                           already_traded: np.ndarray) -> tuple:
    """Return ``(units, outlay, leftover)`` for the integer-unit allocation.

    Steps 3 and 4 of design.md section 8.2:

    * floor: ``units_i = floor(target_value_i / price_i)``;
    * greedy reallocation by descending relative deficit
      ``(target_value_i - units_i*price_i) / max(target_value_i, eps)``, ties
      broken by asset index, buying one unit at a time while
      ``outlay + marginal <= budget`` and blocking the asset otherwise, with

          marginal = price_i*(1 + pct_fee) + (fixed_fee if the fee is not yet charged)

      which is exactly the increment of ``outlay`` for that unit, since ``outlay``
      recounts ``fixed_fee`` over the assets that carry a fee.

    ``already_traded`` marks assets whose fixed fee is accounted for elsewhere in
    the same balance; with the all-False mask (the purchase-from-scratch case)
    the marginal cost is the literal formula of section 8.2. The iteration bound
    is ``budget/min(price) + n``, finite by construction.
    """
    targets = np.asarray(target_values, dtype=float).ravel()
    price = np.asarray(prices, dtype=float).ravel()
    traded = np.asarray(already_traded, dtype=bool).ravel()
    n = targets.size
    cap = float(budget)
    pct = float(pct_fee)
    fixed = float(fixed_fee)

    units = np.zeros(n, dtype=float)
    positive = (targets > 0.0) & (price > 0.0)
    units[positive] = np.floor(targets[positive] / price[positive])

    def total_outlay(current: np.ndarray) -> float:
        charged = int(np.sum((current > 0.0) | traded))
        return float(np.sum(current * price) * (1.0 + pct) + fixed * charged)

    outlay = total_outlay(units)
    blocked = ~positive
    if cap <= 0.0:
        return np.zeros(n, dtype=float), 0.0, cap

    cheapest = float(price[positive].min()) if positive.any() else 1.0
    max_iterations = int(cap / max(cheapest, VALUE_FLOOR)) + n + 1
    for _ in range(max_iterations):
        if bool(blocked.all()):
            break
        deficit = (targets - units * price) / np.maximum(targets, VALUE_FLOOR)
        deficit = np.where(blocked, -np.inf, deficit)
        index = int(np.argsort(-deficit, kind='stable')[0])
        if not np.isfinite(deficit[index]):
            break
        marginal = price[index] * (1.0 + pct)
        if units[index] <= 0.0 and not traded[index]:
            marginal += fixed
        if outlay + marginal <= cap:
            units[index] += 1.0
            outlay += marginal
        else:
            blocked[index] = True

    return units, float(outlay), float(cap - outlay)


# --------------------------------------------------------------------------- #
# Balance
# --------------------------------------------------------------------------- #
def _empty_rows(names: tuple, weights: np.ndarray, prices: np.ndarray,
                current_units: np.ndarray, note: str) -> pd.DataFrame:
    """Rows of a balance with no operations at all (non-positive budget)."""
    frame = pd.DataFrame({
        'asset': list(names),
        'target_weight': weights,
        'target_value': np.zeros_like(weights),
        'price': prices,
        'current_units': current_units,
        'current_weight': np.zeros_like(weights),
        'target_units': current_units,
        'delta_units': np.zeros_like(weights),
        'action': [ACTIONS[2]] * len(names),
        'gross_amount': np.zeros_like(weights),
        'costs': np.zeros_like(weights),
        'final_value': current_units * prices,
        'final_weight': np.zeros_like(weights),
        'deviation': -weights,
        'note_code': [note] * len(names),
    })
    return frame[list(ROW_COLUMNS)]


def build_purchase_balance(weights: np.ndarray, prices: np.ndarray, names: tuple,
                           config: PurchaseConfig,
                           current_positions: np.ndarray = None) -> PurchaseBalance:
    """Translate target weights into orders against ``prices`` (``aligned.prices[-1]``).

    Declared deviation from the literal budget of section 8.2, needed to keep the
    cash invariant true in a rebalance: when there are current positions, the
    worst-case sale costs (``pct_fee*current_value + fixed_fee*n_current``) are
    reserved before computing the notional budget. Without that reservation a
    full-turnover rebalance (sell everything, buy something else) pays the
    percentage fee on both legs while the notional budget only reserved one, and
    the invariant below would fail on a configuration the app supports. With
    ``current_positions=None`` the reservation is zero and the budget is literally
    the one of section 8.2.
    """
    w = np.asarray(weights, dtype=float).ravel()
    price = np.asarray(prices, dtype=float).ravel()
    asset_names = tuple(names)
    n = w.size
    if price.size != n or len(asset_names) != n:
        raise PortfolioError('purchase_inputs_length_mismatch', n_weights=int(n),
                             n_prices=int(price.size), n_names=int(len(asset_names)))
    if np.any(price <= 0.0):
        raise PortfolioError('purchase_price_not_positive')

    current_units = (np.zeros(n, dtype=float) if current_positions is None
                     else np.asarray(current_positions, dtype=float).ravel())
    if current_units.size != n:
        raise PortfolioError('purchase_inputs_length_mismatch', n_weights=int(n),
                             n_positions=int(current_units.size))

    pct = float(config.pct_fee)
    fixed = float(config.fixed_fee)
    capital = float(config.capital)
    reserve = float(config.cash_reserve)

    current_value = float(np.sum(current_units * price))
    total_investable = capital + current_value - reserve
    n_active = int(np.sum(w > ACTIVE_WEIGHT_TOL))
    n_current = int(np.sum(current_units > 0.0))
    sale_cost_reserve = pct * current_value + fixed * n_current
    budget = total_investable - sale_cost_reserve

    notional = (compute_notional_budget(budget, pct, fixed, n_active)
                if budget > 0.0 and n_active > 0 else 0.0)
    if notional <= 0.0:
        note = 'budget_non_positive' if budget <= 0.0 else 'notional_non_positive'
        rows = _empty_rows(asset_names, w, price, current_units, note)
        return PurchaseBalance(rows=rows, capital=capital, current_value=current_value,
                               total_investable=float(total_investable), notional=0.0,
                               invested=current_value, buy_gross=0.0, sell_proceeds=0.0,
                               total_costs=0.0, leftover_cash=max(capital, 0.0),
                               max_deviation=float(np.max(np.abs(w))) if n else 0.0,
                               feasible=False, note_codes=(note,))

    target_value = w * notional

    if config.integer_units:
        target_units, _, _ = allocate_integer_units(
            target_value, price, budget, pct, fixed, current_units > 0.0)
    else:
        target_units = np.where(w != 0.0, target_value / price, 0.0)

    delta_units = target_units - current_units
    gross_amount = delta_units * price
    traded = np.abs(gross_amount) > VALUE_FLOOR
    row_costs = pct * np.abs(gross_amount) + fixed * traded

    buy_mask = gross_amount > VALUE_FLOOR
    sell_mask = gross_amount < -VALUE_FLOOR
    buy_gross = float(np.sum(gross_amount[buy_mask]))
    sell_gross = float(-np.sum(gross_amount[sell_mask]))
    buy_costs = float(np.sum(row_costs[buy_mask]))
    sell_costs = float(np.sum(row_costs[sell_mask]))
    sell_proceeds = float(max(sell_gross - sell_costs, 0.0))
    total_costs = float(buy_costs + sell_costs)
    invested = float(np.sum(target_units * price))
    leftover_cash = float(capital + sell_proceeds - buy_gross - buy_costs)

    available = capital + sell_proceeds - reserve + CASH_INVARIANT_TOL
    if buy_gross + buy_costs > available:
        raise PortfolioError('purchase_cash_invariant_violated',
                             buy_gross=buy_gross, buy_costs=buy_costs,
                             capital=capital, sell_proceeds=sell_proceeds,
                             cash_reserve=reserve)

    current_weight = (current_units * price / current_value if current_value > 0.0
                      else np.zeros(n, dtype=float))
    final_weight = (target_units * price / invested if invested > 0.0
                    else np.zeros(n, dtype=float))
    deviation = final_weight - w

    tolerance = float(config.weight_tolerance)
    action = []
    for index in range(n):
        gap = current_weight[index] - w[index]
        if abs(gap) <= tolerance:
            action.append(ACTIONS[2])
        else:
            action.append(ACTIONS[0] if gap < 0.0 else ACTIONS[1])

    note_code = [''] * n
    notes = []
    unreachable = (w > ACTIVE_WEIGHT_TOL) & (target_units <= 0.0) & (price > budget)
    for index in np.flatnonzero(unreachable):
        note_code[index] = 'price_above_assignable_budget'
        notes.append('price_above_assignable_budget')

    rows = pd.DataFrame({
        'asset': list(asset_names),
        'target_weight': w,
        'target_value': target_value,
        'price': price,
        'current_units': current_units,
        'current_weight': current_weight,
        'target_units': target_units,
        'delta_units': delta_units,
        'action': action,
        'gross_amount': gross_amount,
        'costs': row_costs,
        'final_value': target_units * price,
        'final_weight': final_weight,
        'deviation': deviation,
        'note_code': note_code,
    })[list(ROW_COLUMNS)]

    if leftover_cash < -CASH_INVARIANT_TOL:
        raise PortfolioError('purchase_leftover_cash_negative',
                             leftover_cash=leftover_cash)

    return PurchaseBalance(rows=rows, capital=capital, current_value=current_value,
                           total_investable=float(total_investable),
                           notional=float(notional), invested=invested,
                           buy_gross=buy_gross, sell_proceeds=sell_proceeds,
                           total_costs=total_costs,
                           leftover_cash=float(max(leftover_cash, 0.0)),
                           max_deviation=float(np.max(np.abs(deviation))) if n else 0.0,
                           feasible=True, note_codes=tuple(dict.fromkeys(notes)))
