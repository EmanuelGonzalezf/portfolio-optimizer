"""Installment cash-flow engine (design.md sections 10.1, 10.2, 10.2b and 10.3).

INDEX ORIGIN CONVENTION (section 10.1), binding for this module, for the path
simulator of section 11.3 and for ``generate_rebalance_indices(..., window=0)``::

    calendar[0] = valuation_date              len(calendar) = horizon + 1
    log_returns[t-1] carries calendar[t-1] -> calendar[t],   t = 1..horizon
    (row j of the (horizon, n) array moves calendar[j] -> calendar[j+1]; no row
     ever arrives at calendar[0])

    payment_indices[k] = position in `calendar` of the first business day
                         >= the k-th installment date, clipped to 1 <= i <= horizon
    rebalance_indices  = positions in `calendar`, same range

    cumulative factor up to calendar[t_k]:
        np.exp(log_returns[:t_k].sum(axis=0))      # t_k rows, NOT t_k + 1

So a calendar index ``t`` consumes the FIRST ``t`` rows of the return array.
``payment_indices[k] == 0`` is impossible by construction: an installment falling
on the valuation date itself is pushed to the next business day, which is index 1.

``valuation_date`` is ALWAYS the last aligned price date, never ``date.today()``.

ACCOUNTING IDENTITY (section 10.3), two steps, each cost subtracted exactly once
and returns never added twice::

    wealth_open[k]   = wealth_after[k-1]                    (k = 0: wealth_initial)
    wealth_before[k] = wealth_open[k] + returns_in_period[k] - rebalance_costs_pre[k]
    wealth_after[k]  = wealth_before[k] + external_income[k] - paid[k]
                       - sale_costs[k] - rebalance_costs_post[k]

    wealth_final = wealth_initial + total_returns + total_external - total_paid
                   - (total_sale_costs + total_rebalance_costs)

``rebalance_costs_pre[k]`` holds the recomposition costs paid STRICTLY inside
``(t_{k-1}, t_k)``, which are already reflected in the market value measured at
``t_k``; ``rebalance_costs_post[k]`` holds the one processed at ``t_k`` after the
payment. ``rebalance_costs = pre + post`` is a DERIVED column that takes part in
no identity. Costs after the last installment go to the scalar
``tail_rebalance_costs`` and only reach the aggregate identity.

``returns_in_period[k]`` is summed over SUBSEGMENTS OF CONSTANT UNITS delimited
by the events of the period (previous payment, every intra-period rebalance, end
of period), plus the cash interest of the period::

    returns_in_period[k] = sum_s sum_i units_i^(s) * (price_end(s),i - price_start(s),i)
                           + cash_interest_in_period[k]

This module never imports Streamlit and emits no user-facing text: the ledger
carries English codes in its ``note_code`` column and ui/labels.py renders them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from core.purchase import allocate_integer_units, compute_notional_budget
from core.types import (InfeasibleConstraintsError, PortfolioError, ScheduleError,
                        freeze_array)

__all__ = [
    'InstallmentSchedule',
    'LiquidityPolicy',
    'InitialPosition',
    'CashflowPathResult',
    'FREQUENCY_OFFSETS',
    'SCHEDULE_FRAME_COLUMNS',
    'LEDGER_COLUMNS',
    'COVERED_TOL',
    'MARKET_VALUE_FLOOR',
    'IDENTITY_TOL',
    'SALE_NOT_EXECUTED',
    'REBALANCE_SKIPPED_NO_VALUE',
    'REBALANCE_SKIPPED_COST',
    'TWO_INSTALLMENTS_SAME_DAY',
    'build_schedule',
    'schedule_from_frame',
    'build_calendar',
    'map_payment_indices',
    'aggregate_duplicate_payments',
    'build_initial_position',
    'solve_gross_sale',
    'simulate_cashflow_paths',
    'simulate_cashflow_single',
    'check_wealth_identity',
    'required_additional_capital',
]

FREQUENCY_OFFSETS = {
    'monthly': pd.DateOffset(months=1),
    'quarterly': pd.DateOffset(months=3),
    'semiannual': pd.DateOffset(months=6),
    'annual': pd.DateOffset(years=1),
}

SCHEDULE_FRAME_COLUMNS = ('date', 'amount', 'external_income')

LEDGER_COLUMNS = ('due', 'wealth_open', 'returns_in_period', 'wealth_before',
                  'cash_before', 'mv_before', 'external_income', 'gross_sold',
                  'sale_costs', 'rebalance_costs_pre', 'rebalance_costs_post',
                  'rebalance_costs', 'paid', 'deficit', 'cash_after', 'mv_after',
                  'wealth_after', 'covered', 'note_code')

COVERED_TOL = 1e-6
MARKET_VALUE_FLOOR = 1e-9
IDENTITY_TOL = 1e-6
ACTIVE_WEIGHT_TOL = 1e-12

SALE_NOT_EXECUTED = 'sale_not_executed_costs_exceed_proceeds'
REBALANCE_SKIPPED_NO_VALUE = 'rebalance_skipped_no_market_value'
REBALANCE_SKIPPED_COST = 'rebalance_skipped_cost_exceeds_value'
TWO_INSTALLMENTS_SAME_DAY = 'two_installments_same_business_day'


def _lock(values, dtype=float) -> np.ndarray:
    """Copy ``values`` into a non-writeable array, preserving the asked dtype."""
    arr = np.array(values, dtype=dtype, copy=True)
    arr.flags.writeable = False
    return arr


# --------------------------------------------------------------------------- #
# Calendar (section 10.1)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InstallmentSchedule:
    """Installment calendar: strictly increasing dates, non-negative amounts.

    ``external_income[k]`` is collected on the same date, BEFORE the payment
    (step 4 of section 10.2).
    """

    dates: pd.DatetimeIndex
    amounts: np.ndarray
    external_income: np.ndarray

    def __post_init__(self) -> None:
        dates = pd.DatetimeIndex(self.dates)
        amounts = freeze_array(self.amounts).ravel()
        external = freeze_array(self.external_income).ravel()
        if len(dates) == 0:
            raise ScheduleError('schedule_is_empty')
        if amounts.size != len(dates) or external.size != len(dates):
            raise ScheduleError('schedule_length_mismatch', n_dates=int(len(dates)),
                                n_amounts=int(amounts.size),
                                n_external=int(external.size))
        for row in range(1, len(dates)):
            if dates[row] <= dates[row - 1]:
                raise ScheduleError('schedule_dates_not_increasing', row=int(row))
        invalid = np.flatnonzero(~np.isfinite(amounts) | (amounts < 0.0))
        if invalid.size:
            raise ScheduleError('schedule_amount_invalid', row=int(invalid[0]))
        invalid = np.flatnonzero(~np.isfinite(external) | (external < 0.0))
        if invalid.size:
            raise ScheduleError('schedule_external_income_invalid', row=int(invalid[0]))
        object.__setattr__(self, 'dates', dates)
        object.__setattr__(self, 'amounts', amounts)
        object.__setattr__(self, 'external_income', external)

    @property
    def n_installments(self) -> int:
        return int(len(self.dates))

    @property
    def total_scheduled(self) -> float:
        return float(np.sum(self.amounts))

    def to_tuple(self) -> tuple:
        return (self.dates.asi8.copy(),
                np.asarray(self.amounts, dtype=float).copy(),
                np.asarray(self.external_income, dtype=float).copy())

    @classmethod
    def from_tuple(cls, data: tuple) -> 'InstallmentSchedule':
        dates_i8, amounts, external = data
        dates = pd.DatetimeIndex(
            np.asarray(dates_i8, dtype='int64').view('datetime64[ns]'))
        return cls(dates=dates, amounts=amounts, external_income=external)


def build_schedule(first_date, frequency: str, n_installments: int, amount: float,
                   growth_rate: float, external_income=0.0) -> InstallmentSchedule:
    """Regular calendar with ``amounts[k] = amount * (1 + growth_rate)**k``.

    The growth is PER INSTALLMENT, not annualised (the UI label says so).
    ``frequency`` is a key of ``FREQUENCY_OFFSETS``.
    """
    key = str(frequency)
    if key not in FREQUENCY_OFFSETS:
        raise ScheduleError('unknown_frequency', frequency=key)
    count = int(n_installments)
    if count < 1:
        raise ScheduleError('n_installments_not_positive', n_installments=count)
    base = float(amount)
    if not np.isfinite(base) or base < 0.0:
        raise ScheduleError('schedule_amount_invalid', row=0)
    growth = float(growth_rate)
    if not np.isfinite(growth) or growth <= -1.0:
        raise ScheduleError('schedule_growth_rate_invalid', growth_rate=growth)

    start = pd.Timestamp(first_date)
    offset = FREQUENCY_OFFSETS[key]
    dates = pd.DatetimeIndex([start if k == 0 else start + k * offset
                              for k in range(count)])
    amounts = base * (1.0 + growth) ** np.arange(count, dtype=float)

    income = np.asarray(external_income, dtype=float).ravel()
    if income.size == 1:
        income = np.full(count, float(income[0]))
    elif income.size != count:
        raise ScheduleError('schedule_length_mismatch', n_dates=count,
                            n_external=int(income.size))
    return InstallmentSchedule(dates=dates, amounts=amounts, external_income=income)


def schedule_from_frame(df: pd.DataFrame) -> InstallmentSchedule:
    """Custom schedule from an English-named frame.

    Input contract, enforced: ``'date'`` (parseable, unique, strictly
    increasing), ``'amount'`` (float >= 0), ``'external_income'`` (float >= 0,
    optional, defaults to 0.0 when absent). Any other column is ignored. Raises
    ``ScheduleError`` carrying the offending row index.
    """
    if not isinstance(df, pd.DataFrame):
        raise ScheduleError('schedule_frame_expected')
    for column in ('date', 'amount'):
        if column not in df.columns:
            raise ScheduleError('schedule_column_missing', column=column)
    frame = df.reset_index(drop=True)
    if len(frame) == 0:
        raise ScheduleError('schedule_is_empty')

    dates = pd.to_datetime(frame['date'], errors='coerce')
    invalid = np.flatnonzero(dates.isna().to_numpy())
    if invalid.size:
        raise ScheduleError('schedule_date_invalid', row=int(invalid[0]))

    amounts = pd.to_numeric(frame['amount'], errors='coerce').to_numpy(dtype=float)
    invalid = np.flatnonzero(~np.isfinite(amounts))
    if invalid.size:
        raise ScheduleError('schedule_amount_invalid', row=int(invalid[0]))

    if 'external_income' in frame.columns:
        income = pd.to_numeric(frame['external_income'],
                               errors='coerce').to_numpy(dtype=float)
        invalid = np.flatnonzero(~np.isfinite(income))
        if invalid.size:
            raise ScheduleError('schedule_external_income_invalid', row=int(invalid[0]))
    else:
        income = np.zeros(len(frame), dtype=float)

    return InstallmentSchedule(dates=pd.DatetimeIndex(dates), amounts=amounts,
                               external_income=income)


def _effective_dates(schedule: InstallmentSchedule) -> pd.DatetimeIndex:
    """First business day >= each installment date (local holidays not modelled)."""
    offset = pd.offsets.BDay()
    return pd.DatetimeIndex([offset.rollforward(pd.Timestamp(date).normalize())
                             for date in schedule.dates])


def build_calendar(valuation_date, schedule: InstallmentSchedule,
                   holding_period_days: int = 0) -> pd.DatetimeIndex:
    """Business-day calendar of the simulated horizon (section 10.1).

    ``calendar[0] == valuation_date`` always and ``len(calendar) == horizon + 1``.
    ``pd.bdate_range`` would drop a non-business valuation date, so that single
    case is prepended instead of silently shifting every payment by a day.
    Exposed publicly because ui/compute.py has to send ``calendar.asi8`` across
    the cache boundary (section 13.2) and must not rebuild it with its own rule.
    """
    valuation = pd.Timestamp(valuation_date).normalize()
    holding = int(holding_period_days)
    if holding < 0:
        raise ScheduleError('holding_period_negative', holding_period_days=holding)
    effective = _effective_dates(schedule)
    holding_end = valuation + pd.offsets.BDay(holding)
    horizon_end = max(effective[-1], holding_end)
    calendar = pd.bdate_range(valuation, horizon_end)
    if len(calendar) == 0 or calendar[0] != valuation:
        calendar = pd.DatetimeIndex([valuation]).append(calendar)
    return calendar


def map_payment_indices(valuation_date, schedule: InstallmentSchedule,
                        holding_period_days: int = 0) -> tuple:
    """Return exactly, in this order:

    ``(payment_indices (K,) int, horizon int, effective_payment_dates (K,) DatetimeIndex)``

    with the index origin convention of section 10.1 implemented literally:
    ``payment_indices[k]`` is the position in ``build_calendar(...)`` of the
    first business day >= the k-th installment date, clipped to
    ``1 <= i <= horizon``.
    """
    valuation = pd.Timestamp(valuation_date).normalize()
    calendar = build_calendar(valuation, schedule, holding_period_days)
    horizon = int(len(calendar) - 1)
    if horizon < 1:
        raise ScheduleError('schedule_horizon_not_positive', horizon=horizon)
    effective = _effective_dates(schedule)
    positions = np.asarray(calendar.searchsorted(effective, side='left'), dtype=int)
    payment_indices = np.clip(positions, 1, horizon).astype(int)
    return payment_indices, horizon, effective


def aggregate_duplicate_payments(schedule: InstallmentSchedule,
                                 payment_indices: np.ndarray,
                                 effective_payment_dates: pd.DatetimeIndex = None
                                 ) -> tuple:
    """Collapse installments that land on the same calendar index.

    Section 10.1: "if two installments fall on the same index, they are
    aggregated by adding the amounts and a note is recorded". Returns
    ``(schedule, payment_indices, effective_payment_dates, note_codes)``; with no
    collision the inputs are returned untouched and ``note_codes`` is empty. The
    engine calls this first, so the ledger keeps one row per calendar index and
    ``effective_payment_dates`` stays unique as the ledger index requires.
    """
    indices = np.asarray(payment_indices, dtype=int).ravel()
    unique, inverse, counts = np.unique(indices, return_inverse=True, return_counts=True)
    if unique.size == indices.size:
        return schedule, indices, effective_payment_dates, ()

    amounts = np.zeros(unique.size, dtype=float)
    income = np.zeros(unique.size, dtype=float)
    np.add.at(amounts, inverse, np.asarray(schedule.amounts, dtype=float))
    np.add.at(income, inverse, np.asarray(schedule.external_income, dtype=float))
    first_row = np.zeros(unique.size, dtype=int)
    for row in range(indices.size - 1, -1, -1):
        first_row[inverse[row]] = row
    collapsed = InstallmentSchedule(dates=schedule.dates[first_row], amounts=amounts,
                                    external_income=income)
    dates = (None if effective_payment_dates is None
             else pd.DatetimeIndex(effective_payment_dates)[first_row])
    return collapsed, unique.astype(int), dates, (TWO_INSTALLMENTS_SAME_DAY,)


# --------------------------------------------------------------------------- #
# Liquidity policy and initial position (sections 10.2 and 10.2b)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LiquidityPolicy:
    """Transaction costs and liquidity rules of the installment horizon.

    ``cash_reserve_initial`` is INITIAL cash, not a maintained target: it is set
    aside once when the position is built and, if spent on installments, it is
    NOT replenished with sales. ``sale_rule`` has a single implemented value,
    ``'pro_rata'`` (sale proportional to each position's market value).
    """

    pct_fee: float = 0.001
    fixed_fee: float = 0.0
    allow_sales: bool = True
    cash_reserve_initial: float = 0.0
    cash_interest: bool = False
    sale_rule: str = 'pro_rata'

    def __post_init__(self) -> None:
        if float(self.pct_fee) >= 1.0:
            raise InfeasibleConstraintsError('pct_fee_not_below_one',
                                             pct_fee=float(self.pct_fee))
        if float(self.pct_fee) < 0.0 or float(self.fixed_fee) < 0.0:
            raise ValueError('fee_is_negative')
        if float(self.cash_reserve_initial) < 0.0:
            raise ValueError('cash_reserve_negative')
        if str(self.sale_rule) != 'pro_rata':
            raise ValueError('unknown_sale_rule')

    def to_tuple(self) -> tuple:
        return (float(self.pct_fee), float(self.fixed_fee), bool(self.allow_sales),
                float(self.cash_reserve_initial), bool(self.cash_interest),
                str(self.sale_rule))

    @classmethod
    def from_tuple(cls, data: tuple) -> 'LiquidityPolicy':
        pct_fee, fixed_fee, allow_sales, reserve, cash_interest, sale_rule = data
        return cls(pct_fee=float(pct_fee), fixed_fee=float(fixed_fee),
                   allow_sales=bool(allow_sales),
                   cash_reserve_initial=float(reserve),
                   cash_interest=bool(cash_interest), sale_rule=str(sale_rule))


@dataclass(frozen=True)
class InitialPosition:
    """Resolved starting position: ``capital -> (units, cash)`` (section 10.2b)."""

    units: np.ndarray
    cash: float
    purchase_costs: float
    invested: float
    prices: np.ndarray
    integer_units: bool
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, 'units', freeze_array(self.units).ravel())
        object.__setattr__(self, 'prices', freeze_array(self.prices).ravel())

    @property
    def wealth_initial(self) -> float:
        """``invested + cash``, which equals ``capital - purchase_costs``."""
        return float(self.invested + self.cash)

    @property
    def capital(self) -> float:
        """The capital that produced this position, ``invested + cash + costs``."""
        return float(self.invested + self.cash + self.purchase_costs)

    def to_tuple(self) -> tuple:
        return (np.asarray(self.units, dtype=float).copy(), float(self.cash),
                float(self.purchase_costs), float(self.invested),
                np.asarray(self.prices, dtype=float).copy(),
                bool(self.integer_units), str(self.source))

    @classmethod
    def from_tuple(cls, data: tuple) -> 'InitialPosition':
        units, cash, purchase_costs, invested, prices, integer_units, source = data
        return cls(units=units, cash=float(cash),
                   purchase_costs=float(purchase_costs), invested=float(invested),
                   prices=prices, integer_units=bool(integer_units),
                   source=str(source))


def build_initial_position(capital: float, weights: np.ndarray, prices: np.ndarray,
                           policy: LiquidityPolicy, *, integer_units: bool = False,
                           from_balance=None) -> InitialPosition:
    """Translate ``capital`` into ``(units, cash)`` with the rule of section 10.2b.

    Default branch (``from_balance is None``), with ``n_active = #{i : w_i > 1e-12}``::

        investable     = capital - policy.cash_reserve_initial
        notional       = compute_notional_budget(investable, pct_fee, fixed_fee, n_active)
        units          = notional * weights / prices        (fractional)
        invested       = sum(units * prices)
        purchase_costs = pct_fee * invested + fixed_fee * n_charged
        cash           = capital - invested - purchase_costs   >= cash_reserve_initial

    ``compute_notional_budget`` is IMPORTED from core/purchase.py, not
    reimplemented: the notional budget has a single implementation (FD-5), which
    is what ``test_notional_budget_shared_by_purchase_and_cashflow`` pins down.
    With ``integer_units=True`` the allocation is delegated to
    ``core.purchase.allocate_integer_units`` and inherits its budget guarantee;
    ``n_charged = #{i : units_i > 0}`` so the fixed fee matches the assets really
    bought (identical to ``n_active`` in the fractional branch).

    Raises ``InfeasibleConstraintsError('initial_position_capital_consumed')``
    when ``investable <= 0`` or ``notional <= 0``, before any simulation.
    """
    w = np.asarray(weights, dtype=float).ravel()
    price = np.asarray(prices, dtype=float).ravel()
    if price.size != w.size:
        raise PortfolioError('cashflow_inputs_length_mismatch', n_weights=int(w.size),
                             n_prices=int(price.size))
    if np.any(price <= 0.0):
        raise PortfolioError('cashflow_price_not_positive')

    if from_balance is not None:
        units = np.asarray(from_balance.rows['target_units'], dtype=float).ravel()
        invested = float(np.sum(units * price))
        return InitialPosition(units=units, cash=float(from_balance.leftover_cash),
                               purchase_costs=float(from_balance.total_costs),
                               invested=invested, prices=price,
                               integer_units=bool(integer_units),
                               source='purchase_balance')

    total = float(capital)
    reserve = float(policy.cash_reserve_initial)
    pct = float(policy.pct_fee)
    fixed = float(policy.fixed_fee)
    n_active = int(np.sum(w > ACTIVE_WEIGHT_TOL))
    investable = total - reserve
    if investable <= 0.0 or n_active == 0:
        raise InfeasibleConstraintsError('initial_position_capital_consumed',
                                         capital=total, reserve=reserve,
                                         fees=fixed * max(n_active, 1))
    notional = compute_notional_budget(investable, pct, fixed, n_active)
    if notional <= 0.0:
        raise InfeasibleConstraintsError('initial_position_capital_consumed',
                                         capital=total, reserve=reserve,
                                         fees=fixed * n_active)

    if integer_units:
        units, _, _ = allocate_integer_units(w * notional, price, investable, pct,
                                             fixed, np.zeros(w.size, dtype=bool))
    else:
        units = np.where(w != 0.0, notional * w / price, 0.0)

    invested = float(np.sum(units * price))
    n_charged = int(np.sum(units > 0.0))
    purchase_costs = float(pct * invested + fixed * n_charged)
    cash = float(total - invested - purchase_costs)
    return InitialPosition(units=units, cash=cash, purchase_costs=purchase_costs,
                           invested=invested, prices=price,
                           integer_units=bool(integer_units), source='synthetic')


# --------------------------------------------------------------------------- #
# Forced sale
# --------------------------------------------------------------------------- #
def solve_gross_sale(need, pct_fee: float, fixed_fee: float, n_sellable) -> np.ndarray:
    """``(need + fixed_fee*n_sellable) / (1 - pct_fee)``.

    Solves the coverage equation instead of applying the cost afterwards, so
    ``gross*(1 - pct_fee) - fixed_fee*n_sellable == need`` holds exactly.
    """
    pct = float(pct_fee)
    if pct >= 1.0:
        raise InfeasibleConstraintsError('pct_fee_not_below_one', pct_fee=pct)
    required = np.asarray(need, dtype=float)
    count = np.asarray(n_sellable, dtype=float)
    return (required + float(fixed_fee) * count) / (1.0 - pct)


# --------------------------------------------------------------------------- #
# Engine result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CashflowPathResult:
    """Per-path outcome of the payment engine (section 10.3).

    ``total_gross_sold`` and ``note_codes`` complete the list: the first is what
    the forced-sale assertions read (gross sold vs. amount paid) and the second
    carries the ledger codes of a vectorised run, where no per-row ledger exists.
    """

    n_paths: int
    covered_all: np.ndarray
    n_defaults: np.ndarray
    first_default_idx: np.ndarray
    max_deficit: np.ndarray
    cumulative_deficit: np.ndarray
    final_wealth: np.ndarray
    min_wealth: np.ndarray
    total_costs: np.ndarray
    total_sale_costs: np.ndarray
    total_rebalance_costs: np.ndarray
    total_paid: np.ndarray
    total_external: np.ndarray
    total_returns: np.ndarray
    coverage_by_installment: np.ndarray
    wealth_at_payments: np.ndarray
    cash_at_payments: np.ndarray
    total_gross_sold: np.ndarray
    note_codes: tuple = ()


def _check_identity_steps(wealth_open, returns_in_period, rebalance_pre, wealth_before,
                          external_income, paid, sale_costs, rebalance_post,
                          wealth_after, wealth_initial: float, tol: float) -> None:
    """Vectorised step-by-step identity check; raises ``PortfolioError`` on failure.

    Shared by ``check_wealth_identity`` (ledger form) and by the vectorised engine
    under ``debug_identity=True``, so there is a single implementation.
    """
    tolerance = float(tol) * max(1.0, abs(float(wealth_initial)))
    first_step = wealth_open + returns_in_period - rebalance_pre
    gap = np.abs(first_step - wealth_before)
    if gap.size and float(np.max(gap)) > tolerance:
        raise PortfolioError('wealth_identity_violated', step='wealth_before',
                             max_gap=float(np.max(gap)), tolerance=tolerance)
    second_step = (wealth_before + external_income - paid - sale_costs
                   - rebalance_post)
    gap = np.abs(second_step - wealth_after)
    if gap.size and float(np.max(gap)) > tolerance:
        raise PortfolioError('wealth_identity_violated', step='wealth_after',
                             max_gap=float(np.max(gap)), tolerance=tolerance)
    if wealth_open.size:
        opening = np.atleast_2d(wealth_open)
        closing = np.atleast_2d(wealth_after)
        gap = float(np.max(np.abs(opening[:, 0] - float(wealth_initial))))
        if gap > tolerance:
            raise PortfolioError('wealth_identity_violated', step='wealth_open_first',
                                 max_gap=gap, tolerance=tolerance)
        if opening.shape[1] > 1:
            chain = np.abs(opening[:, 1:] - closing[:, :-1])
            gap = float(np.max(chain))
            if gap > tolerance:
                raise PortfolioError('wealth_identity_violated', step='wealth_open_chain',
                                     max_gap=gap, tolerance=tolerance)


def check_wealth_identity(ledger: pd.DataFrame, wealth_initial: float,
                          tol: float = 1e-6) -> None:
    """Verify the two steps of section 10.3 over a row-per-installment ledger.

    Tolerance ``tol * max(1, wealth_initial)``. A violation raises
    ``PortfolioError``: it is a programming failure, never a user condition, and
    the UI shows it as an error instead of a result.
    """
    frame = ledger
    required = ('wealth_open', 'returns_in_period', 'rebalance_costs_pre',
                'wealth_before', 'external_income', 'paid', 'sale_costs',
                'rebalance_costs_post', 'wealth_after')
    for column in required:
        if column not in frame.columns:
            raise PortfolioError('ledger_column_missing', column=column)
    values = {name: frame[name].to_numpy(dtype=float).reshape(1, -1)
              for name in required}
    _check_identity_steps(values['wealth_open'], values['returns_in_period'],
                          values['rebalance_costs_pre'], values['wealth_before'],
                          values['external_income'], values['paid'],
                          values['sale_costs'], values['rebalance_costs_post'],
                          values['wealth_after'], float(wealth_initial), float(tol))


# --------------------------------------------------------------------------- #
# Vectorised payment engine (section 10.2, steps 1-10)
# --------------------------------------------------------------------------- #
def _simulate_batch(returns: np.ndarray, initial: InitialPosition,
                    schedule: InstallmentSchedule, payment_indices: np.ndarray,
                    policy: LiquidityPolicy, rf_simple_daily: float,
                    rebalance_indices, target_weights, collect_ledger: bool) -> dict:
    """Run one ``(batch, horizon, n)`` block of log returns through the engine.

    The Python loop walks payment and rebalance DATES (<= K + M iterations); every
    magnitude is an array over paths, so days x paths is never iterated.
    """
    block = np.asarray(returns, dtype=float)
    if block.ndim != 3:
        raise PortfolioError('cashflow_batch_not_three_dimensional', ndim=int(block.ndim))
    n_paths, horizon, n_assets = block.shape
    if n_paths == 0:
        raise PortfolioError('cashflow_batch_is_empty')
    if horizon < 1:
        raise PortfolioError('cashflow_horizon_not_positive', horizon=int(horizon))

    price0 = np.asarray(initial.prices, dtype=float)
    if price0.size != n_assets:
        raise PortfolioError('cashflow_asset_count_mismatch', n_prices=int(price0.size),
                             n_batch=int(n_assets))

    n_installments = schedule.n_installments
    pay = np.asarray(payment_indices, dtype=int).ravel()
    if pay.size != n_installments:
        raise PortfolioError('cashflow_payment_indices_mismatch',
                             n_indices=int(pay.size), n_installments=int(n_installments))
    if int(pay.min()) < 1 or int(pay.max()) > horizon:
        raise PortfolioError('cashflow_payment_index_out_of_range',
                             minimum=int(pay.min()), maximum=int(pay.max()),
                             horizon=int(horizon))

    if rebalance_indices is None:
        rebalance = np.empty(0, dtype=int)
    else:
        rebalance = np.unique(np.asarray(rebalance_indices, dtype=int).ravel())
    if rebalance.size:
        if int(rebalance.min()) < 1 or int(rebalance.max()) > horizon:
            raise PortfolioError('cashflow_rebalance_index_out_of_range',
                                 minimum=int(rebalance.min()),
                                 maximum=int(rebalance.max()), horizon=int(horizon))
        if target_weights is None:
            raise PortfolioError('cashflow_rebalance_without_target_weights')
    target = (None if target_weights is None
              else np.asarray(target_weights, dtype=float).ravel())
    if target is not None and target.size != n_assets:
        raise PortfolioError('cashflow_inputs_length_mismatch',
                             n_weights=int(target.size), n_prices=int(n_assets))

    pct = float(policy.pct_fee)
    fixed = float(policy.fixed_fee)
    rf = float(rf_simple_daily)
    earn_interest = bool(policy.cash_interest) and rf != 0.0
    allow_sales = bool(policy.allow_sales)

    cumulative = np.cumsum(block, axis=1)
    units = np.repeat(np.asarray(initial.units, dtype=float)[None, :], n_paths, axis=0)
    units = np.array(units, dtype=float, copy=True)
    cash = np.full(n_paths, float(initial.cash))
    wealth_initial = float(initial.wealth_initial)
    wealth_prev = np.full(n_paths, wealth_initial)

    shape = (n_paths, n_installments)
    due_col = np.zeros(shape)
    wealth_open = np.zeros(shape)
    returns_in_period = np.zeros(shape)
    wealth_before = np.zeros(shape)
    cash_before = np.zeros(shape)
    mv_before = np.zeros(shape)
    external_col = np.zeros(shape)
    gross_sold = np.zeros(shape)
    sale_costs = np.zeros(shape)
    rebalance_pre = np.zeros(shape)
    rebalance_post = np.zeros(shape)
    paid_col = np.zeros(shape)
    deficit_col = np.zeros(shape)
    cash_after = np.zeros(shape)
    mv_after = np.zeros(shape)
    wealth_after = np.zeros(shape)
    covered = np.zeros(shape, dtype=bool)
    row_notes = [''] * n_installments
    note_codes = set()

    segment_returns = np.zeros(n_paths)
    pending_pre = np.zeros(n_paths)
    tail_rebalance_costs = np.zeros(n_paths)
    price_last = np.repeat(price0[None, :], n_paths, axis=0)
    last_index = 0
    processed = 0

    # Payments sort before rebalances at the same index: paying is the
    # obligation, recomposing is discretionary (section 11.3 point 2).
    events = [(int(index), 0, k) for k, index in enumerate(pay)]
    events += [(int(index), 1, -1) for index in rebalance]
    events.sort(key=lambda item: (item[0], item[1]))
    events.append((int(horizon), 2, -1))

    for index, kind, k in events:
        # Step 1: revaluation. Index `index` consumes the FIRST `index` rows.
        if index == 0:
            price_now = np.repeat(price0[None, :], n_paths, axis=0)
        else:
            price_now = price0[None, :] * np.exp(cumulative[:, index - 1, :])
        segment_returns = segment_returns + np.sum(units * (price_now - price_last),
                                                   axis=1)
        elapsed = index - last_index
        # Step 2: cash interest, compounded over the elapsed business days.
        if earn_interest and elapsed > 0:
            accrued = cash * ((1.0 + rf) ** elapsed - 1.0)
            cash = cash + accrued
            segment_returns = segment_returns + accrued
        price_last = price_now
        last_index = index
        # Step 3: no maintenance costs are modelled (RF-C5(2) holds vacuously).

        if kind == 2:
            break

        if kind == 0:
            market_value = units * price_now
            market_total = market_value.sum(axis=1)
            wealth_open[:, k] = wealth_prev
            returns_in_period[:, k] = segment_returns
            rebalance_pre[:, k] = pending_pre
            cash_before[:, k] = cash
            mv_before[:, k] = market_total
            wealth_before[:, k] = cash + market_total

            # Step 4: external income, collected before the payment.
            income = float(schedule.external_income[k])
            external_col[:, k] = income
            cash = cash + income

            # Step 5 and 6: amount due and cash shortfall.
            due = float(schedule.amounts[k])
            due_col[:, k] = due
            need = np.maximum(due - cash, 0.0)

            # Step 7: forced sale, pro rata on market value.
            if allow_sales and bool(np.any(need > 0.0)):
                sellable = market_value > MARKET_VALUE_FLOOR
                n_sellable = sellable.sum(axis=1).astype(float)
                required = solve_gross_sale(need, pct, fixed, n_sellable)
                candidate = np.minimum(required, market_total)
                net = candidate * (1.0 - pct) - fixed * n_sellable
                feasible = (need > 0.0) & (market_total > MARKET_VALUE_FLOOR) \
                    & (n_sellable > 0.0)
                execute = feasible & (net > 0.0)
                if bool(np.any(feasible & ~execute)):
                    note_codes.add(SALE_NOT_EXECUTED)
                    row_notes[k] = SALE_NOT_EXECUTED
                gross = np.where(execute, candidate, 0.0)
                proceeds = np.where(execute, np.maximum(net, 0.0), 0.0)
                row_sale_costs = gross - proceeds
                safe_total = np.where(market_total > 0.0, market_total, 1.0)
                share = np.where(market_total[:, None] > 0.0,
                                 market_value / safe_total[:, None], 0.0)
                sold_value = gross[:, None] * share
                units = units - sold_value / price_now
                cash = cash + proceeds
                gross_sold[:, k] = gross
                sale_costs[:, k] = row_sale_costs

            # Step 8: payment, bounded by the cash available.
            payment = np.minimum(cash, due)
            cash = cash - payment
            shortfall = due - payment
            paid_col[:, k] = payment
            deficit_col[:, k] = shortfall
            covered[:, k] = shortfall <= COVERED_TOL

            # Step 9: record. Step 10 (rule D-7) needs nothing extra: cash and
            # units are non-negative by construction and the deficit becomes an
            # unpaid obligation, never a negative balance nor interest-bearing debt.
            closing_value = (units * price_now).sum(axis=1)
            cash_after[:, k] = cash
            mv_after[:, k] = closing_value
            wealth_after[:, k] = cash + closing_value
            wealth_prev = wealth_after[:, k].copy()
            segment_returns = np.zeros(n_paths)
            pending_pre = np.zeros(n_paths)
            processed = k + 1
            continue

        # Periodic rebalance (section 11.3), processed after the payment.
        market_value = units * price_now
        market_total = market_value.sum(axis=1)
        active = market_total > MARKET_VALUE_FLOOR
        safe_total = np.where(active, market_total, 1.0)
        current = np.where(active[:, None], market_value / safe_total[:, None], 0.0)
        drift = np.abs(target[None, :] - current)
        turnover = drift.sum(axis=1)
        n_traded = (drift * market_total[:, None] > MARKET_VALUE_FLOOR).sum(axis=1)
        raw_cost = turnover * market_total * pct + fixed * n_traded.astype(float)
        execute = active & (raw_cost < market_total)
        if bool(np.any(~active)):
            note_codes.add(REBALANCE_SKIPPED_NO_VALUE)
        if bool(np.any(active & ~execute)):
            note_codes.add(REBALANCE_SKIPPED_COST)
        cost = np.where(execute, raw_cost, 0.0)
        from_cash = np.minimum(cash, cost)
        cash = cash - from_cash
        remainder = market_total - (cost - from_cash)
        units = np.where(execute[:, None],
                         target[None, :] * remainder[:, None] / price_now, units)

        if processed > 0 and int(pay[processed - 1]) == index:
            # The cost lands in `post` and the closing columns of that
            # installment are re-measured, so the ledger row stays consistent.
            rebalance_post[:, processed - 1] += cost
            cash_after[:, processed - 1] = cash
            mv_after[:, processed - 1] = (units * price_now).sum(axis=1)
            wealth_after[:, processed - 1] = (cash_after[:, processed - 1]
                                             + mv_after[:, processed - 1])
            wealth_prev = wealth_after[:, processed - 1].copy()
        elif processed < n_installments:
            pending_pre = pending_pre + cost
        else:
            tail_rebalance_costs = tail_rebalance_costs + cost

    final_wealth = cash + (units * price_last).sum(axis=1)
    total_returns = returns_in_period.sum(axis=1) + segment_returns
    total_sale_costs = sale_costs.sum(axis=1)
    total_rebalance_costs = (rebalance_pre.sum(axis=1) + rebalance_post.sum(axis=1)
                             + tail_rebalance_costs)
    total_paid = paid_col.sum(axis=1)
    total_external = external_col.sum(axis=1)
    total_costs = total_sale_costs + total_rebalance_costs

    expected = (wealth_initial + total_returns + total_external - total_paid
                - total_costs)
    tolerance = IDENTITY_TOL * max(1.0, abs(wealth_initial))
    gap = float(np.max(np.abs(final_wealth - expected)))
    if gap > tolerance:
        raise PortfolioError('wealth_identity_violated', step='aggregate',
                             max_gap=gap, tolerance=tolerance)

    uncovered = ~covered
    any_default = uncovered.any(axis=1)
    first_default = np.where(any_default, np.argmax(uncovered, axis=1), -1)

    result = {
        'n_paths': int(n_paths),
        'covered_all': covered.all(axis=1),
        'n_defaults': uncovered.sum(axis=1).astype(int),
        'first_default_idx': first_default.astype(int),
        'max_deficit': deficit_col.max(axis=1),
        'cumulative_deficit': deficit_col.sum(axis=1),
        'final_wealth': final_wealth,
        'min_wealth': np.minimum(wealth_before.min(axis=1), wealth_after.min(axis=1)),
        'total_costs': total_costs,
        'total_sale_costs': total_sale_costs,
        'total_rebalance_costs': total_rebalance_costs,
        'total_paid': total_paid,
        'total_external': total_external,
        'total_returns': total_returns,
        'covered_count': covered.sum(axis=0).astype(float),
        'wealth_at_payments': wealth_after,
        'cash_at_payments': cash_after,
        'total_gross_sold': gross_sold.sum(axis=1),
        'note_codes': note_codes,
        'tail_rebalance_costs': tail_rebalance_costs,
        'steps': (wealth_open, returns_in_period, rebalance_pre, wealth_before,
                  external_col, paid_col, sale_costs, rebalance_post, wealth_after),
    }
    if collect_ledger:
        result['ledger_data'] = {
            'due': due_col[0], 'wealth_open': wealth_open[0],
            'returns_in_period': returns_in_period[0],
            'wealth_before': wealth_before[0], 'cash_before': cash_before[0],
            'mv_before': mv_before[0], 'external_income': external_col[0],
            'gross_sold': gross_sold[0], 'sale_costs': sale_costs[0],
            'rebalance_costs_pre': rebalance_pre[0],
            'rebalance_costs_post': rebalance_post[0],
            'rebalance_costs': rebalance_pre[0] + rebalance_post[0],
            'paid': paid_col[0], 'deficit': deficit_col[0],
            'cash_after': cash_after[0], 'mv_after': mv_after[0],
            'wealth_after': wealth_after[0], 'covered': covered[0],
            'note_code': list(row_notes),
        }
    return result


def simulate_cashflow_paths(path_batches: Iterable[np.ndarray], initial: InitialPosition,
                            schedule: InstallmentSchedule, payment_indices: np.ndarray,
                            policy: LiquidityPolicy, rf_simple_daily: float,
                            rebalance_indices=None, target_weights=None, *,
                            debug_identity: bool = False) -> CashflowPathResult:
    """Consume the batch iterator of section 11.3 and aggregate.

    Path-level arrays are concatenated in batch order and
    ``coverage_by_installment`` is a count-weighted mean, so the result is
    bit-identical whatever the batch size. Passing a single
    ``(n_paths, horizon, n)`` array wrapped in a one-element list is the
    degenerate case. The aggregate identity is checked ALWAYS (inside every
    batch); the step-by-step one only with ``debug_identity=True``, for cost.
    """
    schedule, payment_indices, _, notes = aggregate_duplicate_payments(
        schedule, payment_indices)
    collected = []
    note_codes = set(notes)
    for batch in path_batches:
        outcome = _simulate_batch(batch, initial, schedule, payment_indices, policy,
                                  rf_simple_daily, rebalance_indices, target_weights,
                                  collect_ledger=False)
        if debug_identity:
            _check_identity_steps(*outcome['steps'], initial.wealth_initial,
                                 IDENTITY_TOL)
        note_codes |= outcome['note_codes']
        collected.append(outcome)
    if not collected:
        raise PortfolioError('cashflow_no_batches')
    return _combine(collected, tuple(sorted(note_codes)))


def _combine(batches: list, note_codes: tuple) -> CashflowPathResult:
    """Concatenate per-path arrays and average coverage weighted by path count."""
    def stack(key, dtype=float):
        return _lock(np.concatenate([item[key] for item in batches]), dtype)

    total_paths = int(sum(item['n_paths'] for item in batches))
    counts = np.sum([item['covered_count'] for item in batches], axis=0)
    coverage = counts / float(total_paths)
    return CashflowPathResult(
        n_paths=total_paths,
        covered_all=stack('covered_all', bool),
        n_defaults=stack('n_defaults', int),
        first_default_idx=stack('first_default_idx', int),
        max_deficit=stack('max_deficit'),
        cumulative_deficit=stack('cumulative_deficit'),
        final_wealth=stack('final_wealth'),
        min_wealth=stack('min_wealth'),
        total_costs=stack('total_costs'),
        total_sale_costs=stack('total_sale_costs'),
        total_rebalance_costs=stack('total_rebalance_costs'),
        total_paid=stack('total_paid'),
        total_external=stack('total_external'),
        total_returns=stack('total_returns'),
        coverage_by_installment=_lock(coverage),
        wealth_at_payments=_lock(np.concatenate(
            [item['wealth_at_payments'] for item in batches], axis=0)),
        cash_at_payments=_lock(np.concatenate(
            [item['cash_at_payments'] for item in batches], axis=0)),
        total_gross_sold=stack('total_gross_sold'),
        note_codes=note_codes,
    )


def simulate_cashflow_single(path_log_returns_2d: np.ndarray, initial: InitialPosition,
                             schedule: InstallmentSchedule, payment_indices: np.ndarray,
                             policy: LiquidityPolicy, rf_simple_daily: float,
                             rebalance_indices=None, target_weights=None,
                             effective_payment_dates: pd.DatetimeIndex = None) -> tuple:
    """``n_paths = 1`` wrapper that also materialises the row-per-installment ledger.

    Returns ``(ledger, result)``. The ledger is indexed by
    ``effective_payment_dates`` when supplied (RF-C19 F) and the step-by-step
    identity is verified here ALWAYS. There is no second implementation of the
    bookkeeping: this calls the same engine with a one-path batch.
    """
    path = np.asarray(path_log_returns_2d, dtype=float)
    if path.ndim != 2:
        raise PortfolioError('cashflow_path_not_two_dimensional', ndim=int(path.ndim))
    schedule, payment_indices, effective_payment_dates, notes = \
        aggregate_duplicate_payments(schedule, payment_indices, effective_payment_dates)
    outcome = _simulate_batch(path[None, :, :], initial, schedule, payment_indices,
                              policy, rf_simple_daily, rebalance_indices,
                              target_weights, collect_ledger=True)
    note_codes = tuple(sorted(outcome['note_codes'] | set(notes)))
    result = _combine([outcome], note_codes)

    data = outcome['ledger_data']
    index = (pd.DatetimeIndex(effective_payment_dates)
             if effective_payment_dates is not None
             else pd.RangeIndex(schedule.n_installments))
    ledger = pd.DataFrame({name: data[name] for name in LEDGER_COLUMNS}, index=index)
    check_wealth_identity(ledger, initial.wealth_initial)
    return ledger, result


# --------------------------------------------------------------------------- #
# Additional capital (section 10.3)
# --------------------------------------------------------------------------- #
def required_additional_capital(initial_capital: float, weights: np.ndarray,
                                prices: np.ndarray, schedule: InstallmentSchedule,
                                payment_indices: np.ndarray, scenario_path: np.ndarray,
                                policy: LiquidityPolicy, rf_simple_daily: float, *,
                                rebalance_indices=None, target_weights=None,
                                integer_units: bool = False, tol: float = 0.01,
                                max_iter: int = 60) -> float:
    """Bisect on TOTAL capital and return ``capital_star - initial_capital``.

    Bracket ``[initial_capital, initial_capital + total_scheduled]``: the upper
    end covers the whole calendar even with zero returns, so it contains the
    solution except in the pathological case, where ``inf`` is returned.
    ``scenario_path`` is 2D ``(horizon, n_assets)``, exactly
    ``deterministic_path(...)[0]``. Every probe goes through
    ``build_initial_position``, so the probed capital is translated with EXACTLY
    the rule of the base case, and the rebalancing arguments are forwarded
    unchanged so the figure answers the same question. The objective is
    non-increasing in capital except for fixed-fee steps, so the result is a
    correct UPPER bound ("estimated additional capital, upper bound" in the UI).
    """
    base = float(initial_capital)
    total_scheduled = schedule.total_scheduled

    def deficit(capital: float) -> float:
        try:
            position = build_initial_position(capital, weights, prices, policy,
                                              integer_units=integer_units)
        except InfeasibleConstraintsError:
            return total_scheduled if total_scheduled > 0.0 else 0.0
        _, outcome = simulate_cashflow_single(scenario_path, position, schedule,
                                              payment_indices, policy, rf_simple_daily,
                                              rebalance_indices, target_weights)
        return float(outcome.cumulative_deficit[0])

    if deficit(base) <= COVERED_TOL:
        return 0.0
    high = base + max(total_scheduled, 0.0)
    if deficit(high) > COVERED_TOL:
        return float('inf')

    low = base
    for _ in range(int(max_iter)):
        if high - low <= float(tol):
            break
        middle = 0.5 * (low + high)
        if deficit(middle) <= COVERED_TOL:
            high = middle
        else:
            low = middle
    return float(high - base)
