"""Numeric formatting, single implementation (design.md section 15).

Consumed by the UI and by the exporter. Thousands separator and decimal mark are
configurable through ``CurrencyFormat``; the defaults follow the Spanish
convention (thousands '.', decimal ','). ``None``, ``NaN`` and ``inf`` render as
the em dash in every formatter, which is the contract the degenerate branch of
``core.stats.realized_metrics`` relies on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

__all__ = [
    'CurrencyFormat',
    'NOT_AVAILABLE',
    'fmt_currency',
    'fmt_pct',
    'fmt_num',
    'fmt_units',
    'sign_class',
    'style_table',
]

NOT_AVAILABLE = '\u2014'


@dataclass(frozen=True)
class CurrencyFormat:
    """Currency rendering options, surfaced as sidebar controls."""

    symbol: str = '$'
    decimals: int = 2
    thousands: str = '.'
    decimal: str = ','
    symbol_first: bool = True

    def to_tuple(self) -> tuple:
        return (self.symbol, int(self.decimals), self.thousands,
                self.decimal, bool(self.symbol_first))

    @classmethod
    def from_tuple(cls, data: tuple) -> 'CurrencyFormat':
        symbol, decimals, thousands, decimal, symbol_first = data
        return cls(symbol=str(symbol), decimals=int(decimals), thousands=str(thousands),
                   decimal=str(decimal), symbol_first=bool(symbol_first))


def _as_finite_float(value: Any) -> float | None:
    """Return ``value`` as a float, or ``None`` when it is missing or not finite."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _group(number: float, decimals: int, thousands: str, decimal: str) -> str:
    """Render ``number`` with grouped thousands and the requested decimal mark."""
    raw = f'{number:,.{decimals}f}'
    return (raw.replace(',', '\x00')
               .replace('.', decimal)
               .replace('\x00', thousands))


def fmt_currency(x: Any, fmt: CurrencyFormat = CurrencyFormat()) -> str:
    """Money amount with symbol, grouped thousands and the configured decimal mark."""
    number = _as_finite_float(x)
    if number is None:
        return NOT_AVAILABLE
    body = _group(number, fmt.decimals, fmt.thousands, fmt.decimal)
    if fmt.symbol_first:
        if body.startswith('-'):
            return f'-{fmt.symbol}{body[1:]}'
        return f'{fmt.symbol}{body}'
    return f'{body} {fmt.symbol}'


def fmt_pct(x: Any, decimals: int = 2, fmt: CurrencyFormat = CurrencyFormat()) -> str:
    """Fraction rendered as a percentage: ``0.1234 -> '12,34 %'``."""
    number = _as_finite_float(x)
    if number is None:
        return NOT_AVAILABLE
    return f'{_group(number * 100.0, decimals, fmt.thousands, fmt.decimal)} %'


def fmt_num(x: Any, decimals: int = 4, fmt: CurrencyFormat = CurrencyFormat()) -> str:
    """Plain number with the configured separators."""
    number = _as_finite_float(x)
    if number is None:
        return NOT_AVAILABLE
    return _group(number, decimals, fmt.thousands, fmt.decimal)


def fmt_units(x: Any, integer: bool, fmt: CurrencyFormat = CurrencyFormat()) -> str:
    """Share count: no decimals when the purchase mode is integer, four otherwise."""
    return fmt_num(x, 0 if integer else 4, fmt)


def sign_class(x: Any) -> str:
    """CSS class driving the green/red colouring of signed figures."""
    number = _as_finite_float(x)
    if number is None or number == 0.0:
        return ''
    return 'pos' if number > 0.0 else 'neg'


def _colour_signed(value: Any) -> str:
    kind = sign_class(value)
    if kind == 'pos':
        return 'color: #4ade80'
    if kind == 'neg':
        return 'color: #f87171'
    return ''


def style_table(df: pd.DataFrame, *,
                pct_cols: Iterable[str] = (),
                money_cols: Iterable[str] = (),
                num_cols: Iterable[str] = (),
                signed_cols: Iterable[str] = (),
                fmt: CurrencyFormat = CurrencyFormat()) -> Any:
    """Return a ``Styler`` with financial formatting and signed colouring.

    Columns absent from ``df`` are ignored, so a caller can pass the full column
    vocabulary of a section without branching on which ones were computed.
    """
    present = set(df.columns)
    formatters: dict = {}
    for column in pct_cols:
        if column in present:
            formatters[column] = lambda v, f=fmt: fmt_pct(v, 2, f)
    for column in money_cols:
        if column in present:
            formatters[column] = lambda v, f=fmt: fmt_currency(v, f)
    for column in num_cols:
        if column in present:
            formatters[column] = lambda v, f=fmt: fmt_num(v, 4, f)

    styler = df.style.format(formatters, na_rep=NOT_AVAILABLE)
    signed = [c for c in signed_cols if c in present]
    if signed:
        elementwise = getattr(styler, 'map', None) or styler.applymap
        styler = elementwise(_colour_signed, subset=signed)
    return styler
