"""Excel workbook builder (design.md section 16).

``core/`` assembles the workbook but owns no visible text: sheet titles and
column headers arrive as mappings from ``ui/labels.py`` (``SHEET_NAMES`` and
``COLUMN_LABELS``). Internal keys are English snake_case; a missing key falls
back to the raw key, so the download never breaks because of a label.

Robustness contract (RF-E4): every sheet is built inside its own ``try`` block.
A sheet that cannot be built is replaced by a one-row sheet carrying the error
note; a section that was never computed is skipped and listed in the parameters
sheet. Neither case aborts the download.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

__all__ = [
    'APP_VERSION',
    'MONTE_CARLO_ROW_CAP',
    'SHEET_KEYS',
    'PCT_FORMAT',
    'MONEY_FORMAT',
    'NUM_FORMAT',
    'ExportPayload',
    'build_workbook',
]

logger = logging.getLogger('portfolio_optimizer')

APP_VERSION = '1.0.0'
MONTE_CARLO_ROW_CAP = 20_000

PCT_FORMAT = '0.00%'
MONEY_FORMAT = '#,##0.00'
NUM_FORMAT = '0.0000'

HEADER_FILL = '141C2E'
HEADER_FONT_COLOR = 'E6EDF3'
MAX_SHEET_TITLE = 31

NOT_COMPUTED = 'not_computed'
SHEET_ERROR = 'sheet_error'
TRUNCATION_NOTE = 'monte_carlo_rows_truncated'
MISSING_SHEETS = 'missing_sheets'

# Internal sheet keys, in workbook order (design.md section 16).
SHEET_KEYS = (
    'parameters',
    'summary',
    'weights',
    'asset_stats',
    'covariance',
    'correlation',
    'efficient_frontier',
    'monte_carlo_weights',
    'purchase_balance',
    'backtest',
    'backtest_series',
    'installment_schedule',
    'initial_position',
    'scenarios',
    'monte_carlo_capacity',
)

WEIGHT_STRATEGY_ORDER = ('min_variance', 'max_sharpe', 'equal_weight')

PURCHASE_ROW_KEYS = ('asset', 'target_weight', 'target_value', 'price', 'current_units',
                     'current_weight', 'target_units', 'delta_units', 'action',
                     'gross_amount', 'costs', 'final_value', 'final_weight',
                     'deviation', 'note_code')

PURCHASE_TOTAL_KEYS = ('capital', 'current_value', 'total_investable', 'notional',
                       'invested', 'buy_gross', 'sell_proceeds', 'total_costs',
                       'leftover_cash', 'max_deviation', 'feasible')

LEDGER_EXPORT_KEYS = ('due', 'wealth_open', 'wealth_before', 'wealth_after', 'sale_costs',
                      'rebalance_costs_pre', 'rebalance_costs_post', 'rebalance_costs',
                      'paid', 'deficit', 'note_code')

SCENARIO_FIELD_KEYS = ('covered_all', 'n_defaults', 'final_wealth', 'min_wealth',
                       'total_paid', 'cumulative_deficit', 'total_costs',
                       'total_gross_sold')

PATH_SPEC_KEYS = ('method', 'n_paths', 'horizon', 'n_assets', 'batch_size', 'block_size',
                  'seed', 'psd_repaired')

PERCENTILE_ORDER = ('p5', 'p25', 'p50', 'p75', 'p95')


# --------------------------------------------------------------------------- #
# Payload
# --------------------------------------------------------------------------- #
@dataclass
class ExportPayload:
    """Everything the workbook can contain. ``None`` means "not computed".

    ``capacity`` carries exactly three elements —the schedule, the aggregated
    metrics and the verdict—. The loose ``CashflowPathResult`` is not part of
    the tuple: the central one already travels inside
    ``CapacityMetrics.deterministic['central']``, together with the ledger and
    the two per-path vectors.

    ``params`` is a free mapping owned by ``ui/``. Two conventions are honoured:
    a value may be the plain value, or a mapping ``{'value': v, 'owner': o}``
    so the parameters sheet can report who set it (design.md section 13.1).
    The optional key ``'initial_position'`` carries the ``InitialPosition``
    dataclass that feeds the ``initial_position`` sheet.
    """

    params: dict = field(default_factory=dict)
    aligned: Any = None
    stats: Any = None
    optimization: dict | None = None
    frontier: Any = None
    weight_cloud: tuple | None = None
    purchase: Any = None
    backtest: Any = None
    capacity: tuple | None = None


@dataclass
class _Sheet:
    """One rendered sheet: English header keys plus already-translated values."""

    header: tuple
    rows: list
    formats: dict = field(default_factory=dict)
    notes: tuple = ()


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _label(labels: dict, key: Any) -> str:
    """Translate ``key`` with ``labels``, falling back to the raw English key."""
    text = str(key)
    value = labels.get(text) if isinstance(labels, dict) else None
    if value is None or str(value) == '':
        return text
    return str(value)


def _cell(value: Any) -> Any:
    """Coerce ``value`` into something openpyxl writes without complaining."""
    if value is None:
        return None
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
        return pd.Timestamp(value).strftime('%Y-%m-%d')
    if value is pd.NaT:
        return None
    return str(value)


def _scalar(value: Any) -> float:
    """First element of an array-like, as a float."""
    array = np.asarray(value, dtype=float).ravel()
    return float(array[0]) if array.size else float('nan')


def _frame_rows(frame: pd.DataFrame) -> list:
    return [[_cell(value) for value in row] for row in frame.itertuples(index=False)]


def _title(sheet_names: dict, key: str) -> str:
    return _label(sheet_names, key)[:MAX_SHEET_TITLE]


# --------------------------------------------------------------------------- #
# Sheet builders. Each returns a ``_Sheet`` or ``None`` when not computed.
# --------------------------------------------------------------------------- #
def _build_summary(payload: ExportPayload, labels: dict) -> _Sheet | None:
    if not payload.optimization:
        return None
    header = ('strategy', 'expected_return_annual', 'volatility_annual', 'sharpe',
              'n_positive', 'certified')
    rows = []
    for key in WEIGHT_STRATEGY_ORDER:
        summary = payload.optimization.get(key)
        if summary is None:
            continue
        rows.append([_label(labels, key),
                     _cell(summary.expected_return_annual),
                     _cell(summary.volatility_annual),
                     _cell(summary.sharpe),
                     _cell(summary.n_positive),
                     _cell(bool(summary.certified))])
    if not rows:
        return None
    return _Sheet(header, rows, {1: PCT_FORMAT, 2: PCT_FORMAT, 3: NUM_FORMAT})


def _build_weights(payload: ExportPayload, labels: dict) -> _Sheet | None:
    if not payload.optimization:
        return None
    present = [key for key in WEIGHT_STRATEGY_ORDER if key in payload.optimization]
    if not present:
        return None
    names = _asset_names(payload)
    header = ('asset',) + tuple(_label(labels, key) for key in present)
    rows = []
    for i, name in enumerate(names):
        row = [name]
        for key in present:
            weights = np.asarray(payload.optimization[key].weights, dtype=float)
            row.append(_cell(weights[i]) if i < weights.size else None)
        rows.append(row)
    formats = {column: PCT_FORMAT for column in range(1, len(present) + 1)}
    return _Sheet(header, rows, formats)


def _build_asset_stats(payload: ExportPayload, labels: dict) -> _Sheet | None:
    stats = payload.stats
    if stats is None:
        return None
    header = ('asset', 'n_obs', 'annual_return', 'annual_volatility', 'sharpe',
              'min_return', 'max_return')
    rows = []
    for i, name in enumerate(stats.names):
        rows.append([str(name),
                     _cell(stats.n_obs),
                     _cell(stats.mu_annual[i]),
                     _cell(stats.sigma_annual[i]),
                     _cell(stats.sharpe_individual[i]),
                     _cell(stats.min_return[i]),
                     _cell(stats.max_return[i])])
    formats = {2: PCT_FORMAT, 3: PCT_FORMAT, 4: NUM_FORMAT, 5: PCT_FORMAT, 6: PCT_FORMAT}
    return _Sheet(header, rows, formats)


def _matrix_sheet(matrix: Any, names: tuple, number_format: str) -> _Sheet:
    data = np.asarray(matrix, dtype=float)
    header = ('asset',) + tuple(str(name) for name in names)
    rows = [[str(names[i])] + [_cell(value) for value in data[i]]
            for i in range(len(names))]
    formats = {column: number_format for column in range(1, len(names) + 1)}
    return _Sheet(header, rows, formats)


def _build_covariance(payload: ExportPayload, labels: dict) -> _Sheet | None:
    if payload.stats is None:
        return None
    return _matrix_sheet(payload.stats.cov_annual, payload.stats.names, NUM_FORMAT)


def _build_correlation(payload: ExportPayload, labels: dict) -> _Sheet | None:
    if payload.stats is None:
        return None
    return _matrix_sheet(payload.stats.corr, payload.stats.names, NUM_FORMAT)


def _build_frontier(payload: ExportPayload, labels: dict) -> _Sheet | None:
    frontier = payload.frontier
    if frontier is None or frontier.points is None or frontier.points.empty:
        return None
    names = _asset_names(payload)
    columns = list(frontier.points.columns)
    header = []
    for column in columns:
        if str(column).startswith('w_'):
            index = int(str(column)[2:])
            header.append(str(names[index]) if index < len(names) else str(column))
        else:
            header.append(str(column))
    rows = _frame_rows(frontier.points)
    formats = {0: PCT_FORMAT, 1: PCT_FORMAT, 2: NUM_FORMAT}
    for position in range(3, len(columns)):
        formats[position] = PCT_FORMAT
    return _Sheet(tuple(header), rows, formats)


def _build_monte_carlo_weights(payload: ExportPayload, labels: dict) -> _Sheet | None:
    cloud = payload.weight_cloud
    if cloud is None:
        return None
    vol, ret, sharpe = (np.asarray(item, dtype=float).ravel() for item in cloud[:3])
    total = int(min(vol.size, ret.size, sharpe.size))
    if total == 0:
        return None
    kept = min(total, MONTE_CARLO_ROW_CAP)
    header = ('index', 'vol', 'ret', 'sharpe')
    rows = [[i + 1, _cell(vol[i]), _cell(ret[i]), _cell(sharpe[i])] for i in range(kept)]
    notes = ()
    if kept < total:
        notes = (f'{_label(labels, TRUNCATION_NOTE)} {kept}/{total}',)
    return _Sheet(header, rows, {1: PCT_FORMAT, 2: PCT_FORMAT, 3: NUM_FORMAT}, notes)


def _build_purchase(payload: ExportPayload, labels: dict) -> _Sheet | None:
    balance = payload.purchase
    if balance is None:
        return None
    frame = balance.rows.copy()
    header = tuple(key for key in PURCHASE_ROW_KEYS if key in frame.columns)
    rows = []
    for record in frame.to_dict('records'):
        row = []
        for key in header:
            value = record.get(key)
            if key in ('action', 'note_code') and value not in (None, ''):
                value = _label(labels, value)
            row.append(_cell(value))
        rows.append(row)
    rows.append([None] * len(header))
    for key in PURCHASE_TOTAL_KEYS:
        if not hasattr(balance, key):
            continue
        row = [None] * len(header)
        row[0] = _label(labels, key)
        if len(header) > 1:
            row[1] = _cell(getattr(balance, key))
        rows.append(row)
    formats = {}
    for position, key in enumerate(header):
        if key in ('target_weight', 'current_weight', 'final_weight', 'deviation'):
            formats[position] = PCT_FORMAT
        elif key in ('target_value', 'price', 'gross_amount', 'costs', 'final_value'):
            formats[position] = MONEY_FORMAT
        elif key in ('current_units', 'target_units', 'delta_units'):
            formats[position] = NUM_FORMAT
    return _Sheet(header, rows, formats)


def _build_backtest(payload: ExportPayload, labels: dict) -> _Sheet | None:
    result = payload.backtest
    if result is None or result.metrics is None or result.metrics.empty:
        return None
    metrics = result.metrics.copy()
    header = ('strategy',) + tuple(str(column) for column in metrics.columns)
    rows = []
    for key, record in metrics.iterrows():
        rows.append([_label(labels, key)] + [_cell(value) for value in record.to_list()])
    formats = {}
    for position, column in enumerate(metrics.columns, start=1):
        name = str(column)
        if name in ('total_return', 'cagr', 'mu_cc_real', 'vol_real', 'max_drawdown'):
            formats[position] = PCT_FORMAT
        elif name in ('final_capital', 'total_costs'):
            formats[position] = MONEY_FORMAT
        else:
            formats[position] = NUM_FORMAT
    return _Sheet(header, rows, formats)


def _build_backtest_series(payload: ExportPayload, labels: dict) -> _Sheet | None:
    result = payload.backtest
    if result is None or result.equity is None or result.equity.empty:
        return None
    equity = result.equity.copy()
    header = ('date',) + tuple(_label(labels, column) for column in equity.columns)
    rows = [[_cell(index)] + [_cell(value) for value in record.to_list()]
            for index, record in equity.iterrows()]
    formats = {column: MONEY_FORMAT for column in range(1, equity.shape[1] + 1)}
    return _Sheet(header, rows, formats)


def _build_installments(payload: ExportPayload, labels: dict) -> _Sheet | None:
    if payload.capacity is None:
        return None
    metrics = payload.capacity[1]
    ledger = getattr(metrics, 'central_ledger', None)
    if ledger is None or len(ledger) == 0:
        return None
    frame = ledger.copy()
    keys = tuple(key for key in LEDGER_EXPORT_KEYS if key in frame.columns)
    header = ('date',) + keys
    rows = []
    for index, record in frame.iterrows():
        row = [_cell(index)]
        for key in keys:
            value = record[key]
            if key == 'note_code' and value not in (None, ''):
                value = _label(labels, value)
            row.append(_cell(value))
        rows.append(row)
    formats = {}
    for position, key in enumerate(keys, start=1):
        if key != 'note_code':
            formats[position] = MONEY_FORMAT
    return _Sheet(header, rows, formats)


def _build_initial_position(payload: ExportPayload, labels: dict) -> _Sheet | None:
    position = payload.params.get('initial_position') if payload.params else None
    if position is None:
        return None
    names = _asset_names(payload)
    units = np.asarray(position.units, dtype=float).ravel()
    prices = np.asarray(position.prices, dtype=float).ravel()
    header = ('asset', 'units', 'price', 'value')
    rows = []
    for i in range(units.size):
        name = str(names[i]) if i < len(names) else str(i)
        rows.append([name, _cell(units[i]), _cell(prices[i]),
                     _cell(units[i] * prices[i])])
    rows.append([None] * len(header))
    for key in ('invested', 'purchase_costs', 'cash', 'wealth_initial', 'capital'):
        rows.append([_label(labels, key), _cell(getattr(position, key, None)), None, None])
    rows.append([_label(labels, 'integer_units'), _cell(bool(position.integer_units)),
                 None, None])
    rows.append([_label(labels, 'position_source'), _label(labels, position.source),
                 None, None])
    return _Sheet(header, rows, {1: NUM_FORMAT, 2: MONEY_FORMAT, 3: MONEY_FORMAT})


def _build_scenarios(payload: ExportPayload, labels: dict) -> _Sheet | None:
    if payload.capacity is None:
        return None
    metrics = payload.capacity[1]
    deterministic = getattr(metrics, 'deterministic', None) or {}
    if not deterministic:
        return None
    header = ('scenario',) + SCENARIO_FIELD_KEYS
    rows = []
    for key in ('conservative', 'central', 'favorable'):
        result = deterministic.get(key)
        if result is None:
            continue
        row = [_label(labels, key)]
        for field_key in SCENARIO_FIELD_KEYS:
            value = getattr(result, field_key, None)
            if value is None:
                row.append(None)
            elif field_key == 'covered_all':
                row.append(_cell(bool(np.asarray(value).ravel()[0])))
            else:
                row.append(_cell(_scalar(value)))
        rows.append(row)
    if not rows:
        return None
    formats = {}
    for position, field_key in enumerate(SCENARIO_FIELD_KEYS, start=1):
        if field_key in ('covered_all', 'n_defaults'):
            continue
        formats[position] = MONEY_FORMAT
    return _Sheet(header, rows, formats)


def _build_monte_carlo_capacity(payload: ExportPayload, labels: dict) -> _Sheet | None:
    if payload.capacity is None:
        return None
    schedule, metrics, verdict = payload.capacity
    header = ('item', 'value', 'threshold', 'comparator', 'passed')
    rows = [
        [_label(labels, 'initial_capital'), _cell(metrics.initial_capital), None, None, None],
        [_label(labels, 'total_scheduled'), _cell(metrics.total_scheduled), None, None, None],
        [_label(labels, 'n_installments'), _cell(metrics.n_installments), None, None, None],
        [_label(labels, 'p_cover_all'), _cell(metrics.p_cover_all), None, None, None],
        [_label(labels, 'margin_ratio'), _cell(metrics.margin_ratio), None, None, None],
        [_label(labels, 'conservative_deficit_ratio'),
         _cell(metrics.conservative_deficit_ratio), None, None, None],
        [_label(labels, 'additional_capital_required'),
         _cell(metrics.additional_capital_required), None, None, None],
        [_label(labels, 'n_paths'), _cell(metrics.mc_final_wealth.size), None, None, None],
    ]
    percentiles = metrics.final_wealth_percentiles or {}
    for key in PERCENTILE_ORDER:
        if key in percentiles:
            rows.append([_label(labels, key), _cell(percentiles[key]), None, None, None])
    if verdict is not None:
        rows.append([_label(labels, 'category'), _label(labels, verdict.category),
                     None, None, None])
        rows.append([_label(labels, 'rule_fired'), _cell(verdict.rule_fired),
                     None, None, None])
        for criterion in verdict.criteria:
            rows.append([_label(labels, criterion.name),
                         _cell(criterion.value),
                         _cell(criterion.threshold),
                         str(criterion.comparator),
                         _cell(bool(criterion.passed))])
    spec = getattr(metrics, 'path_spec', None)
    if spec is not None:
        rows.append([None] * len(header))
        for key in PATH_SPEC_KEYS:
            rows.append([_label(labels, key), _cell(getattr(spec, key, None)),
                         None, None, None])
    coverage = np.asarray(metrics.coverage_by_installment, dtype=float).ravel()
    if coverage.size:
        rows.append([None] * len(header))
        base = _label(labels, 'coverage_by_installment')
        for k in range(coverage.size):
            rows.append([f'{base} {k + 1}', _cell(coverage[k]), None, None, None])
    for code in getattr(metrics, 'note_codes', ()):
        rows.append([_label(labels, code), None, None, None, None])
    return _Sheet(header, rows, {})


def _asset_names(payload: ExportPayload) -> tuple:
    if payload.stats is not None:
        return tuple(str(name) for name in payload.stats.names)
    if payload.aligned is not None:
        return tuple(str(name) for name in payload.aligned.names)
    return ()


_BUILDERS: dict = {
    'summary': _build_summary,
    'weights': _build_weights,
    'asset_stats': _build_asset_stats,
    'covariance': _build_covariance,
    'correlation': _build_correlation,
    'efficient_frontier': _build_frontier,
    'monte_carlo_weights': _build_monte_carlo_weights,
    'purchase_balance': _build_purchase,
    'backtest': _build_backtest,
    'backtest_series': _build_backtest_series,
    'installment_schedule': _build_installments,
    'initial_position': _build_initial_position,
    'scenarios': _build_scenarios,
    'monte_carlo_capacity': _build_monte_carlo_capacity,
}


# --------------------------------------------------------------------------- #
# Parameters sheet (built last, written first)
# --------------------------------------------------------------------------- #
def _build_parameters(payload: ExportPayload, labels: dict, missing: list) -> _Sheet:
    header = ('parameter', 'value', 'owner')
    rows = []
    for key, raw in (payload.params or {}).items():
        if key == 'initial_position':
            continue
        owner = ''
        value = raw
        if isinstance(raw, dict) and 'value' in raw:
            value = raw.get('value')
            owner = str(raw.get('owner', ''))
        rows.append([_label(labels, key), _cell(value), _label(labels, owner) if owner else ''])
    rows.append([_label(labels, 'app_version'), APP_VERSION, ''])
    rows.append([_label(labels, 'generated_at'),
                 dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ''])
    if missing:
        rows.append([None, None, None])
        rows.append([_label(labels, MISSING_SHEETS), None, None])
        for key in missing:
            rows.append([_label(labels, key), _label(labels, NOT_COMPUTED), ''])
    return _Sheet(header, rows, {})


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #
def _write_sheet(workbook: Workbook, title: str, sheet: _Sheet, labels: dict) -> None:
    worksheet = workbook.create_sheet(title=title)
    worksheet.append([_label(labels, key) for key in sheet.header])
    fill = PatternFill('solid', fgColor=HEADER_FILL)
    font = Font(bold=True, color=HEADER_FONT_COLOR)
    for cell in worksheet[1]:
        cell.fill = fill
        cell.font = font
        cell.alignment = Alignment(horizontal='center')
    for row in sheet.rows:
        worksheet.append(list(row))
    for column, number_format in sheet.formats.items():
        letter = worksheet.cell(row=1, column=column + 1).column_letter
        for cell in worksheet[letter][1:]:
            if isinstance(cell.value, (int, float)) and not isinstance(cell.value, bool):
                cell.number_format = number_format
    for note in sheet.notes:
        worksheet.append([])
        worksheet.append([note])
    worksheet.freeze_panes = 'A2'


def build_workbook(payload: ExportPayload, *, sheet_names: dict,
                   column_labels: dict) -> bytes:
    """Return the ``.xlsx`` bytes of the session report.

    ``sheet_names`` maps the internal sheet keys of ``SHEET_KEYS`` to visible
    titles; ``column_labels`` maps every other internal key (column headers,
    strategy keys, note codes, parameter names) to visible text. Both fall back
    to the raw English key, so a missing entry degrades the label and nothing
    else.
    """
    sheet_names = sheet_names or {}
    labels = column_labels or {}

    built: dict = {}
    missing: list = []
    for key in SHEET_KEYS:
        if key == 'parameters':
            continue
        try:
            sheet = _BUILDERS[key](payload, labels)
        except Exception as exc:                      # noqa: BLE001 - RF-E4
            logger.error('export sheet failed: %s', key, exc_info=True)
            sheet = _Sheet(('note',), [[f'{_label(labels, SHEET_ERROR)}: {exc}']])
        if sheet is None:
            missing.append(key)
            continue
        built[key] = sheet

    built['parameters'] = _build_parameters(payload, labels, missing)

    workbook = Workbook()
    workbook.remove(workbook.active)
    for key in SHEET_KEYS:
        sheet = built.get(key)
        if sheet is None:
            continue
        _write_sheet(workbook, _title(sheet_names, key), sheet, labels)

    stream = BytesIO()
    workbook.save(stream)
    logger.info('workbook built with %d sheets, %d skipped', len(built), len(missing))
    return stream.getvalue()
