"""File reading, column detection, cleaning and alignment (design.md section 6).

Every value this module returns is English and code-shaped: column names,
``ParseReport`` codes and ``PortfolioError`` codes. ``ui/labels.py`` owns the
visible text.

Hard rules kept here:

* a bad price produces a counted drop, never a zero return (RF-D8);
* filling acts on prices, never on returns;
* the decimal separator is detected per column before cleaning, so ``12,34``
  never becomes ``1234``;
* duplicated dates keep the last occurrence.
"""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass
from pathlib import PurePath

import numpy as np
import pandas as pd

from core.types import (AlignedData, AlignmentError, AssetSeries, DataLoadError,
                        DataQualityReport)

__all__ = [
    'ColumnMapping',
    'MissingPolicy',
    'ParsePreferences',
    'ParseReport',
    'MISSING_METHODS',
    'list_sheets',
    'read_tabular',
    'detect_columns',
    'parse_series',
    'deduplicate_names',
    'align_assets',
    'missing_policy_preview',
    'load_assets',
]

MISSING_METHODS = ('drop', 'ffill', 'interpolate')
PREVIEW_COLUMNS = ('resulting_obs', 'filled_cells', 'lost_dates')
PER_ASSET_COLUMNS = ('asset', 'rows_raw', 'invalid', 'duplicates', 'first', 'last',
                     'max_gap_days', 'lost_in_alignment')

MIN_ASSETS = 2
MIN_COMMON_DATES = 10
EXPECTED_ASSETS = 5
HEADER_SCAN_ROWS = 10
HEADER_NON_NUMERIC_SHARE = 0.60
PARSEABLE_SHARE = 0.60
DECIMAL_MAIN_THRESHOLD = 0.60
DECIMAL_MARGIN_THRESHOLD = 0.10
EXCEL_EPOCH = '1899-12-30'
EXCEL_SERIAL_RANGE = (1000.0, 80000.0)
MULTI_DAY_SPAN_SHARE = 0.02
GAP_NOTE_DAYS = 10

_CSV_SUFFIXES = {'.csv', '.txt', '.tsv'}
_OHLCV_NAMES = {'open', 'high', 'low', 'close', 'adj close', 'adj_close', 'volume',
                'apertura', 'maximo', 'minimo', 'cierre', 'volumen'}
# Patterns consumed by pandas ``.str`` accessors stay plain strings on purpose.
_CURRENCY_NOISE = r'[$%\u20ac\s\u00a0]'
_ES_NUMBER = r'^-?\d{1,3}(\.\d{3})*(,\d+)?$|^-?\d+,\d+$'
_US_NUMBER = r'^-?\d{1,3}(,\d{3})*(\.\d+)?$|^-?\d+\.\d+$'
_ISO_DATE = r'^\d{4}-\d{2}-\d{2}'
_DATE_TOKENS = re.compile(r'[/\-.]')
_ADJUSTED = re.compile(r'adj|ajustad', re.I)


# --------------------------------------------------------------------------- #
# Public contract
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ColumnMapping:
    """Which column holds the dates and which ones hold prices, per file."""

    file_name: str
    date_column: str
    price_columns: tuple
    detected_format: str
    adjusted_flags: dict


@dataclass(frozen=True)
class MissingPolicy:
    """How holes in the wide price frame are treated before the intersection."""

    method: str = 'drop'
    limit_days: int = 5

    def to_tuple(self) -> tuple:
        return (str(self.method), int(self.limit_days))

    @classmethod
    def from_tuple(cls, data: tuple) -> 'MissingPolicy':
        method, limit_days = data
        return cls(method=str(method), limit_days=int(limit_days))


@dataclass(frozen=True)
class ParsePreferences:
    """Per-file overrides of the automatic reading decisions."""

    sheet: object = 0
    header_row: object = None
    decimal: str = 'auto'
    dayfirst: str = 'auto'

    def to_tuple(self) -> tuple:
        return (self.sheet, self.header_row, str(self.decimal), str(self.dayfirst))

    @classmethod
    def from_tuple(cls, data: tuple) -> 'ParsePreferences':
        sheet, header_row, decimal, dayfirst = data
        return cls(sheet=sheet, header_row=header_row,
                   decimal=str(decimal), dayfirst=str(dayfirst))


@dataclass(frozen=True)
class ParseReport:
    """What the quality panel shows for one file.

    Accumulated across ``read_tabular`` -> ``detect_columns`` -> ``parse_series``:
    the later stages fill the fields they are the first to know about through
    ``_record``, the single place allowed to write into a frozen report.
    """

    file_name: str
    sheet_used: str = ''
    header_row_used: int = 0
    csv_separator: object = None
    decimal_detected: str = 'dot'
    date_format_applied: str = 'iso'
    date_format_source: str = 'detected'
    nat_dropped: int = 0
    ohlcv_collapsed: bool = False
    warning_codes: tuple = ()


def _record(report: ParseReport, **changes) -> ParseReport:
    """Write ``changes`` into ``report`` in place and return it."""
    for key, value in changes.items():
        object.__setattr__(report, key, value)
    return report


def _add_warning(report: ParseReport, code: str) -> ParseReport:
    if code not in report.warning_codes:
        _record(report, warning_codes=report.warning_codes + (code,))
    return report


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _normalize(name: object) -> str:
    """Lowercase, accent-free, underscore-free version of a column name."""
    text = str(name).strip().lower().replace('_', ' ')
    decomposed = unicodedata.normalize('NFKD', text)
    return ''.join(ch for ch in decomposed if not unicodedata.combining(ch)).strip()


def _stem(file_name: str) -> str:
    return PurePath(str(file_name)).stem


def _is_csv(file_name: str) -> bool:
    return PurePath(str(file_name)).suffix.lower() in _CSV_SUFFIXES


def _clean_text(values: pd.Series) -> pd.Series:
    """Strip currency symbols, percent marks and every flavour of space."""
    return (values.astype(str)
                  .str.strip()
                  .str.replace(_CURRENCY_NOISE, '', regex=True))


def _non_empty_mask(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    return values.notna() & (text != '') & (text.str.lower() != 'nan')


def _numeric_fraction(column: pd.Series) -> float:
    """Share of non-empty cells that can be read as a number under either locale."""
    if pd.api.types.is_numeric_dtype(column):
        return 1.0
    if pd.api.types.is_datetime64_any_dtype(column):
        return 0.0
    mask = _non_empty_mask(column)
    if not mask.any():
        return 0.0
    cleaned = _clean_text(column[mask])
    best = 0.0
    for decimal in ('dot', 'comma'):
        parsed = pd.to_numeric(_apply_decimal(cleaned, decimal), errors='coerce')
        best = max(best, float(parsed.notna().mean()))
    return best


def _date_fraction(column: pd.Series) -> float:
    """Share of non-empty cells that can be read as a date."""
    if pd.api.types.is_datetime64_any_dtype(column):
        return 1.0
    mask = _non_empty_mask(column)
    if not mask.any():
        return 0.0
    values = column[mask]
    if pd.api.types.is_numeric_dtype(column):
        numbers = pd.to_numeric(values, errors='coerce')
        return float(numbers.between(*EXCEL_SERIAL_RANGE).mean())
    text = values.astype(str).str.strip()
    serial = pd.to_numeric(text, errors='coerce')
    serial_share = float(serial.between(*EXCEL_SERIAL_RANGE).mean())
    if serial_share >= PARSEABLE_SHARE:
        return serial_share
    parsed = pd.to_datetime(text, errors='coerce', dayfirst=True, format='mixed')
    return float(parsed.notna().mean())


def _apply_decimal(cleaned: pd.Series, decimal: str) -> pd.Series:
    if decimal == 'comma':
        return cleaned.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    return cleaned.str.replace(',', '', regex=False)


def _detect_decimal(column: pd.Series) -> str:
    """Decide the decimal mark of a text price column with the 0,60 / 0,10 thresholds."""
    mask = _non_empty_mask(column)
    if not mask.any():
        return 'dot'
    cleaned = _clean_text(column[mask])
    share_comma = float(cleaned.str.match(_ES_NUMBER).mean())
    share_dot = float(cleaned.str.match(_US_NUMBER).mean())
    if share_comma >= DECIMAL_MAIN_THRESHOLD and share_comma - share_dot >= DECIMAL_MARGIN_THRESHOLD:
        return 'comma'
    if share_dot >= DECIMAL_MAIN_THRESHOLD and share_dot - share_comma >= DECIMAL_MARGIN_THRESHOLD:
        return 'dot'
    return 'ambiguous'


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def list_sheets(file_bytes: bytes, file_name: str) -> list:
    """Sheet names of a workbook; empty list for a delimited text file."""
    if _is_csv(file_name):
        return []
    return list(_open_workbook(file_bytes, file_name).sheet_names)


def _open_workbook(file_bytes: bytes, file_name: str) -> pd.ExcelFile:
    try:
        return pd.ExcelFile(io.BytesIO(file_bytes))
    except ImportError as exc:
        raise DataLoadError('excel_engine_missing', file=str(file_name),
                            detail=str(exc)) from exc
    except Exception as exc:  # unreadable workbook
        raise DataLoadError('workbook_unreadable', file=str(file_name),
                            detail=str(exc)) from exc


def _decode(file_bytes: bytes) -> str:
    for encoding in ('utf-8-sig', 'utf-8', 'latin-1'):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode('latin-1', errors='replace')


def _sniff_separator(text: str) -> str:
    """Pick the delimiter with the most consistent count over the first lines."""
    lines = [line for line in text.splitlines() if line.strip()][:20]
    if not lines:
        return ','
    best, best_score = ',', -1.0
    for candidate in (';', ',', '\t', '|'):
        counts = [line.count(candidate) for line in lines]
        if min(counts) < 1:
            continue
        score = min(counts) - (max(counts) - min(counts))
        if score > best_score:
            best, best_score = candidate, score
    return best


def _frame_with_header(raw: pd.DataFrame, header_row: int) -> pd.DataFrame:
    names = [str(v).strip() if _scalar_non_empty(v) else f'column_{i}'
             for i, v in enumerate(raw.iloc[header_row].tolist())]
    body = raw.iloc[header_row + 1:].reset_index(drop=True)
    body.columns = _unique_names(names)
    body = body.dropna(axis=1, how='all')
    return body.infer_objects()


def _unique_names(names: list) -> list:
    seen: dict = {}
    result = []
    for name in names:
        if name in seen:
            seen[name] += 1
            result.append(f'{name}_{seen[name]}')
        else:
            seen[name] = 1
            result.append(name)
    return result


def _scalar_non_empty(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    return str(value).strip() not in ('', 'nan', 'NaN', 'None')


def _has_parseable_pair(frame: pd.DataFrame) -> bool:
    """Condition (b): at least one date column and a different numeric column.

    Plus one tie-break the two literal conditions need to behave as intended: the
    row right below the candidate must already be data. Without it a title line
    sitting above the table satisfies both conditions, because the real header
    row is then just one unparseable row inside a long parseable column.
    """
    if frame.shape[1] < 2 or frame.empty:
        return False
    date_cols = [c for c in frame.columns if _date_fraction(frame[c]) >= PARSEABLE_SHARE]
    if not date_cols:
        return False
    date_col = date_cols[0]
    numeric_cols = [c for c in frame.columns
                    if c != date_col and _numeric_fraction(frame[c]) >= PARSEABLE_SHARE]
    if not numeric_cols:
        return False
    first_date = frame[date_col].iloc[:1]
    first_numeric = frame[numeric_cols[0]].iloc[:1]
    return (_date_fraction(first_date) >= PARSEABLE_SHARE
            and _numeric_fraction(first_numeric) >= PARSEABLE_SHARE)


def _detect_header_row(raw: pd.DataFrame) -> int | None:
    """First of the 10 leading rows meeting both numeric conditions of section 6.2."""
    limit = min(HEADER_SCAN_ROWS, len(raw))
    for index in range(limit):
        row = raw.iloc[index]
        non_empty = [v for v in row.tolist() if _scalar_non_empty(v)]
        if not non_empty:
            continue
        as_series = pd.Series(non_empty, dtype=object)
        numeric = pd.to_numeric(_clean_text(as_series), errors='coerce')
        non_numeric_share = 1.0 - float(numeric.notna().mean())
        if non_numeric_share < HEADER_NON_NUMERIC_SHARE:
            continue
        if _has_parseable_pair(_frame_with_header(raw, index)):
            return index
    return None


def read_tabular(file_bytes: bytes, file_name: str,
                 prefs: ParsePreferences = ParsePreferences()
                 ) -> tuple[pd.DataFrame, ParseReport]:
    """Read one file into a frame, reporting sheet, header row and separator."""
    report = ParseReport(file_name=str(file_name))
    if _is_csv(file_name):
        text = _decode(file_bytes)
        separator = _sniff_separator(text)
        try:
            raw = pd.read_csv(io.StringIO(text), sep=separator, engine='python',
                              header=None, dtype=object, keep_default_na=False)
        except Exception as exc:
            raise DataLoadError('csv_unreadable', file=str(file_name),
                                detail=str(exc)) from exc
        _record(report, csv_separator=separator, sheet_used='')
    else:
        workbook = _open_workbook(file_bytes, file_name)
        sheet = prefs.sheet if prefs.sheet is not None else 0
        names = list(workbook.sheet_names)
        if isinstance(sheet, int):
            if sheet >= len(names):
                raise DataLoadError('sheet_not_found', file=str(file_name), sheet=sheet)
            sheet_name = names[sheet]
        else:
            if str(sheet) not in names:
                raise DataLoadError('sheet_not_found', file=str(file_name), sheet=str(sheet))
            sheet_name = str(sheet)
        raw = pd.read_excel(workbook, sheet_name=sheet_name, header=None)
        _record(report, sheet_used=sheet_name)

    raw = raw.dropna(axis=0, how='all').reset_index(drop=True)
    if raw.empty:
        raise DataLoadError('file_has_no_rows', file=str(file_name))

    if prefs.header_row is not None:
        header_row = int(prefs.header_row)
        if header_row >= len(raw):
            raise DataLoadError('header_row_out_of_range', file=str(file_name),
                                header_row=header_row, rows=int(len(raw)))
    else:
        detected = _detect_header_row(raw)
        if detected is None:
            header_row = 0
            _add_warning(report, 'header_row_not_detected')
        else:
            header_row = detected

    frame = _frame_with_header(raw, header_row)
    frame = frame.dropna(axis=0, how='all').reset_index(drop=True)
    if frame.empty:
        raise DataLoadError('file_has_no_data_rows', file=str(file_name))
    _record(report, header_row_used=int(header_row))
    return frame, report


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def _pick_date_column(columns: list) -> str:
    for column in columns:
        if re.fullmatch(r'(date|fecha)', _normalize(column)):
            return column
    for column in columns:
        if re.search(r'date|day|time|fecha|dia|periodo', _normalize(column)):
            return column
    return columns[0]


def _pick_price_column(columns: list) -> str:
    for column in columns:
        name = _normalize(column)
        if re.fullmatch(r'adj[\s_.]*close', name) or re.match(r'(precio|cierre)[\s_]*ajustad', name):
            return column
    for column in columns:
        if re.fullmatch(r'(close|cierre)', _normalize(column)):
            return column
    for column in columns:
        if re.search(r'price|close|adj|precio|cierre', _normalize(column)):
            return column
    return columns[0]


def _detect_columns_impl(df: pd.DataFrame, file_name: str) -> tuple[ColumnMapping, bool]:
    columns = [str(c) for c in df.columns]
    if not columns:
        raise DataLoadError('file_has_no_columns', file=str(file_name))
    frame = df.copy()
    frame.columns = columns

    date_column = _pick_date_column(columns)
    others = [c for c in columns if c != date_column]
    if not others:
        raise DataLoadError('no_price_column_found', file=str(file_name))

    # OHLCV rule, evaluated BEFORE deciding the format.
    ohlcv_collapsed = set(_normalize(c) for c in others).issubset(_OHLCV_NAMES)
    if ohlcv_collapsed:
        price_columns = (_pick_price_column(others),)
        detected_format = 'B'
    else:
        numeric = [c for c in others if _numeric_fraction(frame[c]) >= PARSEABLE_SHARE]
        if not numeric:
            raise DataLoadError('no_price_column_found', file=str(file_name))
        if len(numeric) > 1:
            price_columns = tuple(numeric)
            detected_format = 'A'
        else:
            price_columns = (numeric[0],)
            detected_format = 'B'

    flags = {c: bool(_ADJUSTED.search(c)) for c in price_columns}
    mapping = ColumnMapping(file_name=str(file_name), date_column=date_column,
                            price_columns=price_columns,
                            detected_format=detected_format, adjusted_flags=flags)
    return mapping, ohlcv_collapsed


def detect_columns(df: pd.DataFrame, file_name: str) -> ColumnMapping:
    """Detect the date column, the price columns and the file format (A or B)."""
    return _detect_columns_impl(df, file_name)[0]


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _decide_dayfirst(values: pd.Series, prefs: ParsePreferences) -> tuple[bool, str, str]:
    """Return ``(dayfirst, applied_code, source)`` for a text date column."""
    if prefs.dayfirst == 'day_first':
        return True, 'day_first', 'chosen'
    if prefs.dayfirst == 'month_first':
        return False, 'month_first', 'chosen'

    first_tokens, second_tokens = [], []
    for raw in values.astype(str):
        parts = [p for p in _DATE_TOKENS.split(raw.strip()) if p != '']
        if len(parts) < 2:
            continue
        if parts[0].isdigit():
            first_tokens.append(int(parts[0]))
        if parts[1].isdigit():
            second_tokens.append(int(parts[1]))
    if any(13 <= t <= 31 for t in first_tokens):
        return True, 'day_first', 'detected'
    if any(13 <= t <= 31 for t in second_tokens):
        return False, 'month_first', 'detected'
    return True, 'day_first', 'chosen'


def _parse_dates(column: pd.Series, prefs: ParsePreferences,
                 report: ParseReport) -> pd.Series:
    """Parse a date column: Excel serial, ISO, or tokenised day/month decision."""
    if pd.api.types.is_datetime64_any_dtype(column):
        _record(report, date_format_applied='iso', date_format_source='detected')
        return pd.to_datetime(column, errors='coerce')

    mask = _non_empty_mask(column)
    values = column.where(mask)

    if pd.api.types.is_numeric_dtype(column):
        numbers = pd.to_numeric(values, errors='coerce')
        _record(report, date_format_applied='excel_serial', date_format_source='detected')
        return pd.to_datetime(numbers, unit='D', origin=EXCEL_EPOCH, errors='coerce')

    text = values.astype(str).str.strip()
    numeric_candidate = pd.to_numeric(text.where(mask), errors='coerce')
    if mask.any() and float(numeric_candidate[mask].notna().mean()) >= PARSEABLE_SHARE:
        _record(report, date_format_applied='excel_serial', date_format_source='detected')
        return pd.to_datetime(numeric_candidate, unit='D', origin=EXCEL_EPOCH, errors='coerce')

    present = text[mask]
    if len(present) and float(present.str.match(_ISO_DATE).mean()) >= PARSEABLE_SHARE:
        _record(report, date_format_applied='iso', date_format_source='detected')
        return pd.to_datetime(text, errors='coerce', format='mixed')

    dayfirst, applied, source = _decide_dayfirst(present, prefs)
    _record(report, date_format_applied=applied, date_format_source=source)
    return pd.to_datetime(text, errors='coerce', dayfirst=dayfirst, format='mixed')


def _parse_prices(column: pd.Series, prefs: ParsePreferences,
                  report: ParseReport) -> tuple[pd.Series, str]:
    """Parse a price column, deciding the decimal mark before cleaning."""
    if pd.api.types.is_numeric_dtype(column):
        return pd.to_numeric(column, errors='coerce'), 'dot'

    if prefs.decimal in ('comma', 'dot'):
        decimal = prefs.decimal
    else:
        detected = _detect_decimal(column)
        if detected == 'ambiguous':
            decimal = 'dot'
            _add_warning(report, 'decimal_separator_ambiguous')
        else:
            decimal = detected

    mask = _non_empty_mask(column)
    cleaned = _clean_text(column.where(mask))
    numbers = pd.to_numeric(_apply_decimal(cleaned, decimal), errors='coerce')
    return numbers.where(mask), decimal


def parse_series(df: pd.DataFrame, mapping: ColumnMapping, file_name: str,
                 prefs: ParsePreferences, report: ParseReport) -> list:
    """Turn one raw frame into cleaned ``AssetSeries``, counting every drop."""
    frame = df.copy()
    frame.columns = [str(c) for c in frame.columns]
    rows_raw = int(len(frame))

    dates = _parse_dates(frame[mapping.date_column], prefs, report)
    if dates.notna().sum() == 0:
        raise DataLoadError('date_column_unparseable', file=str(file_name),
                            column=str(mapping.date_column))

    decimals_seen = []
    series_list = []
    nat_dropped = 0
    for price_column in mapping.price_columns:
        prices, decimal = _parse_prices(frame[price_column], prefs, report)
        decimals_seen.append(decimal)

        asset = pd.DataFrame({'date': dates, 'price': prices})
        nat_dropped = max(nat_dropped, int(asset['date'].isna().sum()))
        asset = asset[asset['date'].notna()]

        finite = np.isfinite(asset['price'].to_numpy(dtype=float, na_value=np.nan))
        valid = finite & (asset['price'].to_numpy(dtype=float, na_value=np.nan) > 0.0)
        dropped_invalid = int((~valid).sum())
        asset = asset[valid]

        asset = asset.sort_values('date', kind='stable')
        before = len(asset)
        asset = asset.drop_duplicates(subset='date', keep='last')
        dropped_duplicates = int(before - len(asset))

        if asset.empty:
            raise DataLoadError('asset_has_no_valid_rows', file=str(file_name),
                                column=str(price_column))

        name = (price_column if mapping.detected_format == 'A' else _stem(file_name))
        series_list.append(AssetSeries(
            name=str(name),
            source_file=str(file_name),
            price_column=str(price_column),
            is_adjusted=bool(mapping.adjusted_flags.get(price_column, False)),
            dates=pd.DatetimeIndex(asset['date'].to_numpy()),
            prices=asset['price'].to_numpy(dtype=float),
            rows_raw=rows_raw,
            dropped_invalid=dropped_invalid,
            dropped_duplicates=dropped_duplicates,
        ))

    _record(report, nat_dropped=int(nat_dropped),
            decimal_detected=decimals_seen[0] if decimals_seen else 'dot')
    return series_list


def deduplicate_names(series: list) -> list:
    """Disambiguate repeated asset names with ``_2``, ``_3``, ... suffixes."""
    counts: dict = {}
    result = []
    for item in series:
        counts[item.name] = counts.get(item.name, 0) + 1
        if counts[item.name] == 1:
            result.append(item)
            continue
        renamed = AssetSeries(name=f'{item.name}_{counts[item.name]}',
                              source_file=item.source_file,
                              price_column=item.price_column,
                              is_adjusted=item.is_adjusted,
                              dates=item.dates, prices=item.prices,
                              rows_raw=item.rows_raw,
                              dropped_invalid=item.dropped_invalid,
                              dropped_duplicates=item.dropped_duplicates)
        result.append(renamed)
    return result


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def _wide_frame(series: list, start=None, end=None) -> pd.DataFrame:
    columns = {}
    for item in series:
        columns[item.name] = pd.Series(np.asarray(item.prices, dtype=float),
                                       index=pd.DatetimeIndex(item.dates))
    wide = pd.DataFrame(columns).sort_index()
    if start is not None:
        wide = wide[wide.index >= pd.Timestamp(start)]
    if end is not None:
        wide = wide[wide.index <= pd.Timestamp(end)]
    return wide


def _apply_policy(wide: pd.DataFrame, method: str, limit_days: int) -> tuple[pd.DataFrame, dict]:
    """Fill prices (never returns) and count how many cells each asset gained."""
    if method not in MISSING_METHODS:
        raise DataLoadError('unknown_missing_method', method=str(method))
    before = wide.isna()
    if method == 'ffill':
        filled = wide.ffill(limit=int(limit_days))
    elif method == 'interpolate':
        filled = wide.interpolate(method='time', limit=int(limit_days), limit_area='inside')
    else:
        filled = wide
    gained = (before & filled.notna())
    return filled, {str(c): int(gained[c].sum()) for c in wide.columns}


def _span_days(dates: pd.DatetimeIndex) -> np.ndarray:
    """Business days spanned by each consecutive pair of aligned dates."""
    if len(dates) < 2:
        return np.zeros(0, dtype=int)
    spans = []
    for previous, current in zip(dates[:-1], dates[1:]):
        spans.append(max(len(pd.bdate_range(previous, current)) - 1, 1))
    return np.asarray(spans, dtype=int)


def missing_policy_preview(series: list, start=None, end=None,
                           limit_days: int = 5) -> pd.DataFrame:
    """One row per policy ('drop','ffill','interpolate'). Columns are EXACTLY and only
    ('resulting_obs', 'filled_cells', 'lost_dates') -- English, since this is a core/
    return value. ui/labels.py::PREVIEW_COLUMN_LABELS renames them for display.
    Computed once after cleaning, before the user picks a policy (RF-D7).
    """
    wide = _wide_frame(series, start, end)
    union = int(len(wide.index))
    rows = {}
    for method in MISSING_METHODS:
        filled, gained = _apply_policy(wide, method, limit_days)
        resulting = int(len(filled.dropna()))
        rows[method] = {'resulting_obs': resulting,
                        'filled_cells': int(sum(gained.values())),
                        'lost_dates': int(union - resulting)}
    return pd.DataFrame.from_dict(rows, orient='index')[list(PREVIEW_COLUMNS)]


def align_assets(series: list, policy: MissingPolicy = MissingPolicy(),
                 start=None, end=None) -> AlignedData:
    """Run the seven-stage pipeline of section 6.3 and report its effect."""
    if len(series) < MIN_ASSETS:
        raise AlignmentError('alignment_too_few_assets', n=int(len(series)))

    wide = _wide_frame(series, start, end)
    filled, gained = _apply_policy(wide, policy.method, policy.limit_days)
    aligned = filled.dropna()
    common_dates = int(len(aligned.index))
    if common_dates < MIN_COMMON_DATES:
        raise AlignmentError('alignment_too_few_common_dates', n=common_dates)

    names = tuple(str(c) for c in aligned.columns)
    dates = pd.DatetimeIndex(aligned.index)
    prices = aligned.to_numpy(dtype=float)
    log_returns = np.diff(np.log(prices), axis=0)
    return_dates = dates[1:]

    spans = _span_days(dates)
    multi_day = int(np.sum(spans > 1))
    max_span = int(spans.max()) if spans.size else 0

    warning_codes = []
    if len(names) != EXPECTED_ASSETS:
        warning_codes.append('asset_count_not_five')
    if log_returns.shape[0] > 1:
        variances = log_returns.var(axis=0, ddof=1)
        if np.any(variances <= 0.0):
            warning_codes.append('zero_variance_asset')
    if multi_day > MULTI_DAY_SPAN_SHARE * max(log_returns.shape[0], 1):
        warning_codes.append('returns_span_multiple_days')

    non_adjusted = tuple(item.name for item in series if not item.is_adjusted)
    if len(non_adjusted) == len(series) and series:
        warning_codes.append('prices_not_adjusted')

    rows = []
    aligned_set = set(dates)
    for item in series:
        own = pd.DatetimeIndex(item.dates)
        if start is not None:
            own = own[own >= pd.Timestamp(start)]
        if end is not None:
            own = own[own <= pd.Timestamp(end)]
        gaps = own.to_series().diff().dt.days.dropna()
        rows.append({
            'asset': item.name,
            'rows_raw': int(item.rows_raw),
            'invalid': int(item.dropped_invalid),
            'duplicates': int(item.dropped_duplicates),
            'first': own[0] if len(own) else pd.NaT,
            'last': own[-1] if len(own) else pd.NaT,
            'max_gap_days': int(gaps.max()) if len(gaps) else 0,
            'lost_in_alignment': int(sum(1 for d in own if d not in aligned_set)),
        })
    per_asset = pd.DataFrame(rows, columns=list(PER_ASSET_COLUMNS))
    if (per_asset['max_gap_days'] > GAP_NOTE_DAYS).any():
        warning_codes.append('large_calendar_gap')

    quality = DataQualityReport(per_asset=per_asset, common_dates=common_dates,
                                missing_policy=str(policy.method),
                                filled_cells=gained,
                                non_adjusted_assets=non_adjusted,
                                returns_spanning_multiple_days=multi_day,
                                max_return_span_days=max_span,
                                warning_codes=tuple(warning_codes))
    return AlignedData(names=names, dates=dates, prices=prices,
                       return_dates=return_dates, log_returns=log_returns,
                       quality=quality)


def parse_files(files: list, overrides: dict | None = None,
                prefs: dict | None = None) -> tuple:
    """Read and clean every file without aligning yet.

    Returns ``(series, reports)`` with the deduplicated ``AssetSeries`` list and
    one ``ParseReport`` per file. ``load_assets`` is this plus ``align_assets``;
    the split exists because ``missing_policy_preview`` needs the cleaned series
    before the user picks a policy (RF-D7).
    """
    overrides = overrides or {}
    prefs = prefs or {}
    collected: list = []
    reports: list = []
    for file_name, file_bytes in files:
        preferences = prefs.get(file_name, ParsePreferences())
        frame, report = read_tabular(file_bytes, file_name, preferences)
        detected, ohlcv_collapsed = _detect_columns_impl(frame, file_name)
        mapping = overrides.get(file_name, detected)
        _record(report, ohlcv_collapsed=bool(ohlcv_collapsed))
        collected.extend(parse_series(frame, mapping, file_name, preferences, report))
        reports.append(report)
    return deduplicate_names(collected), tuple(reports)


def load_assets(files: list, overrides: dict | None = None,
                prefs: dict | None = None,
                policy: MissingPolicy = MissingPolicy(),
                start=None, end=None) -> tuple:
    """Read every file, clean it and align the universe.

    ``files`` is a list of ``(file_name, file_bytes)``. ``overrides`` and
    ``prefs`` are keyed by file name and win over the automatic detection.
    """
    collected, reports = parse_files(files, overrides, prefs)
    aligned = align_assets(collected, policy, start, end)
    return aligned, reports
