"""Tests for core/data_loader.py and the synthetic sample data.

Covers CA-33, CA-34, CA-35 and CA-37 (design.md section 21). No test reads a
file from the user's workspace: every input is built in memory or under tmp_path.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from core.data_loader import (MissingPolicy, ParsePreferences, align_assets, deduplicate_names,
                              detect_columns, list_sheets, load_assets,
                              missing_policy_preview, parse_series, read_tabular)
from core.types import AlignmentError, AssetSeries, DataLoadError
from data.generate_sample_data import ASSET_NAMES, generate_prices, write_sample_files

DATE_FORMAT_CODES = {'day_first', 'month_first', 'iso', 'excel_serial'}


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _csv(lines) -> bytes:
    return ('\n'.join(lines) + '\n').encode('utf-8')


def _business_days(n: int, start: str = '2024-01-02') -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _simple_csv(n: int = 14, start_price: float = 100.0) -> bytes:
    dates = _business_days(n)
    lines = ['Fecha,Adj Close']
    for index, date in enumerate(dates):
        lines.append(f'{date.strftime("%Y-%m-%d")},{start_price + index:.2f}')
    return _csv(lines)


def _series(name: str, dates, prices, **kwargs) -> AssetSeries:
    return AssetSeries(name=name, source_file=f'{name}.csv',
                       price_column=kwargs.get('price_column', 'Adj Close'),
                       is_adjusted=kwargs.get('is_adjusted', True),
                       dates=pd.DatetimeIndex(dates),
                       prices=np.asarray(prices, dtype=float),
                       rows_raw=kwargs.get('rows_raw', len(prices)),
                       dropped_invalid=kwargs.get('dropped_invalid', 0),
                       dropped_duplicates=kwargs.get('dropped_duplicates', 0))


def _parse_single(file_bytes: bytes, file_name: str,
                  prefs: ParsePreferences = ParsePreferences()):
    frame, report = read_tabular(file_bytes, file_name, prefs)
    mapping = detect_columns(frame, file_name)
    series = parse_series(frame, mapping, file_name, prefs, report)
    return series, mapping, report


# --------------------------------------------------------------------------- #
# Format A and Format B end to end (CA-33)
# --------------------------------------------------------------------------- #
def test_format_a_from_a_single_workbook(tmp_path):
    written = write_sample_files(tmp_path)
    payload = written['excel'].read_bytes()
    aligned, reports = load_assets([(written['excel'].name, payload)])

    assert aligned.names == ASSET_NAMES
    assert len(aligned.dates) == 1260
    assert aligned.log_returns.shape == (1259, 5)
    assert reports[0].sheet_used == 'Precios'
    assert reports[0].ohlcv_collapsed is False
    assert detect_columns(read_tabular(payload, 'a.xlsx')[0], 'a.xlsx').detected_format == 'A'


def test_format_b_from_five_files(tmp_path):
    written = write_sample_files(tmp_path)
    files = [(path.name, path.read_bytes()) for path in written['csv']]
    aligned, reports = load_assets(files)

    assert aligned.names == ASSET_NAMES
    assert len(aligned.dates) == 1260
    assert all(report.csv_separator == ',' for report in reports)
    assert all(report.decimal_detected == 'dot' for report in reports)
    assert aligned.quality.non_adjusted_assets == ()


def test_both_routes_agree_on_prices(tmp_path):
    written = write_sample_files(tmp_path)
    from_excel, _ = load_assets([(written['excel'].name, written['excel'].read_bytes())])
    from_csv, _ = load_assets([(path.name, path.read_bytes()) for path in written['csv']])
    assert np.allclose(from_excel.prices, from_csv.prices, atol=1e-8)


# --------------------------------------------------------------------------- #
# es-ES parsing (CA-33)
# --------------------------------------------------------------------------- #
def test_spanish_csv_keeps_magnitudes():
    dates = _business_days(12)
    values = ['12,34', '1.234,56', '0,99', '2.500,00', '45,10', '7,05',
              '1.000,25', '19,99', '3.141,59', '8,88', '123,45', '2,50']
    lines = ['Fecha;Cierre ajustado']
    for date, value in zip(dates, values):
        lines.append(f'{date.strftime("%d/%m/%Y")};{value}')
    series, mapping, report = _parse_single(_csv(lines), 'ACTIVO_ES.csv')

    assert report.csv_separator == ';'
    assert report.decimal_detected == 'comma'
    assert mapping.detected_format == 'B'
    assert mapping.price_columns == ('Cierre ajustado',)
    prices = series[0].prices
    assert prices[0] == pytest.approx(12.34, abs=1e-12)
    assert prices[1] == pytest.approx(1234.56, abs=1e-12)
    assert not np.any(np.isclose(prices, 1234.0))
    assert series[0].is_adjusted is True


def test_spanish_sample_variant_loads(tmp_path):
    written = write_sample_files(tmp_path)
    files = [(path.name, path.read_bytes()) for path in written['csv_es']]
    aligned, reports = load_assets(files)
    assert len(aligned.names) == 5
    assert all(report.decimal_detected == 'comma' for report in reports)
    assert all(report.date_format_applied == 'day_first' for report in reports)


def test_day_first_is_detected_when_a_day_exceeds_twelve():
    dates = ['13/03/2024', '14/03/2024', '15/03/2024', '18/03/2024', '19/03/2024',
             '20/03/2024', '21/03/2024', '22/03/2024', '25/03/2024', '26/03/2024',
             '27/03/2024', '28/03/2024']
    lines = ['Fecha,Adj Close'] + [f'{d},{100 + i}.50' for i, d in enumerate(dates)]
    series, _, report = _parse_single(_csv(lines), 'X.csv')
    assert report.date_format_applied == 'day_first'
    assert report.date_format_source == 'detected'
    assert series[0].dates[0] == pd.Timestamp('2024-03-13')


def test_ambiguous_dates_fall_back_to_the_spanish_default():
    dates = ['01/03/2024', '04/03/2024', '05/03/2024', '06/03/2024', '07/03/2024',
             '08/03/2024', '11/03/2024', '12/03/2024', '01/04/2024', '02/04/2024',
             '03/04/2024', '04/04/2024']
    lines = ['Fecha,Adj Close'] + [f'{d},{100 + i}.50' for i, d in enumerate(dates)]
    series, _, report = _parse_single(_csv(lines), 'X.csv')
    assert report.date_format_applied == 'day_first'
    assert report.date_format_source == 'chosen'
    assert series[0].dates[0] == pd.Timestamp('2024-03-01')

    forced = ParsePreferences(dayfirst='month_first')
    other, _, other_report = _parse_single(_csv(lines), 'X.csv', forced)
    assert other_report.date_format_applied == 'month_first'
    assert other_report.date_format_source == 'chosen'
    assert other[0].dates[0] == pd.Timestamp('2024-01-03')


def test_date_format_codes_are_english():
    series, _, report = _parse_single(_simple_csv(), 'X.csv')
    assert report.date_format_applied in DATE_FORMAT_CODES
    assert 'aaaa' not in report.date_format_applied
    assert report.date_format_source in {'detected', 'chosen'}
    assert len(series) == 1


def test_excel_serial_dates_use_the_1899_epoch():
    lines = ['Fecha,Adj Close'] + [f'{45000 + i},{100 + i}.25' for i in range(12)]
    series, _, report = _parse_single(_csv(lines), 'X.csv')
    assert report.date_format_applied == 'excel_serial'
    expected = pd.Timestamp('1899-12-30') + pd.Timedelta(days=45000)
    assert series[0].dates[0] == expected


# --------------------------------------------------------------------------- #
# Sheets, headers and OHLCV
# --------------------------------------------------------------------------- #
def _workbook_bytes(sheets: dict) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    return buffer.getvalue()


def test_two_sheet_workbook_reads_the_first_one():
    dates = _business_days(12)
    first = pd.DataFrame({'Fecha': dates, 'ACTIVO_X': np.linspace(10, 20, 12),
                          'ACTIVO_Y': np.linspace(30, 40, 12)})
    second = pd.DataFrame({'Fecha': dates, 'ACTIVO_Z': np.linspace(50, 60, 12)})
    payload = _workbook_bytes({'Precios': first, 'Otra': second})

    assert list_sheets(payload, 'libro.xlsx') == ['Precios', 'Otra']
    frame, report = read_tabular(payload, 'libro.xlsx')
    assert report.sheet_used == 'Precios'
    assert detect_columns(frame, 'libro.xlsx').price_columns == ('ACTIVO_X', 'ACTIVO_Y')

    other, other_report = read_tabular(payload, 'libro.xlsx', ParsePreferences(sheet='Otra'))
    assert other_report.sheet_used == 'Otra'
    assert detect_columns(other, 'libro.xlsx').price_columns == ('ACTIVO_Z',)


def test_header_three_rows_down_is_autodetected():
    dates = _business_days(12)
    lines = ['Reporte interno,', 'Generado el 2024,', ',', 'Fecha,Adj Close']
    for index, date in enumerate(dates):
        lines.append(f'{date.strftime("%Y-%m-%d")},{100 + index}.75')
    frame, report = read_tabular(_csv(lines), 'X.csv')
    assert report.header_row_used == 3
    assert list(frame.columns) == ['Fecha', 'Adj Close']
    assert 'header_row_not_detected' not in report.warning_codes


def _ohlcv_csv(n: int = 12, base: float = 50.0) -> bytes:
    dates = _business_days(n)
    lines = ['Date,Open,High,Low,Close,Adj Close,Volume']
    for index, date in enumerate(dates):
        price = base + index
        lines.append(f'{date.strftime("%Y-%m-%d")},{price:.2f},{price + 1:.2f},'
                     f'{price - 1:.2f},{price + 0.5:.2f},{price + 0.4:.2f},1000000')
    return _csv(lines)


def test_ohlcv_file_is_collapsed_to_format_b_with_adjusted_close():
    frame, report = read_tabular(_ohlcv_csv(), 'AAPL.csv')
    mapping = detect_columns(frame, 'AAPL.csv')
    assert mapping.detected_format == 'B'
    assert mapping.price_columns == ('Adj Close',)
    assert mapping.adjusted_flags['Adj Close'] is True

    aligned, reports = load_assets([('AAPL.csv', _ohlcv_csv()),
                                    ('MSFT.csv', _ohlcv_csv(base=300.0))])
    assert aligned.names == ('AAPL', 'MSFT')
    assert all(report.ohlcv_collapsed for report in reports)
    assert 'asset_count_not_five' in aligned.quality.warning_codes


# --------------------------------------------------------------------------- #
# Cleaning counters (CA-34)
# --------------------------------------------------------------------------- #
def test_duplicate_dates_keep_the_last_value_and_are_counted():
    dates = _business_days(12)
    lines = ['Fecha,Adj Close']
    for index, date in enumerate(dates):
        lines.append(f'{date.strftime("%Y-%m-%d")},{100 + index}.00')
    lines.append(f'{dates[3].strftime("%Y-%m-%d")},999.00')
    series, _, _ = _parse_single(_csv(lines), 'X.csv')
    asset = series[0]
    assert asset.dropped_duplicates == 1
    assert len(asset.dates) == 12
    assert asset.prices[3] == pytest.approx(999.0, abs=1e-12)


def test_invalid_prices_are_dropped_and_counted():
    dates = _business_days(14)
    bad = {3: 'abc', 5: '-10.00', 7: '0.00', 9: ''}
    lines = ['Fecha,Adj Close']
    for index, date in enumerate(dates):
        value = bad.get(index, f'{100 + index}.00')
        lines.append(f'{date.strftime("%Y-%m-%d")},{value}')
    series, _, _ = _parse_single(_csv(lines), 'X.csv')
    asset = series[0]
    assert asset.dropped_invalid == 4
    assert len(asset.dates) == 10
    assert (asset.prices > 0).all()


def test_an_invalid_price_never_becomes_a_zero_return():
    dates = _business_days(14)
    good = _series('GOOD', dates, np.linspace(100.0, 113.0, 14))
    prices = list(np.linspace(50.0, 63.0, 14))
    broken_dates = [d for i, d in enumerate(dates) if i != 6]
    broken_prices = [p for i, p in enumerate(prices) if i != 6]
    broken = _series('BROKEN', broken_dates, broken_prices, dropped_invalid=1)

    aligned = align_assets([good, broken], MissingPolicy('drop'))
    assert len(aligned.dates) == 13
    assert aligned.log_returns.shape == (12, 2)
    assert not np.any(aligned.log_returns == 0.0)
    assert aligned.quality.returns_spanning_multiple_days == 1
    assert aligned.quality.max_return_span_days == 2


def test_names_are_preserved_and_disambiguated():
    dates = _business_days(12)
    first = _series('ACTIVO', dates, np.linspace(10.0, 21.0, 12))
    second = _series('ACTIVO', dates, np.linspace(30.0, 41.0, 12))
    renamed = deduplicate_names([first, second])
    assert [item.name for item in renamed] == ['ACTIVO', 'ACTIVO_2']
    aligned = align_assets(renamed, MissingPolicy('drop'))
    assert aligned.names == ('ACTIVO', 'ACTIVO_2')


def test_dates_are_sorted_chronologically():
    dates = _business_days(12)
    shuffled = list(dates[::-1])
    lines = ['Fecha,Adj Close']
    for index, date in enumerate(shuffled):
        lines.append(f'{date.strftime("%Y-%m-%d")},{100 + index}.00')
    series, _, _ = _parse_single(_csv(lines), 'X.csv')
    assert series[0].dates.is_monotonic_increasing


# --------------------------------------------------------------------------- #
# Alignment and missing policies (CA-34, CA-35)
# --------------------------------------------------------------------------- #
def test_partial_overlap_reports_the_common_dates():
    dates = _business_days(20)
    first = _series('A', dates[:15], np.linspace(10.0, 24.0, 15))
    second = _series('B', dates[5:], np.linspace(50.0, 64.0, 15))
    aligned = align_assets([first, second], MissingPolicy('drop'))
    assert aligned.quality.common_dates == 10
    assert len(aligned.dates) == 10
    lost = dict(zip(aligned.quality.per_asset['asset'],
                    aligned.quality.per_asset['lost_in_alignment']))
    assert lost == {'A': 5, 'B': 5}


def test_forward_fill_counts_filled_cells():
    dates = _business_days(16)
    first = _series('A', dates, np.linspace(10.0, 25.0, 16))
    keep = [index for index in range(16) if index not in (4, 9)]
    second = _series('B', dates[keep], np.linspace(50.0, 65.0, 16)[keep])

    dropped = align_assets([first, second], MissingPolicy('drop'))
    filled = align_assets([first, second], MissingPolicy('ffill', 5))
    assert dropped.quality.common_dates == 14
    assert filled.quality.common_dates == 16
    assert filled.quality.filled_cells['B'] == 2
    assert filled.quality.filled_cells['A'] == 0
    assert filled.quality.missing_policy == 'ffill'


def test_missing_policy_preview_columns_are_english():
    dates = _business_days(16)
    first = _series('A', dates, np.linspace(10.0, 25.0, 16))
    keep = [index for index in range(16) if index not in (4, 9)]
    second = _series('B', dates[keep], np.linspace(50.0, 65.0, 16)[keep])

    preview = missing_policy_preview([first, second], None, None, 5)
    assert list(preview.columns) == ['resulting_obs', 'filled_cells', 'lost_dates']
    assert list(preview.index) == ['drop', 'ffill', 'interpolate']
    assert preview.loc['drop', 'resulting_obs'] == 14
    assert preview.loc['ffill', 'resulting_obs'] == 16
    assert preview.loc['interpolate', 'resulting_obs'] == 16
    assert preview.loc['drop', 'resulting_obs'] <= preview.loc['ffill', 'resulting_obs']
    assert preview.loc['drop', 'filled_cells'] == 0
    assert preview.loc['drop', 'lost_dates'] == 2


def test_period_selection_trims_the_history():
    dates = _business_days(30)
    first = _series('A', dates, np.linspace(10.0, 39.0, 30))
    second = _series('B', dates, np.linspace(50.0, 79.0, 30))
    aligned = align_assets([first, second], MissingPolicy('drop'),
                           start=dates[5], end=dates[19])
    assert len(aligned.dates) == 15
    assert aligned.dates[0] == dates[5]
    assert aligned.dates[-1] == dates[19]


def test_multi_day_return_span_is_counted():
    dates = _business_days(20)
    first = _series('A', dates, np.linspace(10.0, 29.0, 20))
    keep = [index for index in range(20) if index not in (5, 10)]
    second = _series('B', dates[keep], np.linspace(50.0, 69.0, 20)[keep])

    aligned = align_assets([first, second], MissingPolicy('drop'))
    assert len(aligned.dates) == 18
    assert aligned.log_returns.shape[0] == 17
    assert aligned.quality.returns_spanning_multiple_days == 2
    assert aligned.quality.max_return_span_days == 2
    assert 'returns_span_multiple_days' in aligned.quality.warning_codes

    clean = align_assets([first, _series('B', dates, np.linspace(50.0, 69.0, 20))],
                         MissingPolicy('drop'))
    assert clean.quality.returns_spanning_multiple_days == 0
    assert clean.quality.max_return_span_days == 1
    assert 'returns_span_multiple_days' not in clean.quality.warning_codes


def test_non_adjusted_universe_is_flagged_once():
    dates = _business_days(12)
    first = _series('A', dates, np.linspace(10.0, 21.0, 12),
                    is_adjusted=False, price_column='Close')
    second = _series('B', dates, np.linspace(50.0, 61.0, 12),
                     is_adjusted=False, price_column='Close')
    aligned = align_assets([first, second], MissingPolicy('drop'))
    assert aligned.quality.non_adjusted_assets == ('A', 'B')
    assert aligned.quality.warning_codes.count('prices_not_adjusted') == 1


# --------------------------------------------------------------------------- #
# Validations
# --------------------------------------------------------------------------- #
def test_alignment_rejects_a_single_asset():
    dates = _business_days(12)
    with pytest.raises(AlignmentError) as error:
        align_assets([_series('A', dates, np.linspace(10.0, 21.0, 12))], MissingPolicy('drop'))
    assert error.value.code == 'alignment_too_few_assets'


def test_alignment_rejects_too_few_common_dates():
    dates = _business_days(20)
    first = _series('A', dates[:9], np.linspace(10.0, 18.0, 9))
    second = _series('B', dates[:9], np.linspace(50.0, 58.0, 9))
    with pytest.raises(AlignmentError) as error:
        align_assets([first, second], MissingPolicy('drop'))
    assert error.value.code == 'alignment_too_few_common_dates'
    assert error.value.ctx['n'] == 9


def test_zero_variance_asset_is_flagged():
    dates = _business_days(12)
    first = _series('A', dates, np.linspace(10.0, 21.0, 12))
    flat = _series('B', dates, np.full(12, 50.0))
    aligned = align_assets([first, flat], MissingPolicy('drop'))
    assert 'zero_variance_asset' in aligned.quality.warning_codes


def test_unreadable_date_column_raises_a_coded_error():
    lines = ['Fecha,Adj Close'] + [f'texto_{i},{100 + i}.00' for i in range(12)]
    with pytest.raises(DataLoadError) as error:
        _parse_single(_csv(lines), 'X.csv')
    assert error.value.code in {'date_column_unparseable', 'no_price_column_found'}


# --------------------------------------------------------------------------- #
# Sample data (CA-37)
# --------------------------------------------------------------------------- #
def test_sample_data_is_valid(tmp_path):
    first = generate_prices(42)
    second = generate_prices(42)
    assert first.shape == (1260, 5)
    assert (first.to_numpy() > 0).all()
    assert first.isna().sum().sum() == 0
    assert np.array_equal(first.to_numpy(), second.to_numpy())
    assert tuple(first.columns) == ASSET_NAMES
    assert not np.array_equal(generate_prices(7).to_numpy(), first.to_numpy())

    written = write_sample_files(tmp_path)
    by_excel, _ = load_assets([(written['excel'].name, written['excel'].read_bytes())])
    by_csv, _ = load_assets([(path.name, path.read_bytes()) for path in written['csv']])
    assert len(by_excel.names) == 5
    assert len(by_csv.names) == 5
    assert by_excel.quality.common_dates == 1260
    assert by_csv.quality.common_dates == 1260
