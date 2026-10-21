"""Section 1 — Inicio y carga de datos (design.md section 14, block 1).

Panels: file upload, column mapping, missing-value policy, analysis period in
read-only mode, data quality and preview. Every number comes from
``core.data_loader`` through ``ui/compute.py``; this module only paints.
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from core.data_loader import (EXPECTED_ASSETS, GAP_NOTE_DAYS, MISSING_METHODS,
                              ColumnMapping, MissingPolicy, ParsePreferences)
from core.formatting import fmt_num, fmt_pct, style_table
from data.generate_sample_data import ASSET_NAMES, generate_prices
from ui.common import base_fingerprint_params, export_button, stale_badge
from ui.compute import (build_overrides_key, build_prefs_key, cached_inspect_file,
                        cached_load_assets, cached_missing_preview)
from ui.labels import (COLUMN_LABELS, DATE_FORMAT_LABELS, PREVIEW_COLUMN_LABELS,
                       WARNING_LABELS, label)
from ui.state import (K_ALIGNED, K_MISSING_POLICY, K_OVERRIDES, K_PARSE_PREFS,
                      K_PARSE_REPORTS, K_UPLOADED, clear_results, store_fingerprint)
from ui.theme import metric_card, panel, small_note, status_box

__all__ = ['render']

SAMPLE_FILE_NAME = 'ejemplo_cinco_activos.csv'
DECIMAL_CHOICES = ('auto', 'comma', 'dot')
DAYFIRST_CHOICES = ('auto', 'day_first', 'month_first')
PARSE_REPORT_FIELDS = ('file_name', 'sheet_used', 'header_row_used', 'csv_separator',
                       'decimal_detected', 'date_format_applied',
                       'date_format_source', 'nat_dropped', 'ohlcv_collapsed')
PREVIEW_ROWS = 5


def _sample_bytes() -> bytes:
    """The five sample assets as one in-memory CSV (format A). Nothing is written."""
    prices = generate_prices()
    frame = prices.reset_index()
    frame.columns = ['Fecha', *ASSET_NAMES]
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, date_format='%Y-%m-%d')
    return buffer.getvalue().encode('utf-8')


def _render_upload_panel() -> None:
    uploaded = st.file_uploader(
        'Archivos de precios (Excel o CSV)', accept_multiple_files=True,
        type=['xlsx', 'xls', 'csv'], key='s1_uploader')
    small_note('Formato A: un archivo con una columna de fechas y cinco de precios. '
               'Formato B: un archivo por activo con fecha y precio de cierre.')
    if uploaded:
        files = [(item.name, item.getvalue()) for item in uploaded]
        if files != list(st.session_state.get(K_UPLOADED) or []):
            st.session_state[K_UPLOADED] = files

    left, right = st.columns(2)
    with left:
        if st.button('Cargar datos de ejemplo', key='s1_sample'):
            st.session_state[K_UPLOADED] = [(SAMPLE_FILE_NAME, _sample_bytes())]
            st.session_state[K_OVERRIDES] = {}
            st.session_state[K_PARSE_PREFS] = {}
    with right:
        if st.button('Limpiar todo', key='s1_clear'):
            clear_results()
            st.cache_data.clear()
            status_box('Se borraron los datos, los resultados y las cachés.', 'ok')


def _preferences(file_name: str) -> ParsePreferences:
    prefs = dict(st.session_state.get(K_PARSE_PREFS) or {})
    return prefs.get(file_name, ParsePreferences())


def _store_preferences(file_name: str, prefs: ParsePreferences) -> None:
    current = dict(st.session_state.get(K_PARSE_PREFS) or {})
    if current.get(file_name) != prefs:
        current[file_name] = prefs
        st.session_state[K_PARSE_PREFS] = current


def _store_override(file_name: str, mapping: ColumnMapping | None) -> None:
    current = dict(st.session_state.get(K_OVERRIDES) or {})
    if mapping is None:
        current.pop(file_name, None)
    else:
        current[file_name] = mapping
    st.session_state[K_OVERRIDES] = current


def _render_mapping_panel(files: list) -> None:
    for file_name, file_bytes in files:
        with st.expander(file_name):
            prefs = _preferences(file_name)
            try:
                frame, report, detected, sheets = cached_inspect_file(
                    file_name, file_bytes, prefs.to_tuple())
            except Exception as error:             # noqa: BLE001 - per-file isolation
                status_box(f'No se pudo inspeccionar «{file_name}»: {error}', 'err')
                continue

            sheet = prefs.sheet
            if len(sheets) > 1:
                options = list(sheets)
                index = options.index(sheet) if sheet in options else 0
                sheet = st.selectbox('Hoja', options, index=index,
                                     key=f's1_sheet_{file_name}')
            header_row = int(st.number_input(
                'Fila de cabecera (0 = primera fila)', min_value=0, max_value=200,
                value=int(report.header_row_used), step=1,
                key=f's1_header_{file_name}'))
            decimal = st.selectbox(
                'Separador decimal', DECIMAL_CHOICES,
                index=DECIMAL_CHOICES.index(prefs.decimal)
                if prefs.decimal in DECIMAL_CHOICES else 0,
                format_func=lambda key: label(DATE_FORMAT_LABELS, key)
                if key != 'auto' else 'Automático',
                key=f's1_decimal_{file_name}')
            dayfirst = st.selectbox(
                'Formato de fecha', DAYFIRST_CHOICES,
                index=DAYFIRST_CHOICES.index(prefs.dayfirst)
                if prefs.dayfirst in DAYFIRST_CHOICES else 0,
                format_func=lambda key: label(DATE_FORMAT_LABELS, key)
                if key != 'auto' else 'Automático',
                key=f's1_dayfirst_{file_name}')
            _store_preferences(file_name, ParsePreferences(
                sheet=sheet, header_row=header_row, decimal=str(decimal),
                dayfirst=str(dayfirst)))

            columns = [str(column) for column in frame.columns]
            date_index = (columns.index(detected.date_column)
                          if detected.date_column in columns else 0)
            date_column = st.selectbox('Columna de fechas', columns, index=date_index,
                                       key=f's1_date_{file_name}')
            default_prices = [column for column in detected.price_columns
                              if column in columns]
            price_columns = st.multiselect(
                'Columnas de precios', [c for c in columns if c != date_column],
                default=[c for c in default_prices if c != date_column],
                key=f's1_prices_{file_name}')

            badges = [f'Formato detectado: {detected.detected_format}']
            if report.csv_separator:
                badges.append(f'Separador CSV: «{report.csv_separator}»')
            badges.append('Separador decimal: '
                          + label(DATE_FORMAT_LABELS, report.decimal_detected))
            if report.ohlcv_collapsed:
                badges.append('OHLCV detectado')
            small_note(' · '.join(badges))

            manual = (date_column != detected.date_column
                      or tuple(price_columns) != tuple(detected.price_columns))
            if manual and price_columns:
                _store_override(file_name, ColumnMapping(
                    file_name=file_name, date_column=str(date_column),
                    price_columns=tuple(str(c) for c in price_columns),
                    detected_format=detected.detected_format,
                    adjusted_flags=dict(detected.adjusted_flags)))
            else:
                _store_override(file_name, None)
            if manual and not price_columns:
                status_box('Selecciona al menos una columna de precios para aplicar '
                           'el mapeo manual.', 'info')


def _render_missing_panel(files: list, params: dict) -> MissingPolicy:
    stored = st.session_state.get(K_MISSING_POLICY) or MissingPolicy()
    left, right = st.columns([0.9, 1.1])
    with left:
        method = st.radio(
            'Política de valores faltantes', MISSING_METHODS,
            index=MISSING_METHODS.index(stored.method)
            if stored.method in MISSING_METHODS else 0,
            format_func=lambda key: label(PREVIEW_COLUMN_LABELS, key),
            key='s1_missing_method')
        limit_days = int(st.number_input(
            'Límite de días de relleno', min_value=0, max_value=60,
            value=int(stored.limit_days), step=1, key='s1_limit_days'))
    policy = MissingPolicy(method=str(method), limit_days=limit_days)
    st.session_state[K_MISSING_POLICY] = policy

    with right:
        try:
            preview = cached_missing_preview(
                tuple(files), build_overrides_key(st.session_state.get(K_OVERRIDES)),
                build_prefs_key(st.session_state.get(K_PARSE_PREFS)),
                params.get('period_start'), params.get('period_end'), limit_days)
        except Exception as error:                 # noqa: BLE001
            status_box(f'No se pudo calcular la comparación de políticas: {error}',
                       'info')
        else:
            table = preview.copy()
            table.index = [label(PREVIEW_COLUMN_LABELS, key) for key in table.index]
            table = table.rename(columns=lambda key: label(PREVIEW_COLUMN_LABELS, key))
            st.dataframe(table, use_container_width=True)
            small_note('Política activa: ' + label(PREVIEW_COLUMN_LABELS, method))
    return policy


def _render_quality_panel(aligned, params: dict) -> None:
    quality = aligned.quality
    cards = st.columns(4)
    with cards[0]:
        metric_card('Activos alineados', str(len(aligned.names)))
    with cards[1]:
        metric_card('Fechas comunes', fmt_num(quality.common_dates, 0, params['fmt']),
                    'Observaciones conservadas tras la alineación')
    with cards[2]:
        metric_card('Rango', f'{aligned.dates[0]:%d/%m/%Y} – '
                             f'{aligned.dates[-1]:%d/%m/%Y}')
    with cards[3]:
        total_cells = max(len(aligned.dates) * len(aligned.names), 1)
        filled = sum(int(value) for value in (quality.filled_cells or {}).values())
        metric_card('Celdas rellenadas', fmt_pct(filled / total_cells, 2,
                                                 params['fmt']),
                    f'{filled} celdas con política '
                    f'{label(PREVIEW_COLUMN_LABELS, quality.missing_policy)}')

    per_asset = quality.per_asset.copy()
    per_asset = per_asset.rename(columns=lambda key: label(COLUMN_LABELS, key))
    st.dataframe(per_asset, use_container_width=True, hide_index=True)

    reports = st.session_state.get(K_PARSE_REPORTS) or ()
    if reports:
        rows = [{field: getattr(report, field) for field in PARSE_REPORT_FIELDS}
                for report in reports]
        frame = pd.DataFrame(rows).rename(columns=lambda key: label(COLUMN_LABELS, key))
        st.dataframe(frame, use_container_width=True, hide_index=True)

    codes = list(quality.warning_codes or ())
    for report in reports:
        for code in report.warning_codes or ():
            if code not in codes:
                codes.append(code)
    if quality.missing_policy == 'ffill' and 'ffill_applied' not in codes:
        status_box('Se arrastró el último precio disponible: los retornos de los días '
                   'rellenados son cero y la volatilidad queda subestimada.', 'info')
    if len(aligned.names) != EXPECTED_ASSETS and 'asset_count_not_five' not in codes:
        codes.append('asset_count_not_five')
    for code in codes:
        kind = 'err' if code == 'zero_variance_asset' else 'info'
        status_box(label(WARNING_LABELS, code), kind)
    if quality.max_return_span_days > GAP_NOTE_DAYS:
        small_note(f'Hueco máximo entre observaciones alineadas: '
                   f'{quality.max_return_span_days} días naturales')


def _render_preview_panel(aligned) -> None:
    frame = pd.DataFrame(aligned.prices, index=aligned.dates,
                         columns=list(aligned.names))
    head = frame.head(PREVIEW_ROWS)
    tail = frame.tail(PREVIEW_ROWS)
    shown = pd.concat([head, tail]) if len(frame) > 2 * PREVIEW_ROWS else frame
    shown = shown.copy()
    shown.index = [f'{stamp:%d/%m/%Y}' for stamp in shown.index]
    st.dataframe(style_table(shown, num_cols=list(aligned.names)),
                 use_container_width=True)
    small_note(f'Primeras y últimas {PREVIEW_ROWS} filas de las '
               f'{len(frame)} observaciones alineadas')


def render(params: dict) -> None:
    fingerprint_params = base_fingerprint_params(params)

    with panel('Carga de archivos'):
        _render_upload_panel()

    files = list(st.session_state.get(K_UPLOADED) or [])
    if not files:
        with panel('Estado'):
            status_box('Todavía no hay archivos. Sube los precios de los cinco '
                       'activos o carga los datos de ejemplo.', 'info')
        return

    with panel('Mapeo de columnas'):
        _render_mapping_panel(files)

    with panel('Valores faltantes'):
        policy = _render_missing_panel(files, params)

    with panel('Periodo de análisis'):
        start, end = params.get('period_start'), params.get('period_end')
        if start and end:
            status_box(f'Periodo activo: {start} a {end}', 'ok')
        else:
            status_box('Periodo activo: toda la serie disponible', 'info')
        small_note('Se configura en la barra lateral')

    fingerprint_params['missing_policy'] = policy
    fingerprint_params['overrides'] = st.session_state.get(K_OVERRIDES)
    fingerprint_params['parse_prefs'] = st.session_state.get(K_PARSE_PREFS)

    with panel('Procesar y alinear'):
        stale_badge('aligned_data', fingerprint_params)
        if st.button('Procesar y alinear datos', type='primary', key='s1_load'):
            try:
                aligned, reports = cached_load_assets(
                    tuple(files),
                    build_overrides_key(st.session_state.get(K_OVERRIDES)),
                    build_prefs_key(st.session_state.get(K_PARSE_PREFS)),
                    policy.to_tuple(), params.get('period_start'),
                    params.get('period_end'))
            except Exception as error:             # noqa: BLE001
                status_box(f'No se pudieron alinear los activos: {error}', 'err')
            else:
                st.session_state[K_ALIGNED] = aligned
                st.session_state[K_PARSE_REPORTS] = reports
                store_fingerprint('aligned_data', fingerprint_params)
                status_box(f'Universo alineado: {len(aligned.names)} activos y '
                           f'{len(aligned.dates)} fechas comunes.', 'ok')

    aligned = st.session_state.get(K_ALIGNED)
    if aligned is None:
        return

    with panel('Calidad de los datos'):
        _render_quality_panel(aligned, params)

    with panel('Vista previa'):
        _render_preview_panel(aligned)

    with panel('Informe'):
        export_button(params, 's1_export')
