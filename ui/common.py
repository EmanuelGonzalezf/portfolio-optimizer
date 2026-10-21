"""Helpers shared by the seven sections of ui/sections/.

Nothing here computes a financial figure (RNF-2): the module only assembles the
fingerprint payload of design.md section 13.1b, paints the amber stale badge of
RF-P4, derives the asset statistics through ``ui/compute.py`` and builds the
Excel download from ``st.session_state``.
"""

from __future__ import annotations

import streamlit as st

from core.export import ExportPayload, build_workbook
from ui.compute import cached_asset_stats
from ui.labels import EXPORT_LABELS, SHEET_NAMES, WARNING_LABELS, label
from ui.state import (K_ALIGNED, K_BACKTEST_RESULT, K_CAPACITY_INPUTS,
                      K_CAPACITY_METRICS, K_CAPACITY_SCHEDULE, K_CAPACITY_VERDICT,
                      K_CLOUD, K_FRONTIER, K_INITIAL_POSITION, K_MISSING_POLICY,
                      K_OPTIMIZATION, K_OVERRIDES, K_PARSE_PREFS,
                      K_PURCHASE_BALANCE, K_STATS, K_THRESHOLDS, K_UPLOADED,
                      is_stale, store_fingerprint)
from ui.theme import PALETTE, badge, status_box

__all__ = [
    'EXPORT_FILE_NAME', 'EXPORT_MIME', 'SECTION_LOCAL_KEYS',
    'base_fingerprint_params', 'stale_badge', 'local_badge', 'ensure_stats',
    'export_payload', 'export_button', 'no_data_box', 'no_optimization_box',
]

EXPORT_FILE_NAME = 'portfolio_optimizer_reporte.xlsx'
EXPORT_MIME = ('application/vnd.openxmlformats-officedocument'
               '.spreadsheetml.sheet')

# Declared dependencies owned by a section rather than by the sidebar. They are
# seeded to ``None`` so every fingerprint hashes the same key set regardless of
# which sections the user has visited.
SECTION_LOCAL_KEYS = (
    'capital_s5', 'pct_fee_s5', 'fixed_fee_s5', 'cash_reserve', 'integer_units',
    'weight_tolerance', 'current_positions',
    'window', 'rebalance_rule', 'capital_s6', 'cost_rate',
    'capital_s7', 'pct_fee_s7', 'fixed_fee_s7', 'position_source',
    'cash_reserve_initial', 'holding_period_days', 'schedule', 'liquidity_policy',
    'k_sigma', 'mc_method', 'block_size', 'n_paths', 'capacity_rebalance_rule',
    'capacity_thresholds',
)

NO_DATA_MESSAGE = ('Carga primero los precios en la sección «1. Inicio y carga de '
                   'datos».')
NO_OPTIMIZATION_MESSAGE = ('Ejecuta primero la optimización en la sección '
                           '«3. Optimización de portafolios».')


def base_fingerprint_params(params: dict) -> dict:
    """Dict with every globally owned dependency key of FINGERPRINT_DEPS."""
    payload: dict = {key: None for key in SECTION_LOCAL_KEYS}
    payload.update({
        'files': st.session_state.get(K_UPLOADED),
        'overrides': st.session_state.get(K_OVERRIDES),
        'parse_prefs': st.session_state.get(K_PARSE_PREFS),
        'missing_policy': st.session_state.get(K_MISSING_POLICY),
        'period': params.get('period'),
        'trading_days': params.get('trading_days'),
        'rf': params.get('rf'),
        'allow_short': params.get('allow_short'),
        'max_weight': params.get('max_weight'),
        'short_lb': params.get('short_lb'),
        'seed': params.get('seed'),
        'n_frontier_points': params.get('n_frontier_points'),
        'n_weight_sims': params.get('n_weight_sims'),
    })
    return payload


def stale_badge(result_key: str, fingerprint_params: dict) -> None:
    """Amber badge of RF-P4 when the stored result no longer matches the inputs."""
    if is_stale(result_key, fingerprint_params):
        st.markdown(badge(label(WARNING_LABELS, 'stale_results'), PALETTE['warn']),
                    unsafe_allow_html=True)


def local_badge(is_local: bool) -> None:
    """Badge shown when a section overrides a sidebar value (section 13.1)."""
    if is_local:
        st.markdown(badge(label(WARNING_LABELS, 'local_value'), PALETTE['accent_2']),
                    unsafe_allow_html=True)


def no_data_box() -> None:
    status_box(NO_DATA_MESSAGE, 'info')


def no_optimization_box() -> None:
    status_box(NO_OPTIMIZATION_MESSAGE, 'info')


def ensure_stats(params: dict, fingerprint_params: dict) -> tuple:
    """``(aligned, stats)`` with the statistics derived on demand and cached.

    ``cached_asset_stats`` is memoised, so calling this on every rerun costs a
    dictionary lookup once the figures exist.
    """
    aligned = st.session_state.get(K_ALIGNED)
    if aligned is None:
        return None, None
    stats = cached_asset_stats(aligned.log_returns, tuple(aligned.names),
                               int(len(aligned.return_dates)),
                               int(params['trading_days']), float(params['rf']))
    st.session_state[K_STATS] = stats
    store_fingerprint('asset_stats', fingerprint_params)
    return aligned, stats


def _export_params(params: dict) -> dict:
    """The parameters sheet payload: ``{name: {'value': v, 'owner': o}}``."""
    fmt = params.get('fmt')
    payload: dict = {
        'rf_annual_effective': {'value': params.get('rf'), 'owner': 'sidebar'},
        'trading_days': {'value': params.get('trading_days'), 'owner': 'sidebar'},
        'allow_short': {'value': params.get('allow_short'), 'owner': 'sidebar'},
        'max_weight': {'value': params.get('max_weight'), 'owner': 'sidebar'},
        'short_lower_bound': {'value': params.get('short_lb'), 'owner': 'sidebar'},
        'n_weight_sims': {'value': params.get('n_weight_sims'), 'owner': 'sidebar'},
        'n_frontier_points': {'value': params.get('n_frontier_points'),
                              'owner': 'sidebar'},
        'seed': {'value': params.get('seed'), 'owner': 'sidebar'},
        'capital': {'value': params.get('capital'), 'owner': 'sidebar'},
        'pct_fee': {'value': params.get('pct_fee'), 'owner': 'sidebar'},
        'fixed_fee': {'value': params.get('fixed_fee'), 'owner': 'sidebar'},
        'period_start': {'value': params.get('period_start'), 'owner': 'sidebar'},
        'period_end': {'value': params.get('period_end'), 'owner': 'sidebar'},
        'currency_symbol': {'value': getattr(fmt, 'symbol', '$'), 'owner': 'sidebar'},
    }
    policy = st.session_state.get(K_MISSING_POLICY)
    if policy is not None:
        payload['missing_method'] = {'value': policy.method, 'owner': 'section'}
        payload['limit_days'] = {'value': policy.limit_days, 'owner': 'section'}
    for name, value in (st.session_state.get(K_CAPACITY_INPUTS) or {}).items():
        payload[str(name)] = {'value': value, 'owner': 'section'}
    thresholds = st.session_state.get(K_THRESHOLDS)
    if thresholds is not None:
        payload.update({
            'p_sufficient': {'value': thresholds.p_sufficient, 'owner': 'section'},
            'p_tight': {'value': thresholds.p_tight, 'owner': 'section'},
            'p_high_risk': {'value': thresholds.p_high_risk, 'owner': 'section'},
            'margin_min': {'value': thresholds.margin_min, 'owner': 'section'},
            'conservative_deficit_tolerance': {
                'value': thresholds.conservative_deficit_tolerance,
                'owner': 'section'},
        })
    initial = st.session_state.get(K_INITIAL_POSITION)
    if initial is not None:
        # Reserved key: feeds the PosicionInicial sheet, never printed as a
        # parameter row (core/export.py convention).
        payload['initial_position'] = initial
    return payload


def export_payload(params: dict) -> ExportPayload:
    """Assemble the nine closed fields of ``ExportPayload`` from the session."""
    cloud = st.session_state.get(K_CLOUD)
    metrics = st.session_state.get(K_CAPACITY_METRICS)
    verdict = st.session_state.get(K_CAPACITY_VERDICT)
    schedule = st.session_state.get(K_CAPACITY_SCHEDULE)
    capacity = None
    if metrics is not None and verdict is not None and schedule is not None:
        capacity = (schedule, metrics, verdict)
    return ExportPayload(
        params=_export_params(params),
        aligned=st.session_state.get(K_ALIGNED),
        stats=st.session_state.get(K_STATS),
        optimization=st.session_state.get(K_OPTIMIZATION),
        frontier=st.session_state.get(K_FRONTIER),
        weight_cloud=None if cloud is None else tuple(cloud[1:4]),
        purchase=st.session_state.get(K_PURCHASE_BALANCE),
        backtest=st.session_state.get(K_BACKTEST_RESULT),
        capacity=capacity,
    )


def export_button(params: dict, key: str) -> None:
    """Excel download. Sections that were never computed are simply omitted."""
    try:
        data = build_workbook(export_payload(params), sheet_names=SHEET_NAMES,
                              column_labels=EXPORT_LABELS)
    except Exception:                              # noqa: BLE001 - RF-E4
        status_box('No se pudo generar el informe de Excel', 'err')
        return
    st.download_button('Descargar informe en Excel', data=data,
                       file_name=EXPORT_FILE_NAME, mime=EXPORT_MIME, key=key)
