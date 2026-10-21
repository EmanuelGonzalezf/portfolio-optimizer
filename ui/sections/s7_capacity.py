"""Section 7 — Capacidad de pago de cuotas (design.md section 14, block 7).

The widest section of the application: its own page with the nine blocks of the
design, the three deterministic scenarios, the Monte Carlo run, the indicators
of RF-C17, the six visualisations of RF-C19 and the automatic reading of
RF-C20/RF-C21. Every figure is produced by ``core/`` and only rendered here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.backtest import REBALANCE_RULES
from core.cashflow import (FREQUENCY_OFFSETS, LiquidityPolicy,
                           SCHEDULE_FRAME_COLUMNS, build_calendar,
                           build_initial_position, build_schedule,
                           map_payment_indices, schedule_from_frame)
from core.formatting import fmt_currency, fmt_pct, sign_class, style_table
from core.scenarios import (BLOCK_SIZE_RANGE, CapacityThresholds, NO_SCHEDULED_OBLIGATIONS,
                            PATH_METHODS, classify_payment_capacity)
from ui.charts import (PLOTLY_CONFIG, cash_timeline, coverage_by_installment,
                       coverage_summary, final_wealth_histogram,
                       scenario_comparison, wealth_vs_installments)
from ui.common import (base_fingerprint_params, ensure_stats, export_button,
                       local_badge, no_data_box, no_optimization_box, stale_badge)
from ui.compute import cached_capacity
from ui.labels import (CATEGORY_LABELS, COLUMN_LABELS, FREQUENCY_LABELS, NOTE_LABELS,
                       REBALANCE_LABELS, SCHEDULE_COLUMN_LABELS, STRATEGY_LABELS,
                       label)
from ui.narrative import build_narrative
from ui.state import (K_CAPACITY_INPUTS, K_CAPACITY_METRICS, K_CAPACITY_SCHEDULE,
                      K_CAPACITY_VERDICT, K_INITIAL_POSITION, K_OPTIMIZATION,
                      K_PURCHASE_BALANCE, K_SCHEDULE_IS_CUSTOM, K_THRESHOLDS,
                      effective_value, store_fingerprint)
from ui.theme import PALETTE, metric_card, panel, section_header, small_note, status_box

__all__ = ['render']

STRATEGY_ORDER = ('min_variance', 'max_sharpe', 'equal_weight')
POSITION_SOURCES = ('from_capital', 'from_balance')
NO_REBALANCE = 'none'

CATEGORY_COLORS = {
    'sufficient': PALETTE['accent'],
    'tight': PALETTE['accent_3'],
    'high_risk': PALETTE['warn'],
    'insufficient': PALETTE['err'],
}
CATEGORY_KIND = {'sufficient': 'ok', 'tight': 'info', 'high_risk': 'info',
                 'insufficient': 'err'}

LEDGER_VIEW_COLUMNS = ('due', 'wealth_before', 'cash_before', 'external_income',
                       'gross_sold', 'sale_costs', 'rebalance_costs', 'paid',
                       'deficit', 'cash_after', 'wealth_after', 'covered',
                       'note_code')

GUARANTEE_WARNING = (
    'Las cuotas son obligaciones contractuales incluso si la inversión pierde valor. '
    'Esta herramienta no recomienda endeudarse ni invertir por la rentabilidad '
    'esperada, y un escenario central favorable no convierte la operación en segura.')

MC_ASSUMPTIONS = """
- **Independencia y distribución idéntica**: las trayectorias gaussianas suponen
  retornos i.i.d. multinormales; los retornos reales presentan colas gruesas y
  agrupamiento de volatilidad.
- **Estacionariedad**: los momentos históricos del periodo seleccionado se usan como
  el escenario central del horizonte completo.
- **Bootstrap por bloques**: remuestrea bloques de la serie histórica sin recentrar la
  media, de modo que hereda la tendencia del periodo cargado.
- **Días festivos**: el calendario usa días hábiles de pandas y no modela festivos
  locales ni cierres de mercado concretos.
- **La probabilidad simulada no es una probabilidad real**: es la proporción de
  trayectorias que cubren todas las cuotas bajo estos supuestos.
"""


def _schedule_controls(valuation_date) -> tuple:
    """Return ``(schedule, is_custom)`` from the regular builder or the editor."""
    custom = st.toggle('Editar el calendario cuota a cuota', value=False,
                       key='s7_custom_schedule')
    frequencies = list(FREQUENCY_OFFSETS)
    if not custom:
        left, right = st.columns(2)
        with left:
            first_date = st.date_input(
                'Fecha de la primera cuota',
                value=(pd.Timestamp(valuation_date) + pd.DateOffset(months=1)).date(),
                key='s7_first_date')
            frequency = st.selectbox(
                'Frecuencia', frequencies,
                index=frequencies.index('monthly'),
                format_func=lambda key: label(FREQUENCY_LABELS, key),
                key='s7_frequency')
            n_installments = int(st.number_input(
                'Número de cuotas', min_value=1, max_value=600, value=24, step=1,
                key='s7_n_installments'))
        with right:
            amount = float(st.number_input(
                'Importe de la primera cuota', min_value=0.0, value=2_000.0,
                step=100.0, format='%.2f', key='s7_amount'))
            growth_rate = float(st.number_input(
                'Crecimiento por cuota (fracción)', min_value=-0.5, max_value=0.5,
                value=0.0, step=0.005, format='%.4f', key='s7_growth'))
            external_income = float(st.number_input(
                'Ingreso externo por cuota', min_value=0.0, value=0.0, step=100.0,
                format='%.2f', key='s7_external'))
        schedule = build_schedule(first_date, str(frequency), n_installments, amount,
                                  growth_rate, external_income)
        return schedule, False

    stored = st.session_state.get(K_CAPACITY_SCHEDULE)
    if stored is not None and len(stored.dates):
        base = pd.DataFrame({
            'date': pd.DatetimeIndex(stored.dates),
            'amount': np.asarray(stored.amounts, dtype=float),
            'external_income': np.asarray(stored.external_income, dtype=float)})
    else:
        dates = pd.date_range(pd.Timestamp(valuation_date) + pd.DateOffset(months=1),
                              periods=6, freq='MS')
        base = pd.DataFrame({'date': dates, 'amount': [2_000.0] * 6,
                             'external_income': [0.0] * 6})
    frame = base.rename(columns=lambda key: label(SCHEDULE_COLUMN_LABELS, key))
    edited = st.data_editor(frame, use_container_width=True, hide_index=True,
                            num_rows='dynamic', key='s7_schedule_editor')
    renamed = edited.rename(columns={label(SCHEDULE_COLUMN_LABELS, key): key
                                     for key in SCHEDULE_FRAME_COLUMNS})
    return schedule_from_frame(renamed), True


def _threshold_controls() -> CapacityThresholds:
    stored = st.session_state.get(K_THRESHOLDS) or CapacityThresholds()
    with st.expander('Umbrales de clasificación'):
        columns = st.columns(5)
        with columns[0]:
            p_sufficient = float(st.number_input(
                label(COLUMN_LABELS, 'p_sufficient'), min_value=0.0, max_value=1.0,
                value=float(stored.p_sufficient), step=0.01, format='%.2f',
                key='s7_p_sufficient'))
        with columns[1]:
            p_tight = float(st.number_input(
                label(COLUMN_LABELS, 'p_tight'), min_value=0.0, max_value=1.0,
                value=float(stored.p_tight), step=0.01, format='%.2f',
                key='s7_p_tight'))
        with columns[2]:
            p_high_risk = float(st.number_input(
                label(COLUMN_LABELS, 'p_high_risk'), min_value=0.0, max_value=1.0,
                value=float(stored.p_high_risk), step=0.01, format='%.2f',
                key='s7_p_high_risk'))
        with columns[3]:
            margin_min = float(st.number_input(
                label(COLUMN_LABELS, 'margin_min'), min_value=0.0, max_value=5.0,
                value=float(stored.margin_min), step=0.01, format='%.2f',
                key='s7_margin_min'))
        with columns[4]:
            tolerance = float(st.number_input(
                label(COLUMN_LABELS, 'conservative_deficit_tolerance'),
                min_value=0.0, max_value=1.0,
                value=float(stored.conservative_deficit_tolerance), step=0.01,
                format='%.2f', key='s7_deficit_tolerance'))
        if not p_sufficient >= p_tight >= p_high_risk:
            status_box('Los umbrales deben cumplir suficiente ≥ ajustada ≥ riesgo '
                       'elevado. Se mantienen los valores anteriores.', 'err')
            return stored
    thresholds = CapacityThresholds(p_sufficient=p_sufficient, p_tight=p_tight,
                                    p_high_risk=p_high_risk, margin_min=margin_min,
                                    conservative_deficit_tolerance=tolerance)
    st.session_state[K_THRESHOLDS] = thresholds
    return thresholds


def _render_verdict(metrics, verdict, thresholds, fmt) -> None:
    if NO_SCHEDULED_OBLIGATIONS in (metrics.note_codes or ()):
        status_box(label(NOTE_LABELS, NO_SCHEDULED_OBLIGATIONS), 'info')
        return
    category = str(verdict.category)
    color = CATEGORY_COLORS.get(category, PALETTE['text'])
    st.markdown(
        f'<div style="font-size:26px;font-weight:700;color:{color};'
        f'line-height:1.25">{label(CATEGORY_LABELS, category)}</div>',
        unsafe_allow_html=True)
    status_box(f'Categoría: {label(CATEGORY_LABELS, category)} '
               f'(regla aplicada n.º {verdict.rule_fired})',
               CATEGORY_KIND.get(category, 'info'))
    st.markdown(build_narrative(metrics, verdict, fmt))

    rows = []
    for criterion in verdict.criteria:
        rows.append({
            'item': label(COLUMN_LABELS, criterion.name),
            'value': criterion.value,
            'threshold': criterion.threshold,
            'comparator': criterion.comparator,
            'passed': 'Sí' if criterion.passed else 'No',
        })
    table = pd.DataFrame(rows).rename(columns=lambda key: label(COLUMN_LABELS, key))
    st.dataframe(table, use_container_width=True, hide_index=True)
    small_note(
        f'Umbrales numéricos: cobertura suficiente ≥ {fmt_pct(thresholds.p_sufficient, 0, fmt)} · '
        f'ajustada ≥ {fmt_pct(thresholds.p_tight, 0, fmt)} · '
        f'riesgo elevado ≥ {fmt_pct(thresholds.p_high_risk, 0, fmt)} · '
        f'margen mínimo de capital final ≥ {fmt_pct(thresholds.margin_min, 0, fmt)} · '
        f'tolerancia de déficit conservador ≤ '
        f'{fmt_pct(thresholds.conservative_deficit_tolerance, 0, fmt)}')


def _first(array, default=float('nan')) -> float:
    values = np.asarray(array, dtype=float).ravel()
    return float(values[0]) if values.size else float(default)


def _render_indicators(metrics, horizon_end, fmt) -> None:
    central = (metrics.deterministic or {}).get('central')
    cards = st.columns(4)
    with cards[0]:
        metric_card('Capital inicial', fmt_currency(metrics.initial_capital, fmt),
                    f'{int(metrics.n_installments)} cuotas programadas')
    with cards[1]:
        metric_card('Total de cuotas programadas',
                    fmt_currency(metrics.total_scheduled, fmt))
    with cards[2]:
        metric_card('Trayectorias que cubren todas las cuotas',
                    fmt_pct(metrics.p_cover_all, 1, fmt),
                    f'{int(np.asarray(metrics.mc_covered_all).size)} trayectorias '
                    f'simuladas', variant='sharpe')
    with cards[3]:
        metric_card('Capital adicional necesario',
                    fmt_currency(metrics.additional_capital_required, fmt),
                    'Para cubrir el escenario conservador', variant='vol')

    if central is None:
        return
    cards = st.columns(4)
    with cards[0]:
        metric_card('Total pagado (escenario central)',
                    fmt_currency(_first(central.total_paid), fmt))
    with cards[1]:
        defaults = int(_first(central.n_defaults, 0))
        metric_card('Cuotas incumplidas (central)', str(defaults),
                    'Escenario determinista')
    with cards[2]:
        metric_card('Capital mínimo (en fechas de pago)',
                    fmt_currency(_first(central.min_wealth), fmt))
    with cards[3]:
        metric_card(f'Capital final (al {pd.Timestamp(horizon_end):%d/%m/%Y})',
                    fmt_currency(_first(central.final_wealth), fmt),
                    f'Margen sobre las cuotas: '
                    f'{fmt_pct(metrics.margin_ratio, 1, fmt)}',
                    sign=sign_class(metrics.margin_ratio))

    cards = st.columns(3)
    with cards[0]:
        metric_card('Déficit máximo de caja (central)',
                    fmt_currency(_first(central.max_deficit), fmt))
    with cards[1]:
        index = int(_first(central.first_default_idx, -1))
        metric_card('Primer incumplimiento (central)',
                    'Sin incumplimientos' if index < 0 else f'Cuota {index + 1}')
    with cards[2]:
        percentiles = metrics.final_wealth_percentiles or {}
        metric_card('Mediana del capital final simulado',
                    fmt_currency(percentiles.get('p50', float('nan')), fmt),
                    f'p5 {fmt_currency(percentiles.get("p5", float("nan")), fmt)} · '
                    f'p95 {fmt_currency(percentiles.get("p95", float("nan")), fmt)}')


def _render_reading_levels(metrics, fmt) -> None:
    st.markdown('**Nivel 1 — Capacidad determinista.** '
                f'El escenario central '
                f'{"cubre" if metrics.central_covers_all else "no cubre"} todas las '
                f'cuotas y el conservador '
                f'{"también" if metrics.conservative_covers_all else "no"}. '
                'Son trayectorias únicas construidas con los momentos históricos del '
                'periodo seleccionado.')
    st.markdown('*Nivel 2 — Capacidad simulada.* '
                f'El {fmt_pct(metrics.p_cover_all, 1, fmt)} de las '
                f'{int(np.asarray(metrics.mc_covered_all).size)} trayectorias cubre '
                'todas las cuotas. Es una proporción dentro del modelo, no una '
                'probabilidad real de mercado.')
    small_note('Nivel 3 — Garantía: la simulación no puede darla. Los supuestos '
               'estadísticos pueden fallar y las cuotas siguen siendo exigibles.')


def _render_ledger(metrics, fmt) -> None:
    ledger = metrics.central_ledger
    if ledger is None or len(ledger) == 0:
        status_box('El escenario central no produjo registro de cuotas.', 'info')
        return
    view = ledger.copy()
    columns = [key for key in LEDGER_VIEW_COLUMNS if key in view.columns]
    view = view[columns]
    if 'covered' in view.columns:
        view['covered'] = ['Cubierta' if bool(value) else 'Incumplida'
                           for value in view['covered']]
    if 'note_code' in view.columns:
        view['note_code'] = [label(NOTE_LABELS, value) if value else ''
                             for value in view['note_code']]
    view.insert(0, 'date', [f'{stamp:%d/%m/%Y}' for stamp in view.index])
    view = view.reset_index(drop=True)
    view = view.rename(columns=lambda key: label(COLUMN_LABELS, key))
    money = [label(COLUMN_LABELS, key) for key in columns
             if key not in ('covered', 'note_code')]
    st.dataframe(style_table(view, money_cols=money,
                             signed_cols=[label(COLUMN_LABELS, 'deficit')], fmt=fmt),
                 use_container_width=True, hide_index=True)
    small_note('Los costos de recomposición son la columna derivada '
               '(previos + posteriores al pago).')


def render(params: dict) -> None:
    section_header('Capacidad de pago de cuotas')
    fingerprint_params = base_fingerprint_params(params)
    aligned, stats = ensure_stats(params, fingerprint_params)
    if aligned is None:
        with panel('Capacidad de pago'):
            no_data_box()
        status_box(GUARANTEE_WARNING, 'info')
        return
    optimization = st.session_state.get(K_OPTIMIZATION)
    if optimization is None:
        with panel('Capacidad de pago'):
            no_optimization_box()
        status_box(GUARANTEE_WARNING, 'info')
        return

    fmt = params['fmt']
    valuation_date = pd.Timestamp(aligned.dates[-1])

    with panel('Inversión y posición inicial'):
        left, right = st.columns(2)
        with left:
            capital, capital_local = effective_value(params, 'capital', 'capital_s7')
            capital = float(st.number_input(
                'Capital inicial de la inversión', min_value=0.01,
                value=float(capital), step=1_000.0, format='%.2f', key='capital_s7'))
            local_badge(capital_local)
            strategy = st.radio(
                'Estrategia', STRATEGY_ORDER, horizontal=True,
                format_func=lambda key: label(STRATEGY_LABELS, key),
                key='s7_strategy')
            holding_period_days = int(st.number_input(
                'Periodo de mantenimiento adicional (días bursátiles)',
                min_value=0, max_value=5_040, value=0, step=21,
                key='holding_period_days'))
            rebalance_options = [NO_REBALANCE, *REBALANCE_RULES]
            capacity_rebalance_rule = st.selectbox(
                'Rebalanceo durante el horizonte', rebalance_options,
                format_func=lambda key: ('Sin rebalanceo' if key == NO_REBALANCE
                                         else label(REBALANCE_LABELS, key)),
                key='capacity_rebalance_rule')
        with right:
            pct_fee, fee_local = effective_value(params, 'pct_fee', 'pct_fee_s7')
            pct_fee = float(st.number_input(
                'Comisión porcentual (%)', min_value=0.0, max_value=10.0,
                value=float(pct_fee) * 100.0, step=0.1, format='%.3f',
                key='s7_pct_fee_percent')) / 100.0
            st.session_state['pct_fee_s7'] = pct_fee
            local_badge(fee_local or pct_fee != params['pct_fee'])
            fixed_fee, fixed_local = effective_value(params, 'fixed_fee',
                                                     'fixed_fee_s7')
            fixed_fee = float(st.number_input(
                'Comisión fija por operación', min_value=0.0, value=float(fixed_fee),
                step=1.0, format='%.2f', key='fixed_fee_s7'))
            local_badge(fixed_local)
            cash_reserve_initial = float(st.number_input(
                'Reserva inicial de efectivo (no se repone automáticamente)',
                min_value=0.0, value=0.0, step=500.0, format='%.2f',
                key='cash_reserve_initial'))
            cash_interest = st.toggle('Remunerar el efectivo a la tasa libre de '
                                      'riesgo', value=False, key='s7_cash_interest')
            allow_sales = st.toggle('Permitir ventas para cubrir las cuotas',
                                    value=True, key='s7_allow_sales')
            integer_units = st.toggle('Unidades enteras en la posición inicial',
                                      value=False, key='s7_integer_units')

        balance = st.session_state.get(K_PURCHASE_BALANCE)
        position_source = st.radio(
            'Posición inicial', POSITION_SOURCES,
            format_func=lambda key: label(COLUMN_LABELS, key),
            key='position_source',
            disabled=balance is None)
        if balance is None:
            small_note('El balance de compra de la sección 5 todavía no existe: la '
                       'posición se construye desde el capital.')
            position_source = 'from_capital'

    with panel('Calendario de obligaciones'):
        try:
            schedule, is_custom = _schedule_controls(valuation_date)
        except Exception as error:                 # noqa: BLE001
            status_box(f'El calendario de cuotas no es válido: {error}', 'err')
            status_box(GUARANTEE_WARNING, 'info')
            return
        st.session_state[K_CAPACITY_SCHEDULE] = schedule
        st.session_state[K_SCHEDULE_IS_CUSTOM] = bool(is_custom)

    try:
        calendar = build_calendar(valuation_date, schedule, holding_period_days)
        payment_indices, horizon, effective_dates = map_payment_indices(
            valuation_date, schedule, holding_period_days)
    except Exception as error:                     # noqa: BLE001
        with panel('Horizonte'):
            status_box(f'No se pudo construir el horizonte: {error}', 'err')
        status_box(GUARANTEE_WARNING, 'info')
        return

    with panel('Horizonte simulado'):
        status_box(f'Fecha de valoración: {valuation_date:%d/%m/%Y} · '
                   f'Fin del horizonte: {calendar[-1]:%d/%m/%Y} · '
                   f'{horizon} días bursátiles simulados', 'ok')

    with panel('Liquidez, escenarios y simulación'):
        left, right = st.columns(2)
        with left:
            k_sigma = float(st.number_input(
                'Factor k de los escenarios', min_value=0.0, max_value=3.0,
                value=1.0, step=0.1, format='%.2f', key='k_sigma'))
            if k_sigma == 0.0:
                status_box('Con k = 0 los tres escenarios colapsan en el central y '
                           'el conservador deja de ser informativo.', 'info')
            mc_method = st.selectbox(
                'Método de simulación', PATH_METHODS,
                format_func=lambda key: ('Normal multivariante (Cholesky)'
                                         if key == 'gaussian_cholesky'
                                         else 'Bootstrap por bloques'),
                key='mc_method')
        with right:
            n_paths = int(st.number_input(
                'Número de trayectorias', min_value=200, max_value=50_000,
                value=2_000, step=100, key='n_paths'))
            if n_paths > 20_000:
                small_note('Por encima de 20.000 trayectorias el cálculo puede '
                           'tardar varios minutos.')
            block_cap = int(min(BLOCK_SIZE_RANGE[1],
                                max(BLOCK_SIZE_RANGE[0],
                                    int(len(aligned.return_dates)))))
            block_size = int(st.number_input(
                'Tamaño de bloque', min_value=BLOCK_SIZE_RANGE[0],
                max_value=block_cap, value=min(21, block_cap), step=1,
                key='block_size',
                disabled=mc_method != 'block_bootstrap'))
            small_note(f'Semilla activa: {params["seed"]} (configurable en la barra '
                       f'lateral); dos ejecuciones con la misma semilla coinciden.')
        thresholds = _threshold_controls()

    policy = LiquidityPolicy(pct_fee=pct_fee, fixed_fee=fixed_fee,
                             allow_sales=bool(allow_sales),
                             cash_reserve_initial=cash_reserve_initial,
                             cash_interest=bool(cash_interest))
    summary = optimization[strategy]
    prices = np.asarray(aligned.prices, dtype=float)[-1, :]
    try:
        initial = build_initial_position(
            capital, summary.weights, prices, policy,
            integer_units=bool(integer_units),
            from_balance=balance if position_source == 'from_balance' else None)
    except Exception as error:                     # noqa: BLE001
        with panel('Posición inicial'):
            status_box(f'No se pudo construir la posición inicial: {error}', 'err')
        status_box(GUARANTEE_WARNING, 'info')
        return
    st.session_state[K_INITIAL_POSITION] = initial

    with panel('Posición inicial resultante'):
        frame = pd.DataFrame({
            'asset': [str(name) for name in stats.names],
            'units': np.asarray(initial.units, dtype=float),
            'price': prices,
            'value': np.asarray(initial.units, dtype=float) * prices,
        }).rename(columns=lambda key: label(COLUMN_LABELS, key))
        st.dataframe(
            style_table(frame, num_cols=[label(COLUMN_LABELS, 'units')],
                        money_cols=[label(COLUMN_LABELS, key)
                                    for key in ('price', 'value')], fmt=fmt),
            use_container_width=True, hide_index=True)
        small_note(f'Invertido {fmt_currency(initial.invested, fmt)} · '
                   f'Efectivo {fmt_currency(initial.cash, fmt)} · '
                   f'Costos de compra {fmt_currency(initial.purchase_costs, fmt)} · '
                   f'Patrimonio inicial '
                   f'{fmt_currency(initial.wealth_initial, fmt)}')

    rule = None if capacity_rebalance_rule == NO_REBALANCE \
        else str(capacity_rebalance_rule)
    capacity_inputs = {
        'holding_period_days': holding_period_days,
        'valuation_date': f'{valuation_date:%Y-%m-%d}',
        'horizon_end': f'{calendar[-1]:%Y-%m-%d}',
        'cash_reserve_initial': cash_reserve_initial,
        'allow_sales': bool(allow_sales),
        'cash_interest': bool(cash_interest),
        'k_sigma': k_sigma,
        'method': str(mc_method),
        'n_paths': n_paths,
        'block_size': block_size,
        'n_installments': int(len(schedule.dates)),
    }
    st.session_state[K_CAPACITY_INPUTS] = capacity_inputs
    fingerprint_params.update({
        'capital_s7': capital, 'pct_fee_s7': pct_fee, 'fixed_fee_s7': fixed_fee,
        'position_source': position_source,
        'cash_reserve_initial': cash_reserve_initial,
        'holding_period_days': holding_period_days, 'schedule': schedule,
        'liquidity_policy': policy, 'k_sigma': k_sigma, 'mc_method': mc_method,
        'block_size': block_size, 'n_paths': n_paths,
        'capacity_rebalance_rule': rule, 'capacity_thresholds': thresholds,
    })

    with panel('Ejecución de la simulación'):
        stale_badge('capacity_metrics', fingerprint_params)
        if st.button('Evaluar capacidad de pago', type='primary', key='s7_run'):
            try:
                metrics = cached_capacity(
                    stats.mu_daily, stats.cov_daily, summary.weights,
                    initial.to_tuple(), schedule.to_tuple(), payment_indices,
                    calendar.asi8, effective_dates.asi8, policy.to_tuple(),
                    params['convention'].to_tuple(), k_sigma, n_paths,
                    str(mc_method), block_size, aligned.log_returns, rule,
                    params['seed'])
            except Exception as error:             # noqa: BLE001
                status_box(f'No se pudo evaluar la capacidad de pago: {error}', 'err')
            else:
                st.session_state[K_CAPACITY_METRICS] = metrics
                store_fingerprint('capacity_metrics', fingerprint_params)
                st.session_state[K_CAPACITY_VERDICT] = classify_payment_capacity(
                    metrics, thresholds)
                store_fingerprint('capacity_verdict', fingerprint_params)

    metrics = st.session_state.get(K_CAPACITY_METRICS)
    if metrics is None:
        with panel('Resultados'):
            status_box('Pulsa «Evaluar capacidad de pago» para simular el horizonte '
                       'completo con el calendario configurado.', 'info')
        status_box(GUARANTEE_WARNING, 'info')
        return

    verdict = classify_payment_capacity(metrics, thresholds)
    st.session_state[K_CAPACITY_VERDICT] = verdict

    with panel('Veredicto'):
        _render_verdict(metrics, verdict, thresholds, fmt)

    with panel('Indicadores'):
        _render_indicators(metrics, calendar[-1], fmt)

    with panel('Tres niveles de lectura'):
        _render_reading_levels(metrics, fmt)

    with panel('Patrimonio, cuotas y efectivo'):
        left, right = st.columns(2)
        with left:
            st.plotly_chart(wealth_vs_installments(metrics.central_ledger),
                            use_container_width=True, config=PLOTLY_CONFIG)
        with right:
            st.plotly_chart(cash_timeline(metrics.central_ledger),
                            use_container_width=True, config=PLOTLY_CONFIG)

    with panel('Escenarios y distribución simulada'):
        left, right = st.columns(2)
        with left:
            st.plotly_chart(scenario_comparison(metrics.deterministic),
                            use_container_width=True, config=PLOTLY_CONFIG)
        with right:
            st.plotly_chart(
                final_wealth_histogram(metrics.mc_final_wealth,
                                       metrics.final_wealth_percentiles,
                                       metrics.total_scheduled),
                use_container_width=True, config=PLOTLY_CONFIG)

    with panel('Cobertura de cuotas'):
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                coverage_summary(metrics.p_cover_all,
                                 int(np.asarray(metrics.mc_covered_all).size),
                                 thresholds),
                use_container_width=True, config=PLOTLY_CONFIG)
        with right:
            st.plotly_chart(coverage_by_installment(metrics.coverage_by_installment),
                            use_container_width=True, config=PLOTLY_CONFIG)
        spec = metrics.path_spec
        small_note(f'Método {spec.method} · {spec.n_paths} trayectorias · horizonte '
                   f'{spec.horizon} días · lotes de {spec.batch_size} · semilla '
                   f'{spec.seed}')

    with panel('Tabla de pagos'):
        _render_ledger(metrics, fmt)
        for code in metrics.note_codes or ():
            status_box(label(NOTE_LABELS, code), 'info')

    with panel('Advertencias y supuestos'):
        status_box(GUARANTEE_WARNING, 'err')
        with st.expander('Supuestos del Monte Carlo'):
            st.markdown(MC_ASSUMPTIONS)

    with panel('Informe'):
        export_button(params, 's7_export')
