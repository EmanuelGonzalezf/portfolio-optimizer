"""Section 4 — Frontera eficiente y simulaciones (design.md section 14, block 4)."""

from __future__ import annotations

import streamlit as st

from core.formatting import style_table
from ui.charts import PLOTLY_CONFIG, frontier_chart
from ui.common import (base_fingerprint_params, ensure_stats, no_data_box,
                       no_optimization_box, stale_badge)
from ui.compute import cached_frontier, cached_weight_cloud
from ui.labels import COLUMN_LABELS, WARNING_LABELS, label
from ui.state import K_CLOUD, K_FRONTIER, K_OPTIMIZATION, store_fingerprint
from ui.theme import PALETTE, badge, panel, small_note, status_box

__all__ = ['render']

CHART_NOTE = ('Cada punto es un portafolio aleatorio factible coloreado por su razón '
              'de Sharpe. La curva recorre la frontera eficiente y los rombos ámbar '
              'son los activos individuales.')


def render(params: dict) -> None:
    fingerprint_params = base_fingerprint_params(params)
    aligned, stats = ensure_stats(params, fingerprint_params)
    if aligned is None:
        with panel('Frontera eficiente'):
            no_data_box()
        return
    optimization = st.session_state.get(K_OPTIMIZATION)
    if optimization is None:
        with panel('Frontera eficiente'):
            no_optimization_box()
        return

    with panel('Ejecución'):
        stale_badge('frontier', fingerprint_params)
        stale_badge('weight_cloud', fingerprint_params)
        small_note(f'Puntos solicitados: {params["n_frontier_points"]} · '
                   f'Simulaciones Monte Carlo de pesos: '
                   f'{params["n_weight_sims"]:,}'.replace(',', '.'))
        if st.button('Calcular frontera y nube', type='primary', key='s4_run',
                     disabled=not params['constraints_feasible']):
            rf_cc = params['convention'].rf_cc_annual
            try:
                frontier = cached_frontier(
                    stats.mu_annual, stats.cov_annual, rf_cc, params['allow_short'],
                    params['max_weight'], params['short_lb'],
                    params['n_frontier_points'], params['seed'])
                cloud = cached_weight_cloud(
                    stats.mu_annual, stats.cov_annual, rf_cc, params['allow_short'],
                    params['max_weight'], params['short_lb'],
                    params['n_weight_sims'], params['seed'])
            except Exception as error:             # noqa: BLE001
                status_box(f'No se pudo construir la frontera: {error}', 'err')
            else:
                st.session_state[K_FRONTIER] = frontier
                st.session_state[K_CLOUD] = cloud
                store_fingerprint('frontier', fingerprint_params)
                store_fingerprint('weight_cloud', fingerprint_params)

    frontier = st.session_state.get(K_FRONTIER)
    cloud = st.session_state.get(K_CLOUD)
    if frontier is None:
        with panel('Frontera eficiente'):
            status_box('Pulsa «Calcular frontera y nube» para generar la frontera '
                       'eficiente y los portafolios aleatorios.', 'info')
        return

    sample = cloud[0] if cloud is not None else None
    with panel('Frontera eficiente y nube Monte Carlo'):
        badges = [badge(f'Puntos calculados: {frontier.computed}/{frontier.requested}',
                        PALETTE['accent'])]
        if sample is not None:
            badges.append(badge(f'Muestreo: {sample.method}', PALETTE['accent_2']))
            badges.append(badge(f'Tasa de aceptación: '
                                f'{sample.acceptance_rate * 100:.1f} %',
                                PALETTE['accent_3']))
        st.markdown(''.join(badges), unsafe_allow_html=True)

        if sample is not None and sample.weights.size == 0:
            status_box(label(WARNING_LABELS, sample.message
                             or 'weight_sampling_failed_relax_weight_cap_or_'
                                'short_bound'), 'err')
        elif sample is not None and sample.method != 'dirichlet':
            status_box(label(WARNING_LABELS, 'weight_sampling_biased'), 'info')

        cloud_metrics = None if cloud is None or sample is None \
            or sample.weights.size == 0 else tuple(cloud[1:4])
        st.plotly_chart(
            frontier_chart(frontier, cloud_metrics, stats,
                           optimization.get('min_variance'),
                           optimization.get('max_sharpe')),
            use_container_width=True, config=PLOTLY_CONFIG)
        small_note(CHART_NOTE)
        if frontier.skipped_targets is not None and len(frontier.skipped_targets):
            small_note(f'{len(frontier.skipped_targets)} rentabilidades objetivo no '
                       f'eran alcanzables con las restricciones activas.')

    with panel('Puntos de la frontera'):
        points = frontier.points.copy()
        rename = {'vol': label(COLUMN_LABELS, 'vol'),
                  'ret': label(COLUMN_LABELS, 'ret'),
                  'sharpe': label(COLUMN_LABELS, 'sharpe')}
        for position, name in enumerate(stats.names):
            rename[f'w_{position}'] = str(name)
        points = points.rename(columns=rename)
        pct_cols = [rename['vol'], rename['ret']] + [str(n) for n in stats.names]
        st.dataframe(
            style_table(points, pct_cols=pct_cols,
                        num_cols=[rename['sharpe']], fmt=params['fmt']),
            use_container_width=True, hide_index=True)
        st.download_button('Descargar puntos en CSV',
                           data=frontier.points.to_csv(index=False).encode('utf-8'),
                           file_name='frontera_eficiente.csv', mime='text/csv',
                           key='s4_csv')
