"""Plotly template and the sixteen chart builders (design.md section 12.2).

The template replicates ``plotlyLayout()`` of the prototype literally
(requirements.md section 2.4). None of these functions computes anything: they
all take results that ``core/`` already produced and return a ``go.Figure``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.backtest import STRATEGIES
from core.scenarios import CapacityThresholds
from ui.theme import PALETTE

__all__ = [
    'PLOTLY_CONFIG',
    'base_layout',
    'CYCLE',
    'STRATEGY_COLORS',
    'weights_bar',
    'frontier_chart',
    'correlation_heatmap',
    'covariance_heatmap',
    'normalized_prices',
    'returns_distribution',
    'equity_curves',
    'cumulative_return_chart',
    'drawdown_chart',
    'weights_area',
    'wealth_vs_installments',
    'cash_timeline',
    'scenario_comparison',
    'final_wealth_histogram',
    'coverage_summary',
    'coverage_by_installment',
]

PLOTLY_CONFIG = {'displaylogo': False, 'responsive': True,
                 'modeBarButtonsToRemove': ['select2d', 'lasso2d']}

CYCLE = (PALETTE['accent'], PALETTE['accent_2'], PALETTE['accent_3'],
         PALETTE['warn'], PALETTE['err'])

STRATEGY_COLORS = {
    'min_variance': PALETTE['accent'],
    'max_sharpe': PALETTE['accent_3'],
    'equal_weight': PALETTE['accent_2'],
}

SCENARIO_COLORS = {
    'conservative': PALETTE['warn'],
    'central': PALETTE['accent_2'],
    'favorable': PALETTE['accent'],
}

GRID = PALETTE['border']
AXIS_TITLE = PALETTE['text_dim']
DEFAULT_HEIGHT = 380
FRONTIER_HEIGHT = 520


def base_layout(**over) -> dict:
    """The prototype layout, with ``over`` merged on top of it."""
    layout = {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {'color': PALETTE['text'], 'size': 12},
        'autosize': True,
        'margin': {'t': 18, 'b': 50, 'l': 60, 'r': 20},
        'hoverlabel': {'bgcolor': PALETTE['bg_2'], 'bordercolor': PALETTE['border'],
                       'font': {'color': PALETTE['text']}},
        'xaxis': {'gridcolor': GRID, 'zerolinecolor': GRID,
                  'title': {'font': {'color': AXIS_TITLE}}},
        'yaxis': {'gridcolor': GRID, 'zerolinecolor': GRID,
                  'title': {'font': {'color': AXIS_TITLE}}},
        'legend': {'bgcolor': 'rgba(13,19,32,.7)', 'bordercolor': PALETTE['border'],
                   'borderwidth': 1, 'font': {'color': PALETTE['text']}},
        'height': DEFAULT_HEIGHT,
    }
    layout.update(over)
    return layout


def _as_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float).ravel()


def _strategy_columns(frame: pd.DataFrame) -> list:
    return [column for column in frame.columns if column in STRATEGIES]


def _asset_columns(frame: pd.DataFrame) -> list:
    return [column for column in frame.columns if column not in STRATEGIES]


# --------------------------------------------------------------------------- #
# 1. Weights
# --------------------------------------------------------------------------- #
def weights_bar(summary, names) -> go.Figure:
    """Weight distribution: descending bars, green when positive, red when short."""
    weights = _as_array(summary.weights)
    labels = [str(name) for name in names]
    order = np.argsort(-weights)
    sorted_weights = weights[order]
    sorted_labels = [labels[i] for i in order]
    colors = [PALETTE['accent'] if value >= 0 else PALETTE['err']
              for value in sorted_weights]
    figure = go.Figure(go.Bar(
        x=sorted_labels, y=sorted_weights, marker_color=colors,
        hovertemplate='<b>%{x}</b><br>Peso: %{y:.2%}<extra></extra>'))
    figure.update_layout(**base_layout(
        margin={'t': 10, 'b': 70, 'l': 50, 'r': 15},
        xaxis={'gridcolor': GRID, 'tickangle': -25},
        yaxis={'gridcolor': GRID, 'tickformat': '.0%', 'zerolinecolor': GRID},
        showlegend=False))
    return figure


# --------------------------------------------------------------------------- #
# 2. Efficient frontier and Monte Carlo cloud
# --------------------------------------------------------------------------- #
def frontier_chart(frontier, cloud, stats, min_var, max_sharpe) -> go.Figure:
    """Random cloud coloured by Sharpe, frontier curve, assets and the two stars."""
    figure = go.Figure()
    if cloud is not None:
        vol, ret, sharpe = (_as_array(item) for item in cloud[:3])
        if vol.size:
            figure.add_trace(go.Scattergl(
                x=vol, y=ret, mode='markers', name='Portafolios aleatorios',
                marker={'size': 4, 'color': sharpe, 'colorscale': 'Viridis',
                        'colorbar': {'title': 'Sharpe'}, 'showscale': True},
                hovertemplate=('Volatilidad: %{x:.2%}<br>Rentabilidad: %{y:.2%}'
                               '<extra></extra>')))
    if frontier is not None and frontier.points is not None and not frontier.points.empty:
        figure.add_trace(go.Scatter(
            x=frontier.points['vol'], y=frontier.points['ret'], mode='lines',
            name='Frontera eficiente',
            line={'color': PALETTE['accent_2'], 'width': 3},
            hovertemplate=('Volatilidad: %{x:.2%}<br>Rentabilidad: %{y:.2%}'
                           '<extra></extra>')))
    if stats is not None:
        figure.add_trace(go.Scatter(
            x=_as_array(stats.sigma_annual), y=_as_array(stats.mu_annual),
            mode='markers+text', name='Activos individuales',
            text=[str(name) for name in stats.names], textposition='top center',
            textfont={'size': 11, 'color': PALETTE['text_dim']},
            marker={'size': 10, 'color': PALETTE['warn'],
                    'line': {'color': '#0a0e1a', 'width': 1}},
            hovertemplate=('<b>%{text}</b><br>Volatilidad: %{x:.2%}'
                           '<br>Rentabilidad: %{y:.2%}<extra></extra>')))
    for summary, color, name in ((min_var, PALETTE['accent'], 'Mínima varianza'),
                                 (max_sharpe, PALETTE['accent_3'], 'Máxima Sharpe')):
        if summary is None:
            continue
        figure.add_trace(go.Scatter(
            x=[summary.volatility_annual], y=[summary.expected_return_annual],
            mode='markers', name=name,
            marker={'symbol': 'star', 'size': 22, 'color': color},
            hovertemplate=('<b>' + name + '</b><br>Volatilidad: %{x:.2%}'
                           '<br>Rentabilidad: %{y:.2%}<extra></extra>')))
    figure.update_layout(**base_layout(
        height=FRONTIER_HEIGHT,
        margin={'t': 18, 'b': 60, 'l': 70, 'r': 20},
        xaxis={'gridcolor': GRID, 'tickformat': '.0%',
               'title': {'text': 'Volatilidad (σ)', 'font': {'color': AXIS_TITLE}}},
        yaxis={'gridcolor': GRID, 'tickformat': '.0%',
               'title': {'text': 'Rentabilidad esperada',
                         'font': {'color': AXIS_TITLE}}},
        legend={'x': 0.02, 'y': 0.98, 'bgcolor': 'rgba(13,19,32,.7)',
                'bordercolor': PALETTE['border'], 'borderwidth': 1,
                'font': {'color': PALETTE['text']}}))
    return figure


# --------------------------------------------------------------------------- #
# 3 and 4. Correlation and covariance heatmaps
# --------------------------------------------------------------------------- #
def correlation_heatmap(corr, names) -> go.Figure:
    labels = [str(name) for name in names]
    matrix = np.asarray(corr, dtype=float)
    figure = go.Figure(go.Heatmap(
        z=matrix, x=labels, y=labels, colorscale='RdBu', zmin=-1, zmax=1,
        reversescale=True, text=matrix, texttemplate='%{text:.2f}',
        textfont={'size': 10},
        hovertemplate='%{y} / %{x}: %{z:.4f}<extra></extra>'))
    figure.update_layout(**base_layout(
        margin={'t': 40, 'b': 70, 'l': 90, 'r': 20},
        title={'text': 'Matriz de correlación', 'font': {'size': 14}},
        xaxis={'tickangle': -30, 'gridcolor': GRID},
        yaxis={'autorange': 'reversed', 'gridcolor': GRID}))
    return figure


def covariance_heatmap(cov, names) -> go.Figure:
    labels = [str(name) for name in names]
    matrix = np.asarray(cov, dtype=float)
    figure = go.Figure(go.Heatmap(
        z=matrix, x=labels, y=labels, colorscale='Viridis',
        text=matrix, texttemplate='%{text:.2e}', textfont={'size': 9},
        hovertemplate='%{y} / %{x}: %{z:.6f}<extra></extra>'))
    figure.update_layout(**base_layout(
        margin={'t': 40, 'b': 70, 'l': 90, 'r': 20},
        title={'text': 'Covarianza anualizada', 'font': {'size': 14}},
        xaxis={'tickangle': -30, 'gridcolor': GRID},
        yaxis={'autorange': 'reversed', 'gridcolor': GRID}))
    return figure


# --------------------------------------------------------------------------- #
# 5 and 6. Asset analysis
# --------------------------------------------------------------------------- #
def normalized_prices(aligned) -> go.Figure:
    """Price series rebased to 100 at the first aligned date."""
    prices = np.asarray(aligned.prices, dtype=float)
    figure = go.Figure()
    for i, name in enumerate(aligned.names):
        base = prices[0, i]
        series = prices[:, i] / base * 100.0 if base else prices[:, i]
        figure.add_trace(go.Scatter(
            x=aligned.dates, y=series, mode='lines', name=str(name),
            line={'color': CYCLE[i % len(CYCLE)], 'width': 2},
            hovertemplate='<b>%{fullData.name}</b><br>%{x|%Y-%m-%d}: %{y:.2f}'
                          '<extra></extra>'))
    figure.update_layout(**base_layout(
        yaxis={'gridcolor': GRID,
               'title': {'text': 'Base 100', 'font': {'color': AXIS_TITLE}}}))
    return figure


def returns_distribution(aligned) -> go.Figure:
    """Overlaid histograms of the daily log returns plus the mean of each asset."""
    returns = np.asarray(aligned.log_returns, dtype=float)
    figure = go.Figure()
    for i, name in enumerate(aligned.names):
        color = CYCLE[i % len(CYCLE)]
        figure.add_trace(go.Histogram(
            x=returns[:, i], name=str(name), opacity=0.55, marker_color=color,
            nbinsx=60))
        figure.add_vline(x=float(returns[:, i].mean()), line_width=1,
                         line_dash='dot', line_color=color)
    figure.update_layout(**base_layout(
        barmode='overlay',
        xaxis={'gridcolor': GRID, 'tickformat': '.1%',
               'title': {'text': 'Retorno logarítmico diario',
                         'font': {'color': AXIS_TITLE}}},
        yaxis={'gridcolor': GRID,
               'title': {'text': 'Frecuencia', 'font': {'color': AXIS_TITLE}}}))
    return figure


# --------------------------------------------------------------------------- #
# 7, 8, 9 and 10. Backtesting
# --------------------------------------------------------------------------- #
def equity_curves(backtest, normalize: bool = False) -> go.Figure:
    """Capital per strategy plus the individual assets in grey."""
    equity = backtest.equity
    figure = go.Figure()
    for column in _asset_columns(equity):
        series = equity[column]
        values = series / series.iloc[0] - 1.0 if normalize else series
        figure.add_trace(go.Scatter(
            x=equity.index, y=values, mode='lines', name=str(column),
            line={'color': PALETTE['text_mute'], 'width': 1}, opacity=0.5))
    for column in _strategy_columns(equity):
        series = equity[column]
        values = series / series.iloc[0] - 1.0 if normalize else series
        figure.add_trace(go.Scatter(
            x=equity.index, y=values, mode='lines', name=str(column),
            line={'color': STRATEGY_COLORS.get(column, PALETTE['accent']),
                  'width': 2.5}))
    if normalize:
        figure.add_hline(y=0.0, line_width=1, line_color=PALETTE['border'])
        axis = {'gridcolor': GRID, 'tickformat': '.0%',
                'title': {'text': 'Rentabilidad acumulada',
                          'font': {'color': AXIS_TITLE}}}
    else:
        axis = {'gridcolor': GRID,
                'title': {'text': 'Capital', 'font': {'color': AXIS_TITLE}}}
    figure.update_layout(**base_layout(yaxis=axis))
    return figure


def cumulative_return_chart(backtest) -> go.Figure:
    """``equity_curves`` in ``V/V_0 - 1`` space (RF-BT4)."""
    return equity_curves(backtest, normalize=True)


def drawdown_chart(backtest) -> go.Figure:
    """Filled negative areas, one per strategy."""
    drawdown = backtest.drawdown
    figure = go.Figure()
    for column in _strategy_columns(drawdown):
        figure.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown[column], mode='lines', name=str(column),
            fill='tozeroy', fillcolor='rgba(248,113,113,.25)',
            line={'color': STRATEGY_COLORS.get(column, PALETTE['err']), 'width': 1.5}))
    figure.update_layout(**base_layout(
        yaxis={'gridcolor': GRID, 'tickformat': '.0%',
               'title': {'text': 'Drawdown', 'font': {'color': AXIS_TITLE}}}))
    return figure


def weights_area(weights_history) -> go.Figure:
    """Stacked area of the weight path of one strategy."""
    frame = weights_history
    if isinstance(frame, dict):
        frame = next(iter(frame.values())) if frame else pd.DataFrame()
    figure = go.Figure()
    for i, column in enumerate(frame.columns):
        figure.add_trace(go.Scatter(
            x=frame.index, y=frame[column], mode='lines', name=str(column),
            stackgroup='weights', line={'width': 0.5},
            fillcolor=CYCLE[i % len(CYCLE)],
            hovertemplate='<b>%{fullData.name}</b><br>%{y:.2%}<extra></extra>'))
    figure.update_layout(**base_layout(
        yaxis={'gridcolor': GRID, 'tickformat': '.0%',
               'title': {'text': 'Peso', 'font': {'color': AXIS_TITLE}}}))
    return figure


# --------------------------------------------------------------------------- #
# 11 to 16. Payment capacity
# --------------------------------------------------------------------------- #
def wealth_vs_installments(ledger) -> go.Figure:
    """Wealth line, installment bars on a secondary axis, red default markers."""
    figure = make_subplots(specs=[[{'secondary_y': True}]])
    dates = ledger.index
    figure.add_trace(go.Bar(
        x=dates, y=ledger['due'], name='Cuota exigible',
        marker_color='rgba(56,189,248,.45)'), secondary_y=True)
    figure.add_trace(go.Scatter(
        x=dates, y=ledger['wealth_after'], mode='lines+markers',
        name='Capital restante',
        line={'color': PALETTE['accent'], 'width': 2.5}), secondary_y=False)
    defaults = ledger[ledger['deficit'] > 0]
    if len(defaults):
        figure.add_trace(go.Scatter(
            x=defaults.index, y=defaults['wealth_after'], mode='markers',
            name='Cuotas incumplidas',
            marker={'color': PALETTE['err'], 'size': 11, 'symbol': 'x'}),
            secondary_y=False)
    figure.update_layout(**base_layout(margin={'t': 18, 'b': 50, 'l': 70, 'r': 60}))
    figure.update_yaxes(title_text='Capital', gridcolor=GRID, secondary_y=False)
    figure.update_yaxes(title_text='Cuota', gridcolor=GRID, showgrid=False,
                        secondary_y=True)
    return figure


def cash_timeline(ledger) -> go.Figure:
    """Cash line with the payment dates marked by dotted vertical lines."""
    dates = ledger.index
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=dates, y=ledger['cash_after'], mode='lines+markers',
        name='Efectivo disponible',
        line={'color': PALETTE['accent_2'], 'width': 2.5}))
    figure.add_trace(go.Scatter(
        x=dates, y=ledger['cash_before'], mode='markers',
        name='Efectivo antes del pago',
        marker={'color': PALETTE['warn'], 'size': 8}))
    for moment in dates:
        figure.add_vline(x=moment, line_width=1, line_dash='dot',
                         line_color=PALETTE['border'])
    figure.update_layout(**base_layout(
        yaxis={'gridcolor': GRID,
               'title': {'text': 'Efectivo', 'font': {'color': AXIS_TITLE}}}))
    return figure


def scenario_comparison(results) -> go.Figure:
    """Wealth at each payment date under the three deterministic scenarios."""
    figure = go.Figure()
    for key in ('conservative', 'central', 'favorable'):
        result = results.get(key) if isinstance(results, dict) else None
        if result is None:
            continue
        wealth = np.asarray(result.wealth_at_payments, dtype=float)
        series = wealth[0] if wealth.ndim == 2 else wealth
        figure.add_trace(go.Scatter(
            x=np.arange(1, series.size + 1), y=series, mode='lines+markers',
            name=key, line={'color': SCENARIO_COLORS[key], 'width': 2.5}))
    figure.update_layout(**base_layout(
        xaxis={'gridcolor': GRID, 'dtick': 1,
               'title': {'text': 'Cuota', 'font': {'color': AXIS_TITLE}}},
        yaxis={'gridcolor': GRID,
               'title': {'text': 'Capital antes del pago',
                         'font': {'color': AXIS_TITLE}}}))
    return figure


def final_wealth_histogram(final_wealth, percentiles, total_scheduled=None) -> go.Figure:
    """Monte Carlo distribution of the final capital with p5, p50 and p95."""
    values = _as_array(final_wealth)
    figure = go.Figure(go.Histogram(
        x=values, nbinsx=60, marker_color=PALETTE['accent_2'], opacity=0.85,
        name='Capital final'))
    marks = (('p5', PALETTE['err']), ('p50', PALETTE['accent']),
             ('p95', PALETTE['accent_3']))
    for key, color in marks:
        value = (percentiles or {}).get(key)
        if value is None or not np.isfinite(value):
            continue
        figure.add_vline(x=float(value), line_width=2, line_dash='dash',
                         line_color=color, annotation_text=key,
                         annotation_font_color=color)
    if total_scheduled is not None and np.isfinite(total_scheduled):
        figure.add_vline(x=float(total_scheduled), line_width=2,
                         line_color=PALETTE['warn'],
                         annotation_text='Total de cuotas',
                         annotation_font_color=PALETTE['warn'])
    figure.update_layout(**base_layout(
        xaxis={'gridcolor': GRID,
               'title': {'text': 'Capital final', 'font': {'color': AXIS_TITLE}}},
        yaxis={'gridcolor': GRID,
               'title': {'text': 'Trayectorias', 'font': {'color': AXIS_TITLE}}}))
    return figure


def coverage_summary(p_cover_all, n_paths, thresholds=None) -> go.Figure:
    """Chart (E) of RF-C19: gauge of the share of paths covering every payment.

    The gauge steps are the ``CapacityThresholds`` themselves, so the reader
    sees the classification bands and not an arbitrary colour ramp.
    """
    limits = thresholds or CapacityThresholds()
    value = float(p_cover_all) if np.isfinite(p_cover_all) else 0.0
    figure = go.Figure(go.Indicator(
        mode='number+gauge',
        value=value,
        number={'valueformat': '.1%', 'font': {'size': 34}},
        title={'text': f'sobre {int(n_paths)} trayectorias simuladas',
               'font': {'size': 12, 'color': PALETTE['text_dim']}},
        gauge={
            'axis': {'range': [0.0, 1.0], 'tickformat': '.0%',
                     'tickcolor': PALETTE['text_dim']},
            'bar': {'color': PALETTE['text']},
            'bgcolor': PALETTE['bg_1'],
            'bordercolor': PALETTE['border'],
            'steps': [
                {'range': [0.0, limits.p_high_risk], 'color': 'rgba(248,113,113,.35)'},
                {'range': [limits.p_high_risk, limits.p_tight],
                 'color': 'rgba(251,191,36,.35)'},
                {'range': [limits.p_tight, limits.p_sufficient],
                 'color': 'rgba(56,189,248,.35)'},
                {'range': [limits.p_sufficient, 1.0], 'color': 'rgba(74,222,128,.35)'},
            ],
        }))
    figure.update_layout(**base_layout(height=260,
                                       margin={'t': 40, 'b': 20, 'l': 40, 'r': 40}))
    return figure


def _coverage_color(value: float) -> str:
    """Interpolate red (0) to green (1) for the per-installment bars."""
    low = (248, 113, 113)
    high = (74, 222, 128)
    ratio = 0.0 if not np.isfinite(value) else min(max(float(value), 0.0), 1.0)
    channels = [int(round(low[i] + (high[i] - low[i]) * ratio)) for i in range(3)]
    return f'rgb({channels[0]},{channels[1]},{channels[2]})'


def coverage_by_installment(coverage) -> go.Figure:
    """RF-C17: simulated probability of covering each individual installment."""
    values = _as_array(coverage)
    figure = go.Figure(go.Bar(
        x=np.arange(1, values.size + 1), y=values,
        marker_color=[_coverage_color(value) for value in values],
        hovertemplate='Cuota %{x}<br>Cobertura: %{y:.1%}<extra></extra>'))
    figure.update_layout(**base_layout(
        xaxis={'gridcolor': GRID, 'dtick': 1,
               'title': {'text': 'Cuota', 'font': {'color': AXIS_TITLE}}},
        yaxis={'gridcolor': GRID, 'tickformat': '.0%', 'range': [0.0, 1.0],
               'title': {'text': 'Cobertura simulada',
                         'font': {'color': AXIS_TITLE}}},
        showlegend=False))
    return figure
