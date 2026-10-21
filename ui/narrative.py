"""Short conclusion built from the real results (RF-C20).

This module formats values that ``core/`` already computed and translates keys
with ``ui/labels.py``. There is no financial formula here, not even a ratio:
every number arrives ready from ``CapacityMetrics`` or ``CapacityVerdict``.
"""

from __future__ import annotations

import math

import numpy as np

from core.formatting import NOT_AVAILABLE, CurrencyFormat, fmt_currency, fmt_pct
from ui.labels import CATEGORY_LABELS, label

__all__ = ['build_narrative']


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _central_final_wealth(metrics):
    central = (metrics.deterministic or {}).get('central')
    if central is None:
        return math.nan
    values = np.asarray(central.final_wealth, dtype=float).ravel()
    return float(values[0]) if values.size else math.nan


def build_narrative(metrics, verdict, fmt: CurrencyFormat = CurrencyFormat()) -> str:
    """Answer the six questions of RF-C20 in one paragraph per question.

    ``verdict`` may be ``None`` when the schedule has no obligations: in that
    case the classification is not issued and the text says so instead of
    inventing a category.
    """
    lines: list = []

    capital = fmt_currency(metrics.initial_capital, fmt)
    scheduled = fmt_currency(metrics.total_scheduled, fmt)
    lines.append(
        f'**Capital inicial frente a las obligaciones.** El capital inicial es '
        f'{capital} y el total de cuotas programadas asciende a {scheduled} '
        f'repartido en {int(metrics.n_installments)} cuotas.')

    if metrics.central_covers_all:
        lines.append('**Cobertura en el escenario central.** Todas las cuotas quedan '
                     'cubiertas en el escenario central.')
    else:
        lines.append('**Cobertura en el escenario central.** El escenario central deja '
                     'al menos una cuota sin cubrir.')

    final_wealth = _central_final_wealth(metrics)
    margin = metrics.margin_ratio
    margin_text = fmt_pct(margin) if _finite(margin) else NOT_AVAILABLE
    lines.append(
        f'**Margen de capital final.** El capital final del escenario central es '
        f'{fmt_currency(final_wealth, fmt)}, equivalente a un margen de {margin_text} '
        f'sobre el total de las cuotas.')

    deficit_ratio = metrics.conservative_deficit_ratio
    deficit_text = fmt_pct(deficit_ratio) if _finite(deficit_ratio) else NOT_AVAILABLE
    if metrics.conservative_covers_all:
        lines.append('**Escenario adverso.** El escenario conservador también cubre '
                     'todas las cuotas.')
    else:
        lines.append(
            f'**Escenario adverso.** El escenario conservador deja impagada una parte '
            f'de la obligación total equivalente al {deficit_text}.')

    p_cover = metrics.p_cover_all
    n_paths = int(np.asarray(metrics.mc_covered_all).size)
    p_text = fmt_pct(p_cover, 1) if _finite(p_cover) else NOT_AVAILABLE
    lines.append(
        f'**Capacidad simulada.** El {p_text} de las {n_paths} trayectorias simuladas '
        f'cubre todas las cuotas. Es una proporción de trayectorias bajo los supuestos '
        f'declarados, no una probabilidad real.')

    additional = metrics.additional_capital_required
    if not _finite(additional):
        lines.append('**Capital adicional necesario.** No se pudo acotar el capital '
                     'adicional necesario dentro del rango explorado.')
    elif additional <= 0.0:
        lines.append('**Capital adicional necesario.** No se requiere capital '
                     'adicional en el escenario central.')
    else:
        lines.append(
            f'**Capital adicional necesario.** Harían falta '
            f'{fmt_currency(additional, fmt)} adicionales sobre el capital inicial '
            f'para cubrir todas las cuotas en el escenario central.')

    if verdict is not None:
        lines.append(f'**Categoría asignada.** '
                     f'{label(CATEGORY_LABELS, verdict.category)} '
                     f'(regla {int(verdict.rule_fired)}).')
    else:
        lines.append('**Categoría asignada.** No se emite veredicto: el calendario no '
                     'contiene obligaciones exigibles.')

    return '\n\n'.join(lines)
