"""Single source of visible text (design.md section 2.1).

``core/`` emits English snake_case codes only; this module is the one place
that turns them into Spanish prose. One fallback rule applies everywhere:

    label(mapping, key) -> mapping.get(key, key)

A missing key degrades to the raw English key. Never a ``KeyError``, never an
empty string.
"""

from __future__ import annotations

__all__ = [
    'label',
    'ERROR_LABELS',
    'WARNING_LABELS',
    'NOTE_LABELS',
    'SHEET_NAMES',
    'COLUMN_LABELS',
    'EXPORT_LABELS',
    'PREVIEW_COLUMN_LABELS',
    'DATE_FORMAT_LABELS',
    'SCHEDULE_COLUMN_LABELS',
    'FREQUENCY_LABELS',
    'REBALANCE_LABELS',
    'SCENARIO_LABELS',
    'CATEGORY_LABELS',
    'STRATEGY_LABELS',
    'SECTIONS',
    'GENERIC_ERROR',
]


def label(mapping: dict, key) -> str:
    """Translate ``key`` with ``mapping``; fall back to the raw key."""
    text = str(key)
    value = mapping.get(text) if isinstance(mapping, dict) else None
    if value is None or str(value) == '':
        return text
    return str(value)


GENERIC_ERROR = 'Se produjo un problema al procesar los datos'


# --------------------------------------------------------------------------- #
# Errors: PortfolioError.code, ScheduleError, InfeasibleConstraintsError and the
# ValueError codes raised by core/
# --------------------------------------------------------------------------- #
ERROR_LABELS: dict = {
    # core/types.py - Constraints.validate
    'max_weight_not_positive': 'El límite máximo por activo debe ser mayor que cero',
    'max_weight_below_equal_share':
        'El límite máximo por activo es infactible: no puede ser inferior a 1/n',
    'short_lower_bound_not_negative':
        'Con ventas en corto activadas, la cota inferior debe ser negativa',

    # core/data_loader.py
    'excel_engine_missing': 'Falta el motor de lectura para este tipo de archivo de Excel',
    'workbook_unreadable': 'No se pudo abrir el libro de Excel',
    'csv_unreadable': 'No se pudo leer el archivo CSV',
    'sheet_not_found': 'La hoja indicada no existe en el libro',
    'header_row_out_of_range': 'La fila de cabecera está fuera del rango del archivo',
    'file_has_no_rows': 'El archivo no contiene filas',
    'file_has_no_data_rows': 'El archivo no contiene filas de datos',
    'file_has_no_columns': 'El archivo no contiene columnas',
    'no_price_column_found': 'No se encontró ninguna columna de precios',
    'asset_has_no_valid_rows': 'El activo no conserva ninguna fila válida',
    'date_column_unparseable': 'No se pudo interpretar la columna de fechas',
    'unknown_missing_method': 'Política de valores faltantes desconocida',
    'asset_count_not_positive': 'No hay activos para analizar',
    'too_few_assets': 'Se necesitan al menos dos activos alineados',
    'too_few_common_dates': 'No hay suficientes fechas comunes entre los activos',
    'too_few_returns_to_estimate':
        'La serie no tiene suficientes observaciones para estimar momentos',

    # core/optimize.py y core/frontier.py
    'no_feasible_starting_point':
        'No existe un vector de pesos factible con las restricciones actuales',
    'closed_form_min_variance_undefined':
        'La solución cerrada de mínima varianza no está definida para esta matriz',

    # core/purchase.py
    'pct_fee_not_below_one': 'La comisión porcentual debe ser inferior al 100 %',
    'purchase_inputs_length_mismatch':
        'Los pesos, los precios y los nombres no tienen la misma longitud',
    'purchase_price_not_positive': 'Hay un precio de referencia que no es positivo',
    'purchase_cash_invariant_violated':
        'Inconsistencia interna en el presupuesto del balance de compra',
    'purchase_leftover_cash_negative':
        'Inconsistencia interna: el efectivo remanente resultó negativo',
    'weight_tolerance_out_of_range': 'La tolerancia de peso está fuera del rango admitido',
    'fee_is_negative': 'Las comisiones no pueden ser negativas',

    # core/backtest.py
    'backtest_too_few_returns': 'La serie es demasiado corta para el backtesting',
    'backtest_window_out_of_range':
        'La ventana de estimación debe ser menor que el número de observaciones',
    'backtest_no_rebalance_dates': 'La regla de rebalanceo no genera ninguna fecha',
    'backtest_no_strategies': 'No se seleccionó ninguna estrategia',
    'unknown_strategy': 'Estrategia desconocida',
    'unknown_rebalance_rule': 'Regla de rebalanceo desconocida',
    'rebalance_step_not_positive': 'El paso de rebalanceo debe ser positivo',
    'rebalance_window_negative': 'La ventana de rebalanceo no puede ser negativa',

    # core/cashflow.py
    'initial_position_capital_consumed':
        'La reserva de efectivo y las comisiones consumen todo el capital inicial',
    'wealth_identity_violated': 'Inconsistencia contable interna en el motor de cuotas',
    'ledger_column_missing': 'Falta una columna en el registro de cuotas',
    'schedule_is_empty': 'El calendario de cuotas está vacío',
    'schedule_length_mismatch': 'Las columnas del calendario no tienen la misma longitud',
    'schedule_dates_not_increasing': 'Las fechas del calendario deben ser crecientes',
    'schedule_amount_invalid': 'Hay un importe de cuota inválido',
    'schedule_external_income_invalid': 'Hay un ingreso externo inválido',
    'schedule_date_invalid': 'Hay una fecha de cuota inválida',
    'schedule_column_missing': 'Falta una columna obligatoria en el calendario',
    'schedule_frame_expected': 'El calendario personalizado debe ser una tabla',
    'schedule_growth_rate_invalid': 'El crecimiento de las cuotas es inválido',
    'schedule_horizon_not_positive': 'El horizonte del calendario debe ser positivo',
    'unknown_frequency': 'Frecuencia de cuotas desconocida',
    'n_installments_not_positive': 'El número de cuotas debe ser positivo',
    'holding_period_negative': 'El periodo de mantenimiento no puede ser negativo',
    'cashflow_inputs_length_mismatch':
        'Las entradas del motor de cuotas no tienen dimensiones compatibles',
    'cashflow_price_not_positive': 'Hay un precio inicial que no es positivo',
    'cashflow_batch_not_three_dimensional':
        'El lote de trayectorias no tiene tres dimensiones',
    'cashflow_path_not_two_dimensional': 'La trayectoria no tiene dos dimensiones',
    'cashflow_batch_is_empty': 'El lote de trayectorias está vacío',
    'cashflow_horizon_not_positive': 'El horizonte de simulación debe ser positivo',
    'cashflow_asset_count_mismatch':
        'El número de activos de las trayectorias no coincide con la posición inicial',
    'cashflow_payment_indices_mismatch':
        'Los índices de pago no coinciden con el calendario',
    'cashflow_payment_index_out_of_range':
        'Hay un índice de pago fuera del horizonte simulado',
    'cashflow_rebalance_index_out_of_range':
        'Hay un índice de rebalanceo fuera del horizonte simulado',
    'cashflow_rebalance_without_target_weights': 'El rebalanceo exige pesos objetivo',
    'cashflow_no_batches': 'No se generó ninguna trayectoria',
    'cash_reserve_negative': 'La reserva de efectivo no puede ser negativa',
    'unknown_sale_rule': 'Regla de venta desconocida',

    # core/scenarios.py
    'scenario_inputs_length_mismatch':
        'Las entradas de los escenarios no tienen dimensiones compatibles',
    'scenario_key_missing': 'Falta un escenario obligatorio',
    'unknown_comparator': 'Comparador desconocido',
    'unknown_path_method': 'Método de simulación de trayectorias desconocido',
    'horizon_not_positive': 'El horizonte debe ser positivo',
    'n_paths_not_positive': 'El número de trayectorias debe ser positivo',
    'covariance_shape_mismatch': 'La matriz de covarianza no tiene la forma esperada',
    'block_bootstrap_requires_historical_returns':
        'El bootstrap por bloques necesita la serie histórica de retornos',
    'block_size_out_of_range':
        'El tamaño de bloque está fuera del rango admitido o excede la serie histórica',
}


# --------------------------------------------------------------------------- #
# Warnings: DataQualityReport.warning_codes, ParseReport.warning_codes and the
# diagnostics the sections surface as amber boxes or badges
# --------------------------------------------------------------------------- #
WARNING_LABELS: dict = {
    'asset_count_not_five':
        'El universo alineado no tiene cinco activos: el análisis sigue siendo válido, '
        'pero se aparta del caso previsto',
    'zero_variance_asset':
        'Hay un activo con varianza nula en el periodo seleccionado: la optimización '
        'puede quedar degenerada',
    'returns_span_multiple_days':
        'Una parte de los retornos abarca más de un día bursátil; la anualización '
        'supone un día por fila alineada',
    'prices_not_adjusted':
        'Los precios no parecen ajustados por dividendos ni por eventos corporativos: '
        'las rentabilidades quedan subestimadas',
    'large_calendar_gap': 'Hay huecos de calendario superiores a diez días naturales',
    'header_row_not_detected':
        'No se detectó la fila de cabecera; se usó la primera fila del archivo',
    'decimal_separator_ambiguous':
        'El separador decimal es ambiguo; se aplicó el valor por defecto en español',
    'psd_repaired':
        'La matriz de covarianza no era definida positiva y se reparó con un ridge',
    'weight_sampling_failed_relax_weight_cap_or_short_bound':
        'No se pudo muestrear una nube de pesos factible: relaja el límite máximo por '
        'activo o la cota inferior de ventas en corto',
    'weight_sampling_biased':
        'La tasa de aceptación del muestreo es baja: la nube puede estar sesgada',
    'optimization_not_certified':
        'El optimizador no certificó la solución: no se presenta como óptimo',
    'first_rebalance_equal_weights':
        'En el primer rebalanceo no hubo óptimo verificado y se usó el equiponderado',
    'rebalance_kept_previous_weights':
        'En este rebalanceo no hubo óptimo verificado y se mantuvieron los pesos previos',
    'tangency_undefined_rf_above_all_assets':
        'La tangencia no está definida: la tasa libre de riesgo supera la rentabilidad '
        'máxima alcanzable, por lo que se devuelve la cartera de mínima varianza',
    'stale_results': 'Resultados desactualizados — vuelve a calcular',
    'local_value': 'valor local',
}


# --------------------------------------------------------------------------- #
# Notes: the six codes of core.types.NOTE_CODES plus the two budget notes of
# core/purchase.py
# --------------------------------------------------------------------------- #
NOTE_LABELS: dict = {
    'sale_not_executed_costs_exceed_proceeds':
        'Venta no ejecutada: los costos superaban el producto de la venta',
    'rebalance_skipped_no_market_value':
        'Rebalanceo omitido: no había valor de mercado que recomponer',
    'rebalance_skipped_cost_exceeds_value':
        'Rebalanceo omitido: el costo superaba el valor de la cartera',
    'price_above_assignable_budget':
        'Precio superior al presupuesto asignable: cero unidades',
    'two_installments_same_business_day':
        'Dos cuotas caen en el mismo día hábil y se agregaron en una sola',
    'no_scheduled_obligations':
        'El calendario no tiene obligaciones: no se emite veredicto de capacidad',
    'budget_non_positive': 'El presupuesto disponible no es positivo',
    'notional_non_positive':
        'La comisión fija consume el presupuesto: el nocional invertible no es positivo',
}


# --------------------------------------------------------------------------- #
# Excel sheet titles, 31 characters at most (design.md section 16)
# --------------------------------------------------------------------------- #
SHEET_NAMES: dict = {
    'parameters': 'Parametros',
    'summary': 'Resumen',
    'weights': 'Pesos',
    'asset_stats': 'EstadisticasActivos',
    'covariance': 'Covarianza',
    'correlation': 'Correlacion',
    'efficient_frontier': 'FronteraEficiente',
    'monte_carlo_weights': 'MonteCarloPesos',
    'purchase_balance': 'BalanceCompra',
    'backtest': 'Backtesting',
    'backtest_series': 'BacktestingSerie',
    'installment_schedule': 'CalendarioCuotas',
    'initial_position': 'PosicionInicial',
    'scenarios': 'Escenarios',
    'monte_carlo_capacity': 'MonteCarloCuotas',
}


# --------------------------------------------------------------------------- #
# Column headers. Scope: screen AND export (design.md sections 2.1 and 16)
# --------------------------------------------------------------------------- #
COLUMN_LABELS: dict = {
    # DataQualityReport.per_asset
    'asset': 'Activo',
    'rows_raw': 'Filas originales',
    'invalid': 'Descartadas',
    'duplicates': 'Duplicadas',
    'first': 'Primera fecha',
    'last': 'Última fecha',
    'max_gap_days': 'Hueco máximo (días)',
    'lost_in_alignment': 'Perdidas al alinear',

    # ParseReport
    'file_name': 'Archivo',
    'sheet_used': 'Hoja',
    'header_row_used': 'Fila de cabecera',
    'csv_separator': 'Separador CSV',
    'decimal_detected': 'Separador decimal',
    'date_format_applied': 'Formato de fecha',
    'date_format_source': 'Origen del formato',
    'nat_dropped': 'Fechas descartadas',
    'ohlcv_collapsed': 'OHLCV detectado',
    'detected_format': 'Formato detectado',

    # Asset statistics
    'n_obs': 'Observaciones',
    'annual_return': 'Rentab. anual',
    'annual_volatility': 'Volat. anual',
    'sharpe': 'Sharpe',
    'min_return': 'Mínimo',
    'max_return': 'Máximo',
    'mean_correlation': 'Correlación media',

    # Portfolio summary
    'strategy': 'Estrategia',
    'weight': 'Peso',
    'expected_return': 'Rentab. esperada',
    'expected_return_annual': 'Rentab. anual esperada',
    'volatility': 'Volatilidad',
    'volatility_annual': 'Volatilidad anual',
    'n_positive': 'Activos con posición positiva',
    'certified': 'Certificado',
    'converged': 'Convergió',

    # Frontier and weight cloud
    'index': 'N.º',
    'vol': 'Volatilidad',
    'ret': 'Rentabilidad',

    # Purchase balance (core.purchase.ROW_COLUMNS)
    'target_weight': 'Peso objetivo',
    'target_value': 'Valor objetivo',
    'price': 'Precio de referencia',
    'current_units': 'Unidades actuales',
    'current_weight': 'Peso actual',
    'target_units': 'Unidades objetivo',
    'delta_units': 'Unidades a operar',
    'action': 'Acción',
    'gross_amount': 'Importe bruto',
    'costs': 'Costos',
    'final_value': 'Valor final',
    'final_weight': 'Peso final',
    'deviation': 'Desviación',
    'note_code': 'Nota',
    'buy': 'COMPRAR',
    'sell': 'VENDER',
    'hold': 'MANTENER',

    # Purchase totals
    'capital': 'Capital',
    'current_value': 'Valor actual de la cartera',
    'total_investable': 'Total invertible',
    'notional': 'Nocional asignable',
    'invested': 'Importe invertido',
    'buy_gross': 'Compras brutas',
    'sell_proceeds': 'Producto de las ventas',
    'total_costs': 'Costos totales',
    'leftover_cash': 'Efectivo remanente',
    'max_deviation': 'Desviación máxima',
    'feasible': 'Factible',
    'cash_reserve': 'Reserva de efectivo',
    'weight_tolerance': 'Tolerancia de peso',
    'integer_units': 'Unidades enteras',

    # Backtesting
    'total_return': 'Rentabilidad total',
    'cagr': 'CAGR',
    'mu_cc_real': 'Rentab. log anualizada',
    'vol_real': 'Volatilidad realizada',
    'sharpe_real': 'Sharpe realizado',
    'degenerate': 'Serie degenerada',
    'max_drawdown': 'Máximo drawdown',
    'final_capital': 'Capital final',
    'n_rebalances': 'Rebalanceos',
    'n_observations': 'Observaciones',
    'date': 'Fecha',
    'window': 'Ventana de estimación',
    'rebalance_rule': 'Regla de rebalanceo',
    'initial_capital': 'Capital inicial',
    'cost_rate': 'Costo por turnover',

    # Installment ledger (core.cashflow.LEDGER_COLUMNS)
    'due': 'Cuota exigible',
    'wealth_open': 'Patrimonio de apertura',
    'returns_in_period': 'Rendimientos del tramo',
    'wealth_before': 'Capital antes del pago',
    'cash_before': 'Efectivo antes del pago',
    'mv_before': 'Inversión antes del pago',
    'external_income': 'Ingreso externo',
    'gross_sold': 'Importe retirado de la inversión',
    'sale_costs': 'Costos de venta',
    'rebalance_costs_pre': 'Costos de recomposición (previos)',
    'rebalance_costs_post': 'Costos de recomposición (posteriores)',
    'rebalance_costs': 'Costos de recomposición',
    'paid': 'Pago cubierto',
    'deficit': 'Déficit',
    'cash_after': 'Efectivo tras el pago',
    'mv_after': 'Inversión tras el pago',
    'wealth_after': 'Capital restante',
    'covered': 'Estado',

    # Initial position
    'units': 'Unidades',
    'value': 'Valor',
    'purchase_costs': 'Costos de compra',
    'cash': 'Efectivo',
    'wealth_initial': 'Patrimonio inicial',
    'position_source': 'Origen de la posición',
    'from_capital': 'Construida desde el capital',
    'from_balance': 'Tomada del balance de compra',

    # Capacity metrics and verdict
    'item': 'Concepto',
    'threshold': 'Umbral',
    'comparator': 'Comparador',
    'passed': 'Cumple',
    'scenario': 'Escenario',
    'covered_all': 'Cubre todas las cuotas',
    'n_defaults': 'Cuotas incumplidas',
    'final_wealth': 'Capital final',
    'min_wealth': 'Capital mínimo',
    'total_paid': 'Total pagado',
    'cumulative_deficit': 'Déficit acumulado',
    'total_gross_sold': 'Importe bruto vendido',
    'total_scheduled': 'Total de cuotas programadas',
    'n_installments': 'Número de cuotas',
    'p_cover_all': 'Trayectorias que cubren todas las cuotas',
    'margin_ratio': 'Margen de capital final',
    'central_covers_all': 'El escenario central cubre todas las cuotas',
    'conservative_covers_all': 'El escenario conservador cubre todas las cuotas',
    'conservative_deficit_ratio': 'Déficit relativo en el escenario conservador',
    'additional_capital_required': 'Capital adicional necesario',
    'coverage_by_installment': 'Cobertura de la cuota',
    'category': 'Categoría',
    'rule_fired': 'Regla aplicada',
    'p5': 'Percentil 5',
    'p25': 'Percentil 25',
    'p50': 'Percentil 50 (mediana)',
    'p75': 'Percentil 75',
    'p95': 'Percentil 95',
    'method': 'Método de simulación',
    'n_paths': 'Trayectorias simuladas',
    'horizon': 'Horizonte (días bursátiles)',
    'n_assets': 'Número de activos',
    'batch_size': 'Tamaño de lote',
    'block_size': 'Tamaño de bloque',
    'psd_repaired': 'Covarianza reparada',
    'k_sigma': 'Factor k de los escenarios',

    # Parameters sheet and sidebar
    'parameter': 'Parámetro',
    'owner': 'Propietario',
    'sidebar': 'Barra lateral',
    'section': 'Sección',
    'app_version': 'Versión de la aplicación',
    'generated_at': 'Fecha de generación',
    'missing_sheets': 'Secciones sin calcular',
    'not_computed': 'no calculado',
    'sheet_error': 'No se pudo construir esta hoja',
    'monte_carlo_rows_truncated': 'Filas truncadas',
    'rf_annual_effective': 'Tasa libre de riesgo anual efectiva',
    'trading_days': 'Días bursátiles por año',
    'allow_short': 'Ventas en corto',
    'max_weight': 'Límite máximo por activo',
    'short_lower_bound': 'Cota inferior con ventas en corto',
    'n_weight_sims': 'Simulaciones Monte Carlo de pesos',
    'n_frontier_points': 'Puntos de la frontera eficiente',
    'pct_fee': 'Comisión porcentual',
    'fixed_fee': 'Comisión fija por operación',
    'seed': 'Semilla',
    'currency_symbol': 'Símbolo monetario',
    'period_start': 'Inicio del periodo',
    'period_end': 'Fin del periodo',
    'missing_method': 'Política de valores faltantes',
    'limit_days': 'Límite de días de relleno',
    'holding_period_days': 'Periodo de mantenimiento',
    'valuation_date': 'Fecha de valoración',
    'horizon_end': 'Fin del horizonte',
    'cash_reserve_initial': 'Reserva inicial de efectivo',
    'sale_rule': 'Regla de venta',
    'allow_sales': 'Permitir ventas',
    'cash_interest': 'Remunerar el efectivo',
    'p_sufficient': 'Umbral de capacidad suficiente',
    'p_tight': 'Umbral de capacidad ajustada',
    'p_high_risk': 'Umbral de riesgo elevado',
    'margin_min': 'Margen mínimo exigido',
    'conservative_deficit_tolerance': 'Tolerancia de déficit conservador',
}

EXPORT_LABELS: dict = {}   # filled at the end of the module


PREVIEW_COLUMN_LABELS: dict = {
    'resulting_obs': 'Observaciones resultantes',
    'filled_cells': 'Celdas rellenadas',
    'lost_dates': 'Fechas perdidas',
    'drop': 'Descartar fechas incompletas',
    'ffill': 'Arrastrar el último precio',
    'interpolate': 'Interpolar linealmente',
}


DATE_FORMAT_LABELS: dict = {
    'auto': 'Automático',
    'iso': 'ISO (aaaa-mm-dd)',
    'day_first': 'dd/mm/aaaa',
    'month_first': 'mm/dd/aaaa',
    'excel_serial': 'Serial de Excel',
    'detected': 'Detectado',
    'chosen': 'Valor por defecto en español',
    'forced': 'Forzado por el usuario',
    'comma': 'Coma',
    'dot': 'Punto',
}


SCHEDULE_COLUMN_LABELS: dict = {
    'date': 'Fecha de pago',
    'amount': 'Importe de la cuota',
    'external_income': 'Ingreso externo',
}


FREQUENCY_LABELS: dict = {
    'monthly': 'Mensual',
    'quarterly': 'Trimestral',
    'semiannual': 'Semestral',
    'annual': 'Anual',
}


REBALANCE_LABELS: dict = {
    'W': 'Semanal',
    'M': 'Mensual',
    'Q': 'Trimestral',
    'S': 'Semestral',
    'A': 'Anual',
    'Nd': 'Cada N días bursátiles',
}


SCENARIO_LABELS: dict = {
    'conservative': 'Conservador',
    'central': 'Central',
    'favorable': 'Favorable',
}


# The four literals of RF-C21, exactly as the requirement states them.
CATEGORY_LABELS: dict = {
    'sufficient': 'Capacidad de pago suficiente bajo los supuestos analizados',
    'tight': 'Capacidad de pago ajustada',
    'high_risk': 'Riesgo elevado de insuficiencia de liquidez',
    'insufficient': 'Insuficiencia de capital bajo el escenario evaluado',
}


STRATEGY_LABELS: dict = {
    'min_variance': 'Mínima varianza',
    'max_sharpe': 'Máxima Sharpe',
    'equal_weight': 'Equiponderado',
}


# core/export.py receives a single mapping as ``column_labels`` and resolves
# through it the column headers, the strategy and scenario keys, the note codes
# and the parameter names. Later entries win on a collision, and the only
# collision is harmless ('date' as a ledger header vs. as a schedule header).
EXPORT_LABELS.update(COLUMN_LABELS)
EXPORT_LABELS.update(STRATEGY_LABELS)
EXPORT_LABELS.update(SCENARIO_LABELS)
EXPORT_LABELS.update(CATEGORY_LABELS)
EXPORT_LABELS.update(NOTE_LABELS)
EXPORT_LABELS.update(PREVIEW_COLUMN_LABELS)


SECTIONS: tuple = (
    '1. Inicio y carga de datos',
    '2. Análisis de activos',
    '3. Optimización de portafolios',
    '4. Frontera eficiente y simulaciones',
    '5. Balance de compra',
    '6. Backtesting',
    '7. Capacidad de pago de cuotas',
)
