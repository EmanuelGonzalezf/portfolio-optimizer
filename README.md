# Portfolio Optimizer

Aplicación Streamlit para optimizar portafolios de cinco activos a partir de precios históricos
cargados localmente, evaluarlos con backtesting walk-forward y analizar si un capital inicial
invertido permite cubrir un calendario de cuotas futuras.

No requiere API keys ni servicios externos: trabaja con archivos Excel o CSV propios.

## Requisitos

- Python 3.10 o superior (desarrollado y probado con 3.10.9).
- Windows, macOS o Linux. Los comandos de abajo están en PowerShell.

## Instalación

```powershell
cd portfolio_optimizer
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

En macOS o Linux, sustituye `.\.venv\Scripts\python.exe` por `./.venv/bin/python`.

Versiones fijadas en `requirements.txt`: streamlit 1.40.2, numpy 2.1.3, pandas 2.2.3,
scipy 1.14.1, plotly 5.24.1, openpyxl 3.1.5, pytest 8.3.4, xlrd 2.0.1.

## Ejecución

```powershell
cd portfolio_optimizer
streamlit run app.py
```

## Datos de ejemplo

Si no tienes un archivo propio a mano, genera una serie sintética de cinco activos:

```powershell
.\.venv\Scripts\python.exe data\generate_sample_data.py
```

El script escribe el archivo de ejemplo dentro de `data/`, listo para cargarlo desde la
sección 1 de la aplicación.

## Pruebas

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Las pruebas usan datos sintéticos con semilla fija, así que no necesitan ningún archivo tuyo.
Cubren, entre otras cosas, el cálculo de retornos y covarianzas, la factibilidad de los pesos
óptimos, la ausencia de sesgo de anticipación en el backtesting, los costos de transacción, el
balance de compra dentro del presupuesto, los siete casos obligatorios de capacidad de pago, la
reproducibilidad con semilla y la exportación a Excel.

## Formatos de carga admitidos

- **Formato A**: un archivo con una columna de fechas y cinco columnas de precios de cierre.
- **Formato B**: un archivo por activo, cada uno con fecha y precio de cierre.

La aplicación detecta las columnas automáticamente y permite corregir la asignación a mano.
Ordena cronológicamente, señala fechas duplicadas y precios inválidos, alinea las series por
fechas comunes e informa cuántas observaciones quedan. Los valores faltantes se tratan según la
política que elijas, mostrando antes el efecto de cada opción: en ningún caso un dato inválido se
convierte silenciosamente en una rentabilidad de cero.

Usa precios de cierre **ajustados** cuando los tengas. Si tus precios no incorporan dividendos ni
ajustes por eventos corporativos, la aplicación lo advierte: la rentabilidad quedará subestimada.

## Convenciones financieras

- **Retornos logarítmicos diarios** como espacio de cálculo interno. Las cifras monetarias se
  capitalizan por activo con `exp(r)`; los porcentajes que se muestran al usuario se convierten a
  retorno simple.
- **Anualización** con el número de días bursátiles configurable en la barra lateral (252 por
  defecto): media × días para la rentabilidad, `sqrt(días)` para la volatilidad.
- **Tasa libre de riesgo**: se introduce como tasa anual efectiva y se convierte a composición
  continua con `ln(1 + rf)` antes de entrar en el numerador del Sharpe. Es una desviación
  consciente respecto al prototipo original, que asumía la tasa ya compuesta continuamente, de
  modo que las cifras no son comparables cifra a cifra con él.
- **Optimización** con SLSQP (`scipy.optimize.minimize`) y jacobianos analíticos, multiarranque y
  comprobación explícita de convergencia y factibilidad. Si no se alcanza un óptimo verificado, la
  interfaz lo dice en lugar de presentar el resultado como certificado.

## Backtesting

Walk-forward: en cada fecha de decisión los pesos se estiman **solo** con la ventana histórica
previa y se aplican al periodo siguiente. El tamaño de la ventana, la frecuencia de rebalanceo, el
capital inicial, los costos de transacción y el periodo de evaluación son configurables. Las tres
estrategias (máxima razón de Sharpe, mínima varianza y equiponderado) se comparan sobre el mismo
periodo y el mismo capital inicial.

## Capacidad de pago de cuotas

La sección 7 responde a si el capital invertido alcanza para cubrir un calendario de cuotas.
Distingue explícitamente el **valor de mercado** de las posiciones del **efectivo disponible**:
la rentabilidad de un activo no es dinero disponible hasta que se vende.

En cada fecha de pago, por este orden: se actualiza el valor de mercado según el escenario, se
aplican los costos de rebalanceo que correspondan, se suman los ingresos externos declarados, se
exige la cuota, y si el efectivo no alcanza se venden posiciones de forma pro-rata resolviendo el
**importe bruto** necesario para cubrir la cuota más los costos de venta. Si no se puede cubrir,
se registra el incumplimiento y el escenario no sigue generando rentabilidad sobre capital
inexistente.

Se evalúan tres escenarios (conservador, central y favorable) más una simulación Monte Carlo de
trayectorias futuras que preserva la dependencia entre activos mediante descomposición de Cholesky
de la matriz de covarianza, con bootstrap por bloques como alternativa. El número de trayectorias y
la semilla son configurables, y la semilla hace la simulación reproducible.

El veredicto final se clasifica en cuatro categorías (capacidad suficiente, capacidad ajustada,
riesgo elevado de insuficiencia de liquidez, insuficiencia de capital). Los umbrales numéricos de
cada categoría se muestran en la propia interfaz junto al resultado, para que puedas ver por qué se
asignó una u otra.

## Advertencia

Los escenarios y las simulaciones son **supuestos de modelo**, no predicciones. El porcentaje de
trayectorias Monte Carlo que cubren todas las cuotas es una estimación condicionada a la muestra
histórica utilizada, a los supuestos estadísticos del método (independencia e identidad de
distribución de los retornos diarios, estabilidad de la matriz de covarianza) y al número de
trayectorias. No es una probabilidad real conocida ni una garantía de pago.

Las cuotas son obligaciones que siguen venciendo aunque la inversión pierda valor. Esta aplicación
es una herramienta de análisis académico, no asesoramiento financiero.

## Estructura

```
portfolio_optimizer/
  app.py                  navegación y punto de entrada
  core/                   cálculo puro, sin Streamlit
    types.py  rng.py  formatting.py  stats.py  data_loader.py
    optimize.py  frontier.py  purchase.py  backtest.py
    cashflow.py  scenarios.py  export.py
  ui/                     presentación
    theme.py  charts.py  labels.py  common.py  compute.py
    narrative.py  sidebar.py  state.py
    sections/             una sección por pestaña (s1_data … s7_capacity)
  data/generate_sample_data.py
  tests/
```

Cada fórmula financiera vive en un único sitio dentro de `core/`; la interfaz solo la consume.
Los textos visibles están en español y se resuelven en `ui/labels.py`; el código, incluidas las
claves internas de los DataFrames que produce `core/`, está en inglés.
