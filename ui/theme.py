"""Level 2 of the theme: CSS injected once per run (design.md section 12.1).

Level 1 lives in ``.streamlit/config.toml`` and is the safety net: if a
``data-testid`` selector stops applying after a Streamlit upgrade, the app
stays readable because the base colours are already right. The high-value
components (header, panels, section headers, metric cards, status boxes,
badges) are built with our own HTML, so they do not depend on internal
Streamlit classes at all.

Every hex value, gradient, radius and font size here is taken literally from
the prototype (requirements.md sections 2.1 to 2.3).
"""

from __future__ import annotations

from contextlib import contextmanager

import streamlit as st

__all__ = [
    'PALETTE',
    'GRADIENTS',
    'SHADOW',
    'BODY_BG',
    'FONT_STACK',
    'CSS',
    'inject_theme',
    'render_header',
    'panel',
    'section_header',
    'metric_card',
    'status_box',
    'badge',
    'small_note',
]

PALETTE = {
    'bg_0': '#070a13', 'bg_1': '#0d1320', 'bg_2': '#141c2e', 'bg_3': '#1c2740',
    'border': '#26304a', 'border_soft': '#1d2740',
    'text': '#e6edf3', 'text_dim': '#8b96a8', 'text_mute': '#5b6478',
    'accent': '#4ade80', 'accent_2': '#38bdf8', 'accent_3': '#a78bfa',
    'warn': '#fbbf24', 'err': '#f87171',
}

GRADIENTS = {
    'grad_1': 'linear-gradient(135deg,#4ade80 0%,#38bdf8 100%)',
    'grad_2': 'linear-gradient(135deg,#a78bfa 0%,#38bdf8 100%)',
    'grad_3': 'linear-gradient(135deg,#fbbf24 0%,#f87171 100%)',
}

SHADOW = '0 8px 32px rgba(0,0,0,.4)'
BODY_BG = 'radial-gradient(ellipse at top,#0d1424 0%,#070a13 55%)'
FONT_STACK = ('-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, '
              'sans-serif')

# Status boxes: the three exact colour sets of the prototype.
STATUS_COLOURS = {
    'info': ('rgba(56,189,248,.08)', 'rgba(56,189,248,.3)', '#a5d8ff'),
    'ok': ('rgba(74,222,128,.08)', 'rgba(74,222,128,.3)', '#86efac'),
    'err': ('rgba(248,113,113,.08)', 'rgba(248,113,113,.3)', '#fca5a5'),
}

# Metric card top bar: default, Sharpe and volatility variants.
CARD_VARIANTS = {
    'default': GRADIENTS['grad_1'],
    'sharpe': GRADIENTS['grad_2'],
    'vol': GRADIENTS['grad_3'],
}


CSS = f"""
<style>
:root {{
  --bg-0:{PALETTE['bg_0']}; --bg-1:{PALETTE['bg_1']}; --bg-2:{PALETTE['bg_2']};
  --bg-3:{PALETTE['bg_3']}; --border:{PALETTE['border']};
  --border-soft:{PALETTE['border_soft']}; --text:{PALETTE['text']};
  --text-dim:{PALETTE['text_dim']}; --text-mute:{PALETTE['text_mute']};
  --accent:{PALETTE['accent']}; --accent-2:{PALETTE['accent_2']};
  --accent-3:{PALETTE['accent_3']}; --warn:{PALETTE['warn']}; --err:{PALETTE['err']};
  --grad-1:{GRADIENTS['grad_1']}; --grad-2:{GRADIENTS['grad_2']};
  --grad-3:{GRADIENTS['grad_3']}; --shadow:{SHADOW};
}}

/* body radial gradient */
[data-testid="stAppViewContainer"] {{
  background: {BODY_BG};
  font-family: {FONT_STACK};
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stAppViewContainer"] * {{ font-family: {FONT_STACK}; }}
.block-container {{ padding-top: 1.2rem; max-width: 1500px; }}

/* fixed header with blur */
.po-header {{
  position: sticky; top: 0; z-index: 999;
  background: rgba(13,19,32,.7);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid {PALETTE['border']};
  padding: 12px 4px; margin-bottom: 14px;
}}
.po-header h1 {{
  font-size: 22px; font-weight: 700; letter-spacing: -.01em; margin: 0;
  display: flex; align-items: center; gap: 10px;
  background: {GRADIENTS['grad_1']};
  -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.po-header h1::before {{
  content: ""; width: 10px; height: 10px; border-radius: 50%;
  background: {PALETTE['accent']}; box-shadow: 0 0 12px {PALETTE['accent']};
  flex: 0 0 auto;
}}
.po-header .po-sub {{
  font-size: 11px; text-transform: uppercase; letter-spacing: .05em;
  font-weight: 600; color: {PALETTE['text_dim']}; margin-top: 4px;
}}

/* panel */
.po-panel {{
  background: {PALETTE['bg_1']}; border: 1px solid {PALETTE['border']};
  border-radius: 14px; padding: 22px; box-shadow: {SHADOW}; margin-bottom: 16px;
}}
[data-testid="stVerticalBlockBorderWrapper"] {{
  background: {PALETTE['bg_1']}; border-radius: 14px;
}}
[data-testid="stVerticalBlockBorderWrapper"] > div {{ border-color: {PALETTE['border']}; }}

/* panel h2 with the 4x14 bar */
.po-h2 {{
  display: flex; align-items: center; gap: 8px;
  font-size: 14px; font-weight: 600; letter-spacing: .06em;
  text-transform: uppercase; color: {PALETTE['text_dim']}; margin: 0 0 14px 0;
}}
.po-h2 .bar {{
  width: 4px; height: 14px; border-radius: 2px;
  background: {GRADIENTS['grad_1']}; display: inline-block; flex: 0 0 auto;
}}

/* metric card */
.po-card {{
  background: linear-gradient(180deg,{PALETTE['bg_2']} 0%,{PALETTE['bg_1']} 100%);
  border: 1px solid {PALETTE['border']}; border-radius: 12px; padding: 18px;
  position: relative; overflow: hidden; min-height: 104px;
}}
.po-card::before {{
  content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: {GRADIENTS['grad_1']};
}}
.po-card.sharpe::before {{ background: {GRADIENTS['grad_2']}; }}
.po-card.vol::before {{ background: {GRADIENTS['grad_3']}; }}
.po-card .po-label {{
  font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
  font-weight: 600; color: {PALETTE['text_dim']}; margin-bottom: 10px;
}}
.po-card .po-value {{
  font-size: 28px; font-weight: 700; letter-spacing: -.02em;
  font-variant-numeric: tabular-nums; color: {PALETTE['text']};
}}
.po-card .po-value.pos {{ color: {PALETTE['accent']}; }}
.po-card .po-value.neg {{ color: {PALETTE['err']}; }}
.po-card .po-sub {{ font-size: 12px; color: {PALETTE['text_mute']}; margin-top: 4px; }}

/* strategy pills over st.radio */
div[role="radiogroup"] {{ gap: 4px; }}
[data-testid="stHorizontalBlock"] div[role="radiogroup"],
.po-pills div[role="radiogroup"] {{
  background: {PALETTE['bg_2']}; padding: 5px; border-radius: 10px;
  border: 1px solid {PALETTE['border']}; width: fit-content;
}}
div[role="radiogroup"] label {{
  border-radius: 7px; padding: 8px 16px; color: {PALETTE['text_dim']};
  font-weight: 600; margin: 0;
}}
div[role="radiogroup"] label:has(input:checked) {{
  background: {PALETTE['bg_3']}; color: {PALETTE['text']};
}}
div[role="radiogroup"] label > div:first-child {{ display: none; }}

/* chart container */
.po-chart {{
  background: {PALETTE['bg_1']}; border: 1px solid {PALETTE['border_soft']};
  border-radius: 10px; padding: 8px;
}}
[data-testid="stPlotlyChart"] {{
  background: {PALETTE['bg_1']}; border: 1px solid {PALETTE['border_soft']};
  border-radius: 10px; padding: 8px;
}}

/* tables */
[data-testid="stDataFrame"], [data-testid="stTable"] {{
  font-size: 13px; border-radius: 10px; overflow: hidden;
}}
[data-testid="stDataFrame"] th, [data-testid="stTable"] th {{
  background: {PALETTE['bg_2']}; color: {PALETTE['text_dim']};
  text-transform: uppercase; font-size: 11px; letter-spacing: .05em;
  padding: 10px 12px;
}}
[data-testid="stDataFrame"] td, [data-testid="stTable"] td {{
  border-bottom: 1px solid {PALETTE['border_soft']};
  font-variant-numeric: tabular-nums; padding: 10px 12px;
}}
[data-testid="stDataFrame"] tr:hover td, [data-testid="stTable"] tr:hover td {{
  background: rgba(56,189,248,.03);
}}

/* status boxes */
.po-status {{
  border-radius: 9px; padding: 12px 14px; font-size: 13px; margin: 6px 0;
  border: 1px solid transparent;
}}
.po-status.info {{
  background: {STATUS_COLOURS['info'][0]}; border-color: {STATUS_COLOURS['info'][1]};
  color: {STATUS_COLOURS['info'][2]};
}}
.po-status.ok {{
  background: {STATUS_COLOURS['ok'][0]}; border-color: {STATUS_COLOURS['ok'][1]};
  color: {STATUS_COLOURS['ok'][2]};
}}
.po-status.err {{
  background: {STATUS_COLOURS['err'][0]}; border-color: {STATUS_COLOURS['err'][1]};
  color: {STATUS_COLOURS['err'][2]};
}}

/* badge */
.po-badge {{
  display: inline-block; padding: 2px 8px; border-radius: 6px;
  background: {PALETTE['bg_3']}; color: {PALETTE['text_dim']};
  font-size: 11px; font-weight: 600; letter-spacing: .04em; margin-right: 6px;
}}

/* small note */
.po-note {{ font-size: 11px; color: {PALETTE['text_mute']}; margin-top: 6px; }}

/* buttons */
button[kind="primary"], button[data-testid="baseButton-primary"] {{
  background: {GRADIENTS['grad_1']}; color: #06121a; border: none;
  border-radius: 9px; font-weight: 600;
}}
button[kind="secondary"], button[data-testid="baseButton-secondary"] {{
  background: {PALETTE['bg_3']}; border: 1px solid {PALETTE['border']};
  color: {PALETTE['text']}; border-radius: 9px; font-weight: 600;
}}
[data-testid="stDownloadButton"] button {{
  background: {PALETTE['bg_3']}; border: 1px solid {PALETTE['border']};
  color: {PALETTE['text']}; border-radius: 9px; font-weight: 600;
}}

/* uploader dropzone */
[data-testid="stFileUploaderDropzone"] {{
  background: {PALETTE['bg_2']}; border: 2px dashed {PALETTE['border']};
  border-radius: 12px; padding: 36px;
}}
[data-testid="stFileUploaderDropzone"]:hover {{
  border-color: {PALETTE['accent']}; background: rgba(74,222,128,.05);
}}

/* inputs and selects */
.stNumberInput input, .stTextInput input, .stDateInput input,
.stSelectbox div[data-baseweb="select"], .stMultiSelect div[data-baseweb="select"] {{
  background: {PALETTE['bg_2']}; border: 1px solid {PALETTE['border']};
  border-radius: 8px; color: {PALETTE['text']};
}}
.stNumberInput div:focus-within, .stTextInput div:focus-within,
.stSelectbox div:focus-within {{ border-color: {PALETTE['accent']}; }}

/* scrollbar */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: {PALETTE['bg_1']}; }}
::-webkit-scrollbar-thumb {{ background: {PALETTE['bg_3']}; border-radius: 5px; }}
::-webkit-scrollbar-thumb:hover {{ background: #33405e; }}

/* sidebar */
[data-testid="stSidebar"] {{
  background: {PALETTE['bg_1']}; border-right: 1px solid {PALETTE['border']};
}}
[data-testid="stSidebar"] label, [data-testid="stWidgetLabel"] label p {{
  font-size: 11px; text-transform: uppercase; letter-spacing: .05em;
  font-weight: 600; color: {PALETTE['text_dim']};
}}
[data-testid="stSidebar"] div[role="radiogroup"] label {{
  text-transform: none; letter-spacing: 0; font-size: 13px;
}}
</style>
"""


def inject_theme() -> None:
    """Inject the whole stylesheet once. Called a single time from ``app.py``."""
    st.markdown(CSS, unsafe_allow_html=True)


def render_header(subtitle: str = '') -> None:
    """Sticky product header. The product name is never translated."""
    sub = f'<div class="po-sub">{subtitle}</div>' if subtitle else ''
    st.markdown(f'<div class="po-header"><h1>Portfolio Optimizer</h1>{sub}</div>',
                unsafe_allow_html=True)


def section_header(text: str) -> None:
    """Panel heading with the 4x14 gradient bar of the prototype."""
    st.markdown(f'<div class="po-h2"><span class="bar"></span>{text}</div>',
                unsafe_allow_html=True)


@contextmanager
def panel(title: str = ''):
    """Context manager that renders a prototype panel around its body."""
    container = st.container(border=True)
    with container:
        if title:
            section_header(title)
        yield container


def metric_card(label: str, value: str, sub: str = '', variant: str = 'default',
                sign: str = '') -> None:
    """Metric card in our own HTML.

    ``st.metric`` is not used on purpose: it admits neither the gradient top bar
    nor the type sizes of the prototype without hacking internal classes.
    ``sign`` is the ``'pos'``/``'neg'``/``''`` class of
    ``core.formatting.sign_class``.
    """
    variant_class = variant if variant in CARD_VARIANTS else 'default'
    sub_html = f'<div class="po-sub">{sub}</div>' if sub else ''
    st.markdown(
        f'<div class="po-card {variant_class}">'
        f'<div class="po-label">{label}</div>'
        f'<div class="po-value {sign}">{value}</div>'
        f'{sub_html}</div>',
        unsafe_allow_html=True)


def status_box(msg: str, kind: str = 'info') -> None:
    """Status box with the exact colour sets of the prototype."""
    css_class = kind if kind in STATUS_COLOURS else 'info'
    st.markdown(f'<div class="po-status {css_class}">{msg}</div>',
                unsafe_allow_html=True)


def badge(text: str, color: str | None = None) -> str:
    """Return the badge markup. Several badges usually share one ``st.markdown``."""
    style = f' style="color:{color}"' if color else ''
    return f'<span class="po-badge"{style}>{text}</span>'


def small_note(text: str) -> None:
    """11 px muted note under a control or a chart."""
    st.markdown(f'<div class="po-note">{text}</div>', unsafe_allow_html=True)
