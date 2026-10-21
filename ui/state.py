"""Session keys, per-result fingerprints and the parameter ownership rule.

Design references: section 13.1 (key table and ownership rule) and section
13.1b (visible invalidation, RF-P4).

``compute_fingerprint`` is a pure function: it never touches
``st.session_state``, which is what makes it testable in
``tests/test_formatting.py`` without launching the app.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass

import numpy as np
import pandas as pd
import streamlit as st

__all__ = [
    'K_PARAMS', 'K_FINGERPRINTS', 'K_UPLOADED', 'K_OVERRIDES', 'K_PARSE_PREFS',
    'K_PARSE_REPORTS', 'K_MISSING_POLICY', 'K_INITIAL_POSITION', 'K_THRESHOLDS',
    'K_ALIGNED', 'K_STATS', 'K_OPTIMIZATION', 'K_PATH_SPEC', 'K_FRONTIER',
    'K_CLOUD', 'K_PURCHASE_CONFIG', 'K_PURCHASE_BALANCE', 'K_CURRENT_POSITIONS',
    'K_BACKTEST_CONFIG', 'K_BACKTEST_RESULT', 'K_CAPACITY_INPUTS',
    'K_CAPACITY_SCHEDULE', 'K_SCHEDULE_IS_CUSTOM', 'K_CAPACITY_METRICS',
    'K_CAPACITY_VERDICT', 'K_ACTIVE_SECTION',
    'STATE_KEYS', 'RESULT_KEYS', 'FINGERPRINT_DEPS', 'SIDEBAR_ONLY_PARAMS',
    'init_state', 'compute_fingerprint', 'store_fingerprint', 'stored_fingerprint',
    'is_stale', 'invalidate', 'clear_results', 'effective_value', 'result_ready',
]

# --------------------------------------------------------------------------- #
# Session keys (design.md section 13.1)
# --------------------------------------------------------------------------- #
K_PARAMS = 'params'
K_FINGERPRINTS = 'fingerprints'
K_UPLOADED = 'uploaded_files'
K_OVERRIDES = 'column_overrides'
K_PARSE_PREFS = 'parse_prefs'
K_PARSE_REPORTS = 'parse_reports'
K_MISSING_POLICY = 'missing_policy'
K_INITIAL_POSITION = 'initial_position'
K_THRESHOLDS = 'capacity_thresholds'
K_ALIGNED = 'aligned_data'
K_STATS = 'asset_stats'
K_OPTIMIZATION = 'optimization'
K_PATH_SPEC = 'path_spec'
K_FRONTIER = 'frontier'
K_CLOUD = 'weight_cloud'
K_PURCHASE_CONFIG = 'purchase_config'
K_PURCHASE_BALANCE = 'purchase_balance'
K_CURRENT_POSITIONS = 'current_positions'
K_BACKTEST_CONFIG = 'backtest_config'
K_BACKTEST_RESULT = 'backtest_result'
K_CAPACITY_INPUTS = 'capacity_inputs'
K_CAPACITY_SCHEDULE = 'capacity_schedule'
K_SCHEDULE_IS_CUSTOM = 'capacity_schedule_is_custom'
K_CAPACITY_METRICS = 'capacity_metrics'
K_CAPACITY_VERDICT = 'capacity_verdict'
K_ACTIVE_SECTION = 'active_section'

# key -> default value created by ``init_state`` on the first run.
STATE_KEYS: dict = {
    K_PARAMS: dict,
    K_FINGERPRINTS: dict,
    K_UPLOADED: list,
    K_OVERRIDES: dict,
    K_PARSE_PREFS: dict,
    K_PARSE_REPORTS: tuple,
    K_MISSING_POLICY: lambda: None,
    K_INITIAL_POSITION: lambda: None,
    K_THRESHOLDS: lambda: None,
    K_ALIGNED: lambda: None,
    K_STATS: lambda: None,
    K_OPTIMIZATION: lambda: None,
    K_PATH_SPEC: lambda: None,
    K_FRONTIER: lambda: None,
    K_CLOUD: lambda: None,
    K_PURCHASE_CONFIG: lambda: None,
    K_PURCHASE_BALANCE: lambda: None,
    K_CURRENT_POSITIONS: lambda: None,
    K_BACKTEST_CONFIG: lambda: None,
    K_BACKTEST_RESULT: lambda: None,
    K_CAPACITY_INPUTS: dict,
    K_CAPACITY_SCHEDULE: lambda: None,
    K_SCHEDULE_IS_CUSTOM: lambda: False,
    K_CAPACITY_METRICS: lambda: None,
    K_CAPACITY_VERDICT: lambda: None,
}

# Session keys that hold a computed result, in dependency order.
RESULT_STATE_KEYS: dict = {
    'aligned_data': K_ALIGNED,
    'asset_stats': K_STATS,
    'optimization': K_OPTIMIZATION,
    'frontier': K_FRONTIER,
    'weight_cloud': K_CLOUD,
    'purchase_balance': K_PURCHASE_BALANCE,
    'backtest_result': K_BACKTEST_RESULT,
    'capacity_metrics': K_CAPACITY_METRICS,
    'capacity_verdict': K_CAPACITY_VERDICT,
}


# --------------------------------------------------------------------------- #
# One fingerprint per result (design.md section 13.1b)
# --------------------------------------------------------------------------- #
FINGERPRINT_DEPS: dict = {
    'aligned_data':     ('files', 'overrides', 'parse_prefs', 'missing_policy', 'period'),
    'asset_stats':      ('aligned_data', 'trading_days', 'rf'),
    'optimization':     ('asset_stats', 'rf', 'allow_short', 'max_weight', 'short_lb',
                         'seed'),
    'frontier':         ('optimization', 'n_frontier_points'),
    'weight_cloud':     ('optimization', 'n_weight_sims', 'seed'),
    'purchase_balance': ('optimization', 'capital_s5', 'pct_fee_s5', 'fixed_fee_s5',
                         'cash_reserve', 'integer_units', 'weight_tolerance',
                         'current_positions'),
    'backtest_result':  ('aligned_data', 'trading_days', 'rf', 'allow_short',
                         'max_weight', 'short_lb', 'window', 'rebalance_rule',
                         'capital_s6', 'cost_rate'),
    'capacity_metrics': ('optimization', 'capital_s7', 'pct_fee_s7', 'fixed_fee_s7',
                         'position_source', 'cash_reserve_initial',
                         'holding_period_days', 'schedule', 'liquidity_policy',
                         'k_sigma', 'mc_method', 'block_size', 'n_paths',
                         'capacity_rebalance_rule', 'seed'),
    'capacity_verdict': ('capacity_metrics', 'capacity_thresholds'),
}

RESULT_KEYS: tuple = tuple(FINGERPRINT_DEPS)

# The one exception to the ownership rule of section 13.1: the analysis period
# is configured in the sidebar and nowhere else.
SIDEBAR_ONLY_PARAMS: tuple = ('period', 'period_start', 'period_end')


def _canonical(value):
    """Turn ``value`` into something JSON can serialise deterministically."""
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not np.isfinite(value):
            return repr(value)
        return value
    if isinstance(value, bytes):
        return hashlib.sha1(value).hexdigest()
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, np.ndarray):
        return ['ndarray', list(value.shape),
                hashlib.sha1(np.ascontiguousarray(value).tobytes()).hexdigest()]
    if isinstance(value, pd.DataFrame):
        return ['frame', [str(column) for column in value.columns],
                hashlib.sha1(
                    pd.util.hash_pandas_object(value, index=True).values.tobytes()
                ).hexdigest()]
    if isinstance(value, (pd.Series, pd.Index)):
        return ['series', hashlib.sha1(
            pd.util.hash_pandas_object(pd.Series(value), index=True).values.tobytes()
        ).hexdigest()]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        to_tuple = getattr(value, 'to_tuple', None)
        if callable(to_tuple):
            return _canonical(tuple(to_tuple()))
        return _canonical(asdict(value))
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(
            value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=repr) if isinstance(value, (set, frozenset)) else value
        return [_canonical(item) for item in items]
    return repr(value)


def compute_fingerprint(result_key: str, params: dict,
                        parents: dict | None = None) -> str:
    """sha1 over the canonical JSON of the declared dependencies only.

    A dependency that names another result (``'asset_stats'``, ``'optimization'``,
    ...) contributes that result's stored fingerprint, read from ``parents``, so
    invalidation propagates transitively without recomputing anything.
    """
    if result_key not in FINGERPRINT_DEPS:
        raise KeyError(result_key)
    parents = parents or {}
    payload: dict = {}
    for dependency in FINGERPRINT_DEPS[result_key]:
        if dependency in FINGERPRINT_DEPS:
            payload[dependency] = parents.get(dependency)
        else:
            payload[dependency] = _canonical((params or {}).get(dependency))
    blob = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=repr)
    return hashlib.sha1(blob.encode('utf-8')).hexdigest()


# --------------------------------------------------------------------------- #
# Session helpers
# --------------------------------------------------------------------------- #
def init_state() -> None:
    """Create every session key with its default on the first run."""
    for key, factory in STATE_KEYS.items():
        if key not in st.session_state:
            st.session_state[key] = factory()


def _parents() -> dict:
    return dict(st.session_state.get(K_FINGERPRINTS, {}) or {})


def store_fingerprint(result_key: str, params: dict) -> str:
    """Record the fingerprint a freshly computed result was born with."""
    value = compute_fingerprint(result_key, params, _parents())
    fingerprints = dict(st.session_state.get(K_FINGERPRINTS, {}) or {})
    fingerprints[result_key] = value
    st.session_state[K_FINGERPRINTS] = fingerprints
    return value


def stored_fingerprint(result_key: str):
    return (st.session_state.get(K_FINGERPRINTS, {}) or {}).get(result_key)


def result_ready(result_key: str) -> bool:
    """True when the session holds a value for ``result_key``."""
    state_key = RESULT_STATE_KEYS.get(result_key)
    if state_key is None:
        return False
    return st.session_state.get(state_key) is not None


def is_stale(result_key: str, params: dict) -> bool:
    """True when the stored result was computed with different inputs (RF-P4).

    A result that was never computed is not stale: the section shows its own
    "compute this first" box instead of an amber badge.
    """
    previous = stored_fingerprint(result_key)
    if previous is None or not result_ready(result_key):
        return False
    return previous != compute_fingerprint(result_key, params, _parents())


def invalidate(*result_keys: str) -> None:
    """Drop the stored value and fingerprint of the given results."""
    fingerprints = dict(st.session_state.get(K_FINGERPRINTS, {}) or {})
    for result_key in result_keys:
        state_key = RESULT_STATE_KEYS.get(result_key)
        if state_key is not None:
            st.session_state[state_key] = None
        fingerprints.pop(result_key, None)
    st.session_state[K_FINGERPRINTS] = fingerprints


def clear_results() -> None:
    """Equivalent of the prototype's ``Clear All``: forget data and results."""
    for key, factory in STATE_KEYS.items():
        if key == K_PARAMS:
            continue
        st.session_state[key] = factory()


def effective_value(params: dict, global_key: str, local_key: str,
                    default=None) -> tuple:
    """Resolve the ownership rule of section 13.1.

    Returns ``(value, is_local)``. The sidebar sets the global default; a
    section may override it with its own widget, and then the local value wins
    and the section paints ``badge('valor local')``. The analysis period is the
    single exception: it is never overridable, so this always returns the
    global value for it.
    """
    global_value = (params or {}).get(global_key, default)
    if global_key in SIDEBAR_ONLY_PARAMS:
        return global_value, False
    if local_key not in st.session_state:
        return global_value, False
    local_value = st.session_state[local_key]
    if local_value is None:
        return global_value, False
    return local_value, local_value != global_value
