"""Core utilities for the Streamlit hedge optimizer app."""

from .data import (
    get_dynamic_date_range,
    load_market_data_simple,
    compute_regime_thresholds,
    add_regime_labels,
    suggested_regime_today,
    select_regime_pool,
    prepare_regime_data,
    fetch_intraday_quotes,
    build_regime_explanation,
)
from .engine import (
    Prefs,
    validate_quotes_df_bidask_strict,
    run_hedge_workflow,
)

__all__ = [
    "get_dynamic_date_range",
    "load_market_data_simple",
    "compute_regime_thresholds",
    "add_regime_labels",
    "suggested_regime_today",
    "select_regime_pool",
    "prepare_regime_data",
    "fetch_intraday_quotes",
    "build_regime_explanation",
    "Prefs",
    "validate_quotes_df_bidask_strict",
    "run_hedge_workflow",
]
