"""Data loading and regime preparation utilities for the hedge optimizer."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf

WINDOW_DAYS = 20


def get_dynamic_date_range(years_back: int = 10) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return timezone-aware start/end dates covering ``years_back`` years."""
    today_ny = pd.Timestamp.today(tz="America/New_York").normalize()
    start = today_ny - pd.DateOffset(years=years_back)
    return start.tz_localize(None), today_ny.tz_localize(None)


def _safe_close(df: pd.DataFrame, ticker_col_hint: str) -> pd.Series:
    """Robustly extract a close column from yfinance output."""
    if isinstance(df.columns, pd.MultiIndex):
        for key in [("Close", ticker_col_hint), ("Adj Close", ticker_col_hint)]:
            if key in df.columns:
                return df[key]
        if "Close" in df.columns.get_level_values(0):
            return df.xs("Close", axis=1, level=0).iloc[:, 0]
    for key in ["Close", "Adj Close"]:
        if key in df.columns:
            return df[key]
    raise KeyError("Could not locate a Close column in the downloaded data.")


def load_market_data_simple(
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    window_days: int = WINDOW_DAYS,
) -> pd.DataFrame:
    """Download SPY/VIX history and compute helper columns."""
    spy = yf.download("SPY", start=start_dt, end=end_dt, progress=False, auto_adjust=True)
    vix = yf.download("^VIX", start=start_dt, end=end_dt, progress=False, auto_adjust=False)

    spy_close = _safe_close(spy, "SPY").rename("SPY")
    vix_close = _safe_close(vix, "^VIX").rename("VIX")

    df = pd.concat([spy_close, vix_close], axis=1).dropna()
    df["ret_1d"] = df["SPY"].pct_change()
    df["rv_rolling"] = df["ret_1d"].rolling(window_days).std() * np.sqrt(252)
    return df.dropna()


def compute_regime_thresholds(
    df: pd.DataFrame, q_low: float = 0.33, q_high: float = 0.66
) -> Tuple[float, float]:
    """Return VIX quantile cutoffs used to bucket volatility regimes."""
    vix = df["VIX"].dropna().astype(float)
    lo = float(vix.quantile(q_low))
    hi = float(vix.quantile(q_high))
    if lo >= hi:
        med = float(vix.median())
        lo, hi = med, med + 0.01
    if not np.isfinite(lo) or not np.isfinite(hi):
        med = float(vix.median())
        lo, hi = med, med
    return lo, hi


def add_regime_labels(df: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    """Attach LOW/MID/HIGH regime labels based on VIX quantiles."""
    def _label(vix_val: float) -> str:
        if vix_val <= lo:
            return "LOW"
        if vix_val >= hi:
            return "HIGH"
        return "MID"

    out = df.copy()
    out["regime"] = out["VIX"].astype(float).apply(_label)
    return out


def suggested_regime_today(df_reg: pd.DataFrame) -> str:
    return str(df_reg["regime"].iloc[-1])


def select_regime_pool(
    df_reg: pd.DataFrame, mode: str = "auto", override: str | None = None
) -> Tuple[str, pd.DataFrame]:
    """Return the regime name and subset of rows used for bootstrapping."""
    valid = {"LOW", "MID", "HIGH"}
    if mode == "auto":
        chosen = suggested_regime_today(df_reg)
    else:
        if override is None or str(override).upper() not in valid:
            raise ValueError("Override must be one of: LOW, MID, HIGH")
        chosen = str(override).upper()
    pool = df_reg[df_reg["regime"] == chosen].copy()
    if pool.empty:
        raise ValueError(f"No rows found for regime '{chosen}'.")
    return chosen, pool


def prepare_regime_data(
    years_back: int = 10, q_low: float = 0.33, q_high: float = 0.66
) -> Tuple[pd.DataFrame, float, float]:
    """Load history, compute thresholds, and attach regime labels."""
    start_dt, end_dt = get_dynamic_date_range(years_back)
    df = load_market_data_simple(start_dt, end_dt)
    lo, hi = compute_regime_thresholds(df, q_low=q_low, q_high=q_high)
    df_reg = add_regime_labels(df, lo, hi)
    return df_reg, lo, hi


__all__ = [
    "WINDOW_DAYS",
    "get_dynamic_date_range",
    "load_market_data_simple",
    "compute_regime_thresholds",
    "add_regime_labels",
    "suggested_regime_today",
    "select_regime_pool",
    "prepare_regime_data",
]
